"""OpenAI SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.  Supports both simple
(single-shot JSON) and tool-use (agentic loop) interactions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import (
    BATCH_OUTPUT_SCHEMA,
    BATCH_SYSTEM_PROMPT,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
)
from kong.llm.tools import ToolExecutor
from kong.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


class OpenAIClient:
    """Concrete LLM client using the OpenAI SDK.

    Satisfies the LLMClient protocol from kong.agent.analyzer.

    Usage::

        client = OpenAIClient()
        response = client.analyze_function(prompt)
        response = client.analyze_with_tools(prompt, system, tools, executor)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key, max_retries=5)
        self.usage = TokenUsage()

    def analyze_function(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Send an analysis prompt and return parsed response (no tools)."""
        effective_model = model or self.model
        response = self._client.chat.completions.create(
            model=effective_model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{OUTPUT_SCHEMA}"},
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = response.choices[0].message.content or ""
        self._record_usage(response, effective_model)

        result = Analyzer.parse_llm_json(raw_text)
        result.input_tokens = response.usage.prompt_tokens
        result.output_tokens = response.usage.completion_tokens
        result.raw = raw_text
        return result

    def analyze_function_batch(
        self, prompt: str, *, model: str | None = None
    ) -> list[LLMResponse]:
        """Send a batch analysis prompt and return parsed list of responses."""
        effective_model = model or self.model
        response = self._client.chat.completions.create(
            model=effective_model,
            max_tokens=16384,
            messages=[
                {
                    "role": "system",
                    "content": f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = response.choices[0].message.content or ""
        self._record_usage(response, effective_model)

        responses = Analyzer.parse_llm_json_batch(raw_text)
        for resp in responses:
            resp.input_tokens = response.usage.prompt_tokens
            resp.output_tokens = response.usage.completion_tokens
        return responses

    def analyze_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        max_rounds: int = 10,
    ) -> LLMResponse:
        """Run an agentic tool-use loop.

        Sends the prompt with tool definitions.  When the model returns
        tool_calls, executes each tool via *tool_executor* and feeds results
        back.  Repeats until the model returns a final text response or
        *max_rounds* is exhausted.
        """
        openai_tools = _convert_tools_to_openai(tools)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{system}\n\n{OUTPUT_SCHEMA}"},
            {"role": "user", "content": prompt},
        ]

        total_input = 0
        total_output = 0

        last_text = ""
        for _ in range(max_rounds):
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=openai_tools,
                messages=messages,
            )

            usage = response.usage
            total_input += usage.prompt_tokens
            total_output += usage.completion_tokens
            self._record_usage(response, self.model)

            choice = response.choices[0]
            last_text = choice.message.content or ""

            if choice.finish_reason != "tool_calls":
                result = Analyzer.parse_llm_json(last_text)
                result.input_tokens = total_input
                result.output_tokens = total_output
                result.raw = last_text
                return result

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in choice.message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result_str = tool_executor.execute(tool_call.function.name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

        result = Analyzer.parse_llm_json(last_text)
        result.input_tokens = total_input
        result.output_tokens = total_output
        result.raw = last_text
        return result

    @property
    def total_cost_usd(self) -> float:
        return self.usage.total_cost_usd

    def _record_usage(self, response: Any, model: str | None = None) -> None:
        effective_model = model or self.model
        usage = response.usage
        mu = self.usage._get(effective_model)
        mu.input_tokens += usage.prompt_tokens
        mu.output_tokens += usage.completion_tokens
        cached = (
            getattr(
                getattr(usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            or 0
        )
        mu.cache_read_tokens += cached
        mu.calls += 1
        logger.debug(
            "LLM [%s]: %d in / %d out / %d cached tokens (total: %d calls, $%.4f)",
            effective_model,
            usage.prompt_tokens,
            usage.completion_tokens,
            cached,
            self.usage.calls,
            self.usage.total_cost_usd,
        )
