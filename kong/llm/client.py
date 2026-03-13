"""Anthropic SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.  Supports both simple
(single-shot JSON) and tool-use (agentic loop) interactions.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT, OUTPUT_SCHEMA, SYSTEM_PROMPT
from kong.llm.tools import ToolExecutor
from kong.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-6"


class AnthropicClient:
    """Concrete LLM client using the Anthropic SDK.

    Satisfies the LLMClient protocol from kong.agent.analyzer.

    Usage::

        client = AnthropicClient()
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
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=5)
        self.usage = TokenUsage()

    def analyze_function(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Send an analysis prompt and return parsed response (no tools)."""
        effective_model = model or self.model
        message = self._client.messages.create(
            model=effective_model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": f"{SYSTEM_PROMPT}\n\n{OUTPUT_SCHEMA}",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_text = self._extract_text(message)
        self._record_usage(message, effective_model)

        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = message.usage.input_tokens
        response.output_tokens = message.usage.output_tokens
        response.raw = raw_text
        return response

    def analyze_function_batch(self, prompt: str, *, model: str | None = None) -> list[LLMResponse]:
        """Send a batch analysis prompt and return parsed list of responses."""
        effective_model = model or self.model
        message = self._client.messages.create(
            model=effective_model,
            max_tokens=16384,
            system=[{
                "type": "text",
                "text": f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = self._extract_text(message)
        self._record_usage(message, effective_model)

        responses = Analyzer.parse_llm_json_batch(raw_text)
        for resp in responses:
            resp.input_tokens = message.usage.input_tokens
            resp.output_tokens = message.usage.output_tokens
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
        ``tool_use`` blocks, executes each tool via *tool_executor* and
        feeds results back.  Repeats until the model returns a final text
        response or *max_rounds* is exhausted.
        """
        cached_system = [{
            "type": "text",
            "text": f"{system}\n\n{OUTPUT_SCHEMA}",
            "cache_control": {"type": "ephemeral"},
        }]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        total_input = 0
        total_output = 0

        for _ in range(max_rounds):
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=cached_system,
                tools=tools,
                messages=messages,
            )

            total_input += message.usage.input_tokens
            total_output += message.usage.output_tokens
            self._record_usage(message, self.model)

            if message.stop_reason != "tool_use":
                raw_text = self._extract_text(message)
                response = Analyzer.parse_llm_json(raw_text)
                response.input_tokens = total_input
                response.output_tokens = total_output
                response.raw = raw_text
                return response

            messages.append({"role": "assistant", "content": message.content})

            tool_results: list[dict[str, Any]] = []
            for block in message.content:
                if block.type != "tool_use":
                    continue
                result_str = tool_executor.execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        raw_text = self._extract_text(message)
        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = total_input
        response.output_tokens = total_output
        response.raw = raw_text
        return response

    @property
    def total_cost_usd(self) -> float:
        return self.usage.total_cost_usd

    def _extract_text(self, message: Any) -> str:
        parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def _record_usage(self, message: Any, model: str | None = None) -> None:
        effective_model = model or self.model
        usage = message.usage
        mu = self.usage._get(effective_model)
        mu.input_tokens += usage.input_tokens
        mu.output_tokens += usage.output_tokens
        mu.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        mu.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        mu.calls += 1
        logger.debug(
            "LLM [%s]: %d in / %d out / %d cache_write / %d cache_read tokens "
            "(total: %d calls, $%.4f)",
            effective_model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            self.usage.calls,
            self.usage.total_cost_usd,
        )
