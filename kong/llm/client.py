"""Anthropic SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.  Supports both simple
(single-shot JSON) and tool-use (agentic loop) interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import anthropic

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT
from kong.llm.tools import ToolExecutor

logger = logging.getLogger(__name__)

# Pricing per million tokens (input/output) by model.
# Source: https://docs.anthropic.com/en/docs/about-claude/models
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-20250414": (0.80, 4.0),
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str) -> float:
        input_rate, output_rate = _PRICING.get(model, (3.0, 15.0))
        return (
            (self.input_tokens / 1_000_000) * input_rate
            + (self.output_tokens / 1_000_000) * output_rate
        )


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
        self._client = anthropic.Anthropic(api_key=api_key)
        self.usage = TokenUsage()

    def analyze_function(self, prompt: str) -> LLMResponse:
        """Send an analysis prompt and return parsed response (no tools)."""
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{OUTPUT_SCHEMA}",
                },
            ],
        )

        raw_text = self._extract_text(message)
        self._record_usage(message)

        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = message.usage.input_tokens
        response.output_tokens = message.usage.output_tokens
        response.raw = raw_text
        return response

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
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"{prompt}\n\n{OUTPUT_SCHEMA}"},
        ]

        total_input = 0
        total_output = 0

        for _ in range(max_rounds):
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )

            total_input += message.usage.input_tokens
            total_output += message.usage.output_tokens
            self._record_usage(message)

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
        return self.usage.cost_usd(self.model)

    def _extract_text(self, message: Any) -> str:
        parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def _record_usage(self, message: Any) -> None:
        self.usage.input_tokens += message.usage.input_tokens
        self.usage.output_tokens += message.usage.output_tokens
        self.usage.calls += 1
        logger.debug(
            "LLM call: %d in / %d out tokens (total: %d calls, $%.4f)",
            message.usage.input_tokens,
            message.usage.output_tokens,
            self.usage.calls,
            self.usage.cost_usd(self.model),
        )
