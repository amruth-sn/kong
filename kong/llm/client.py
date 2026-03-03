"""Anthropic SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import anthropic

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT

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

    Usage:
        client = AnthropicClient()
        response = client.analyze_function(prompt)
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
        """Send an analysis prompt and return parsed response."""
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

        raw_text = ""
        for block in message.content:
            if block.type == "text":
                raw_text += block.text

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens

        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.calls += 1

        logger.debug(
            "LLM call: %d in / %d out tokens (total: %d calls, $%.4f)",
            input_tokens, output_tokens,
            self.usage.calls, self.usage.cost_usd(self.model),
        )

        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = input_tokens
        response.output_tokens = output_tokens
        response.raw = raw_text
        return response

    @property
    def total_cost_usd(self) -> float:
        return self.usage.cost_usd(self.model)
