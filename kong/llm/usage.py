"""Token usage tracking for LLM API calls."""

from __future__ import annotations

from dataclasses import dataclass, field

_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


@dataclass
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def cost_usd(self, model: str) -> float:
        input_rate, output_rate = _PRICING.get(model, (3.0, 15.0))
        cache_write_rate = input_rate * 1.25
        cache_read_rate = input_rate * 0.10
        return (
            (self.input_tokens / 1_000_000) * input_rate
            + (self.output_tokens / 1_000_000) * output_rate
            + (self.cache_creation_tokens / 1_000_000) * cache_write_rate
            + (self.cache_read_tokens / 1_000_000) * cache_read_rate
        )


@dataclass
class TokenUsage:
    by_model: dict[str, ModelTokenUsage] = field(default_factory=dict)

    def _get(self, model: str) -> ModelTokenUsage:
        if model not in self.by_model:
            self.by_model[model] = ModelTokenUsage()
        return self.by_model[model]

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(m.cost_usd(model) for model, m in self.by_model.items())
