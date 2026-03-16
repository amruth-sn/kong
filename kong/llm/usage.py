"""Token usage tracking for LLM API calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PricingTier:
    """Per-model pricing in USD per 1M tokens.

    Rates sourced from published Anthropic/OpenAI API pricing pages.
    """

    input_rate: float
    output_rate: float
    cache_write_rate: float = 0.0
    cache_read_rate: float = 0.0


_DEFAULT_PRICING = PricingTier(input_rate=3.0, output_rate=15.0)

PRICING_REGISTRY: dict[str, PricingTier] = {
    # Anthropic: cache_write = input * 1.25, cache_read = input * 0.10
    "claude-opus-4-6": PricingTier(
        5.0, 25.0, cache_write_rate=6.25, cache_read_rate=0.50
    ),
    "claude-sonnet-4-6": PricingTier(
        3.0, 15.0, cache_write_rate=3.75, cache_read_rate=0.30
    ),
    "claude-sonnet-4-20250514": PricingTier(
        3.0, 15.0, cache_write_rate=3.75, cache_read_rate=0.30
    ),
    "claude-haiku-4-5-20251001": PricingTier(
        1.0, 5.0, cache_write_rate=1.25, cache_read_rate=0.10
    ),
    # OpenAI: cached input is 50% of input rate, no cache write billing
    "gpt-4o": PricingTier(2.50, 10.00, cache_read_rate=1.25),
    "gpt-4o-2024-11-20": PricingTier(2.50, 10.00, cache_read_rate=1.25),
    "gpt-4o-mini": PricingTier(0.15, 0.60, cache_read_rate=0.075),
    "gpt-4o-mini-2024-07-18": PricingTier(0.15, 0.60, cache_read_rate=0.075),
    "o1": PricingTier(15.00, 60.00, cache_read_rate=7.50),
    "o3-mini": PricingTier(1.10, 4.40, cache_read_rate=0.55),
}


def get_pricing(model: str) -> PricingTier:
    """Look up pricing for a model, falling back to a sensible default."""
    return PRICING_REGISTRY.get(model, _DEFAULT_PRICING)


@dataclass
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def cost_usd(self, model: str) -> float:
        tier = get_pricing(model)
        return (
            (self.input_tokens / 1_000_000) * tier.input_rate
            + (self.output_tokens / 1_000_000) * tier.output_rate
            + (self.cache_creation_tokens / 1_000_000) * tier.cache_write_rate
            + (self.cache_read_tokens / 1_000_000) * tier.cache_read_rate
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
