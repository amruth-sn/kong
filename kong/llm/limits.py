"""Model-specific limits and rate limiting for LLM API calls."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLimits:
    """Context window and chunking parameters for a specific model.

    max_prompt_chars: safe upper bound on prompt length in characters,
        leaving room for system prompt and output within the context window.
    max_chunk_functions: cap on how many functions to pack into a single
        batch call (prevents the LLM from losing track of results).
    max_output_tokens: maximum output tokens for batch calls.
    """

    max_prompt_chars: int
    max_chunk_functions: int
    max_output_tokens: int


_DEFAULT_LIMITS = ModelLimits(
    max_prompt_chars=400_000,
    max_chunk_functions=120,
    max_output_tokens=16384,
)

MODEL_LIMITS: dict[str, ModelLimits] = {
    # Anthropic — 200k token context (~800k chars) so 400k chars is a safe prompt cap.
    "claude-opus-4-6": _DEFAULT_LIMITS,
    "claude-sonnet-4-6": _DEFAULT_LIMITS,
    "claude-sonnet-4-20250514": _DEFAULT_LIMITS,
    "claude-haiku-4-5-20251001": _DEFAULT_LIMITS,
    # OpenAI — 128k token context (~512k chars) so 350k chars leaves room for overhead.
    "gpt-4o": ModelLimits(350_000, 80, 16384),
    "gpt-4o-2024-11-20": ModelLimits(350_000, 80, 16384),
    "gpt-4o-mini": ModelLimits(350_000, 80, 16384),
    "gpt-4o-mini-2024-07-18": ModelLimits(350_000, 80, 16384),
    # OpenAI reasoning models — 200k context so more expensive, smaller batches.
    "o1": ModelLimits(400_000, 40, 32768),
    "o3-mini": ModelLimits(400_000, 60, 32768),
}


def get_model_limits(model: str) -> ModelLimits:
    """Look up chunking limits for a model, falling back to defaults."""
    return MODEL_LIMITS.get(model, _DEFAULT_LIMITS)


class RateLimiter:
    """Token-bucket style rate limiter for API calls.

    Tracks request timestamps and sleeps proactively to stay below the
    configured requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._token_log: list[tuple[float, int]] = []

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Block until it's safe to make the next request."""
        while True:
            with self._lock:
                now = time.monotonic()
                window_start = now - 60.0
                wait = 0.0

                if self._rpm is not None:
                    self._request_times = [
                        t for t in self._request_times if t > window_start
                    ]
                    if len(self._request_times) >= self._rpm:
                        sleep_until = self._request_times[0] + 60.0
                        wait = max(wait, sleep_until - now)

                if self._tpm is not None and estimated_tokens > 0:
                    self._token_log = [
                        (t, n) for t, n in self._token_log if t > window_start
                    ]
                    total_tokens = sum(n for _, n in self._token_log)
                    if total_tokens + estimated_tokens > self._tpm:
                        sleep_until = (
                            self._token_log[0][0] + 60.0 if self._token_log else now
                        )
                        wait = max(wait, sleep_until - now)

                if wait <= 0:
                    return

            # Sleep outside the lock so other threads aren't blocked.
            time.sleep(wait)

    def record_request(self, tokens_used: int = 0) -> None:
        """Record that a request was made."""
        with self._lock:
            now = time.monotonic()
            self._request_times.append(now)
            if tokens_used > 0:
                self._token_log.append((now, tokens_used))
