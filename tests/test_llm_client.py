"""Tests for the Anthropic LLM client wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from kong.llm.client import (
    AnthropicClient,
    ModelTokenUsage,
    TokenUsage,
    _PRICING,
)


def _mock_message(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic message response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    msg = MagicMock()
    msg.content = [block]
    msg.usage = usage
    return msg


class TestAnthropicClient:
    @patch("kong.llm.client.anthropic.Anthropic")
    def test_analyze_function_parses_json(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "parse_config", "confidence": 85, "classification": "parser"}'
        )

        client = AnthropicClient(api_key="test-key")
        response = client.analyze_function("analyze this function")

        assert response.name == "parse_config"
        assert response.confidence == 85
        assert response.classification == "parser"

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_tracks_token_usage(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=200, output_tokens=80
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("prompt1")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 80
        assert client.usage.calls == 1

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_accumulates_across_calls(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=100, output_tokens=50
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 100
        assert client.usage.calls == 2

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_passes_system_prompt_with_cache_control(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message('{"name": "f"}')

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("test prompt")

        call_kwargs = mock_client.messages.create.call_args
        assert "system" in call_kwargs.kwargs
        system_val = call_kwargs.kwargs["system"]
        assert isinstance(system_val, list)
        assert system_val[0]["cache_control"] == {"type": "ephemeral"}
        assert "reverse engineer" in system_val[0]["text"].lower()

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_model_override(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message('{"name": "f"}')

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("prompt", model="claude-haiku-4-5-20251001")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_per_model_cost_tracking(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=100, output_tokens=50
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2", model="claude-haiku-4-5-20251001")

        assert len(client.usage.by_model) == 2
        assert "claude-sonnet-4-20250514" in client.usage.by_model
        assert "claude-haiku-4-5-20251001" in client.usage.by_model
        assert client.usage.calls == 2

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_handles_malformed_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            "I cannot analyze this function because reasons."
        )

        client = AnthropicClient(api_key="test-key")
        response = client.analyze_function("prompt")

        assert response.name == ""
        assert "Failed to parse" in response.reasoning

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_response_has_token_counts(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=500, output_tokens=200
        )

        client = AnthropicClient(api_key="test-key")
        response = client.analyze_function("prompt")

        assert response.input_tokens == 500
        assert response.output_tokens == 200


class TestModelTokenUsage:
    def test_cost_calculation(self):
        u = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("claude-sonnet-4-20250514")
        assert cost == 3.0 + 15.0

    def test_cost_with_unknown_model_uses_default(self):
        u = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("unknown-model")
        assert cost == 3.0 + 15.0

    def test_cache_token_costs(self):
        u = ModelTokenUsage(
            input_tokens=0, output_tokens=0,
            cache_creation_tokens=1_000_000, cache_read_tokens=1_000_000,
        )
        cost = u.cost_usd("claude-sonnet-4-20250514")
        assert cost == (3.0 * 1.25) + (3.0 * 0.10)


class TestTokenUsage:
    def test_aggregate_properties(self):
        u = TokenUsage()
        u._get("model_a").input_tokens = 100
        u._get("model_a").output_tokens = 50
        u._get("model_b").input_tokens = 200
        u._get("model_b").output_tokens = 80
        u._get("model_a").calls = 1
        u._get("model_b").calls = 2

        assert u.input_tokens == 300
        assert u.output_tokens == 130
        assert u.total_tokens == 430
        assert u.calls == 3

    def test_total_cost_across_models(self):
        u = TokenUsage()
        u._get("claude-sonnet-4-20250514").input_tokens = 1_000_000
        u._get("claude-sonnet-4-20250514").output_tokens = 0
        u._get("claude-haiku-4-5-20251001").input_tokens = 1_000_000
        u._get("claude-haiku-4-5-20251001").output_tokens = 0

        assert u.total_cost_usd == 3.0 + 1.0


class TestPricingValues:
    def test_pricing_entries_exist(self):
        assert "claude-sonnet-4-20250514" in _PRICING
        assert "claude-haiku-4-5-20251001" in _PRICING

    def test_haiku_cheaper_than_sonnet(self):
        haiku_in, haiku_out = _PRICING["claude-haiku-4-5-20251001"]
        sonnet_in, sonnet_out = _PRICING["claude-sonnet-4-20250514"]
        assert haiku_in < sonnet_in
        assert haiku_out < sonnet_out


class TestAnalyzeFunctionBatch:
    def test_batch_returns_list_of_responses(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text=json.dumps([
            {"address": "0x1000", "name": "foo", "confidence": 80},
            {"address": "0x2000", "name": "bar", "confidence": 70},
        ]))]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        results = client.analyze_function_batch("batch prompt", model="claude-haiku-4-5-20251001")

        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"

    def test_batch_uses_batch_system_prompt(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text='[{"name": "f", "confidence": 50}]')]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt")

        call_kwargs = mock_anthropic.messages.create.call_args
        system_text = call_kwargs.kwargs["system"][0]["text"]
        assert "multiple" in system_text.lower() or "batch" in system_text.lower()

    def test_batch_records_usage(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text='[{"name": "f", "confidence": 50}]')]
        mock_message.usage = MagicMock(
            input_tokens=500, output_tokens=200,
            cache_creation_input_tokens=100, cache_read_input_tokens=50,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt")

        assert client.usage.input_tokens == 500
        assert client.usage.output_tokens == 200
        assert client.usage.calls == 1
