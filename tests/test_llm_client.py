"""Tests for the Anthropic LLM client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kong.llm.client import AnthropicClient, TokenUsage, _PRICING


def _mock_message(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic message response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

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
    def test_passes_system_prompt(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message('{"name": "f"}')

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("test prompt")

        call_kwargs = mock_client.messages.create.call_args
        assert "system" in call_kwargs.kwargs
        assert "reverse engineer" in call_kwargs.kwargs["system"].lower()

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


class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_cost_calculation(self):
        u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("claude-sonnet-4-20250514")
        assert cost == 3.0 + 15.0

    def test_cost_with_unknown_model_uses_default(self):
        u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("unknown-model")
        assert cost == 3.0 + 15.0


class TestPricingValues:
    def test_pricing_entries_exist(self):
        assert "claude-sonnet-4-20250514" in _PRICING
        assert "claude-haiku-4-20250414" in _PRICING

    def test_haiku_cheaper_than_sonnet(self):
        haiku_in, haiku_out = _PRICING["claude-haiku-4-20250414"]
        sonnet_in, sonnet_out = _PRICING["claude-sonnet-4-20250514"]
        assert haiku_in < sonnet_in
        assert haiku_out < sonnet_out
