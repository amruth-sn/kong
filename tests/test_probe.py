from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import openai

from kong.config import LLMConfig, LLMProvider
from kong.llm.probe import probe_endpoint


class TestProbeCustom:
    @patch("kong.llm.probe.openai.OpenAI")
    def test_custom_probe_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
        )
        assert probe_endpoint(config) is True
        mock_openai_cls.assert_called_once_with(
            api_key="not-needed",
            base_url="http://localhost:11434/v1",
        )

    @patch("kong.llm.probe.openai.OpenAI")
    def test_custom_probe_with_api_key(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
        )
        assert probe_endpoint(config) is True
        mock_openai_cls.assert_called_once_with(
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )

    @patch("kong.llm.probe.openai.OpenAI")
    def test_custom_probe_connection_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = openai.APIConnectionError(
            request=MagicMock(),
        )
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
        )
        assert probe_endpoint(config) is False

    @patch("kong.llm.probe.openai.OpenAI")
    def test_custom_probe_non_openai_endpoint(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = openai.APIStatusError(
            message="not found",
            response=MagicMock(status_code=404, reason_phrase="Not Found"),
            body=None,
        )
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="https://example.com",
        )
        assert probe_endpoint(config) is False

    @patch("kong.llm.probe.openai.OpenAI")
    def test_custom_probe_auth_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = openai.AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401),
            body=None,
        )
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="https://openrouter.ai/api/v1",
            api_key="bad-key",
        )
        assert probe_endpoint(config) is False


class TestProbeOpenAI:
    @patch("kong.llm.probe.openai.OpenAI")
    def test_openai_probe_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key="sk-test")
        assert probe_endpoint(config) is True
        mock_openai_cls.assert_called_once_with(api_key="sk-test", base_url=None)

    @patch("kong.llm.probe.openai.OpenAI")
    def test_openai_probe_passes_base_url(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="sk-test",
            base_url="https://my-proxy.example.com/v1",
        )
        assert probe_endpoint(config) is True
        mock_openai_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://my-proxy.example.com/v1",
        )

    @patch("kong.llm.probe.openai.OpenAI")
    def test_openai_probe_auth_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = openai.AuthenticationError(
            message="invalid api key",
            response=MagicMock(status_code=401),
            body=None,
        )
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key="bad-key")
        assert probe_endpoint(config) is False


class TestProbeAnthropic:
    @patch("kong.llm.probe.anthropic.Anthropic")
    def test_anthropic_probe_success(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="sk-ant-test")
        assert probe_endpoint(config) is True

    @patch("kong.llm.probe.anthropic.Anthropic")
    def test_anthropic_probe_auth_failure(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.models.list.side_effect = anthropic.AuthenticationError(
            message="invalid api key",
            response=MagicMock(status_code=401),
            body=None,
        )
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="bad-key")
        assert probe_endpoint(config) is False
