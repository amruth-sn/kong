from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import openai

from kong.config import LLMConfig, LLMProvider
from kong.llm.probe import probe_endpoint


class TestProbeCustom:
    @patch("kong.llm.probe.httpx.get")
    def test_custom_probe_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"data": []})
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
        )
        assert probe_endpoint(config) is True
        mock_get.assert_called_once_with(
            "http://localhost:11434/v1/models",
            timeout=10.0,
        )

    @patch("kong.llm.probe.httpx.get")
    def test_custom_probe_connection_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
        )
        assert probe_endpoint(config) is False

    @patch("kong.llm.probe.httpx.get")
    def test_custom_probe_timeout(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ReadTimeout("timed out")
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
        )
        assert probe_endpoint(config) is False

    @patch("kong.llm.probe.httpx.get")
    def test_custom_probe_non_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:11434/v1",
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
