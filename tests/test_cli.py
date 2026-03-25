"""Tests for the Kong CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

import click

from kong.__main__ import _NOT_NEEDED_STR, cli, create_llm_client, resolve_provider, validate_base_url
from kong.config import LLMConfig, LLMProvider
from kong.db import get_custom_config, save_setup


def _complete_setup(tmp_path, monkeypatch):
    """Mark setup as complete with Anthropic as default."""
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    save_setup(
        enabled=[LLMProvider.ANTHROPIC],
        default=LLMProvider.ANTHROPIC,
    )


def test_analyze_missing_binary(tmp_path, monkeypatch):
    _complete_setup(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "/nonexistent/binary"])
    assert result.exit_code != 0


def test_analyze_no_ghidra_installed(tmp_path, monkeypatch):
    """When Ghidra is not installed, show install instructions."""
    _complete_setup(tmp_path, monkeypatch)
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}), \
         patch("kong.config.find_ghidra_install", return_value=None), \
         patch("kong.llm.probe.probe_endpoint", return_value=True):
        result = runner.invoke(cli, ["analyze", str(binary)])

    assert result.exit_code != 0
    assert "not installed" in result.output.lower() or "not found" in result.output.lower()
    assert "brew install ghidra" in result.output


def test_setup_wizard_saves_config(tmp_path, monkeypatch):
    """Setup wizard persists provider selection."""
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")

    runner = CliRunner()
    result = runner.invoke(cli, ["setup"], input="1\n")

    assert result.exit_code == 0
    assert "sk-ant-" in result.output
    assert "..." in result.output


def test_eval_with_test_data(tmp_path):
    """Run eval against a simple test case."""
    import json

    analysis = {
        "binary": {"name": "test"},
        "stats": {"llm_calls": 5, "duration_seconds": 10.0, "cost_usd": 0.05},
        "functions": [
            {"name": "hash_string", "signature": "uint hash_string(byte *str)", "confidence": 95, "address": "0x1000"},
        ],
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))

    source = "unsigned int hash_string(const char *s) {\n    return 0;\n}\n"
    source_path = tmp_path / "test.c"
    source_path.write_text(source)

    runner = CliRunner()
    result = runner.invoke(cli, ["eval", str(analysis_path), str(source_path)])
    assert result.exit_code == 0
    assert "hash_string" in result.output
    assert "Symbol Accuracy" in result.output


class TestBannerCustomProvider:
    def test_env_vars_does_not_have_custom(self):
        from kong.banner import _ENV_VARS

        assert LLMProvider.CUSTOM not in _ENV_VARS

    def test_check_api_key_returns_true_for_custom(self):
        from kong.banner import check_api_key

        assert check_api_key(LLMProvider.CUSTOM) is True


class TestCreateLLMClient:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_custom_returns_openai_client_with_base_url(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
        )
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            max_retries=5,
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_custom_no_auth_passes_empty_string(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
        )
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        mock_openai_cls.assert_called_once_with(
            api_key=_NOT_NEEDED_STR,
            base_url="http://localhost:11434/v1",
            max_retries=5,
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_openai_returns_openai_client_no_base_url(self, mock_openai_cls, tmp_path, monkeypatch):
        from kong.llm.openai_client import OpenAIClient

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        mock_openai_cls.assert_called_once_with(
            api_key=None,
            base_url=None,
            max_retries=5,
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_openai_uses_codex_oauth_when_api_key_missing(self, mock_openai_cls, tmp_path, monkeypatch):
        from kong.llm.openai_client import OpenAIClient

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(
            '{"OPENAI_API_KEY": null, "tokens": {"access_token": "oauth-token"}}'
        )

        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        mock_openai_cls.assert_called_once_with(
            api_key="oauth-token",
            base_url=None,
            max_retries=5,
        )

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_anthropic_returns_anthropic_client(self, mock_anthropic_cls):
        from kong.llm.client import AnthropicClient

        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-6")
        client = create_llm_client(config)
        assert isinstance(client, AnthropicClient)

    @patch("kong.__main__.CodexClient")
    def test_codex_returns_codex_client(self, mock_codex_cls):
        config = LLMConfig(provider=LLMProvider.CODEX, model="gpt-5-codex", max_output_tokens=2048)
        client = create_llm_client(config)
        assert client is mock_codex_cls.return_value
        mock_codex_cls.assert_called_once_with(
            model="gpt-5-codex",
            max_output_tokens=2048,
        )


class TestResolveProviderCustom:
    def test_base_url_implies_custom(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        provider = resolve_provider(base_url="http://localhost:8000/v1")
        assert provider is LLMProvider.CUSTOM

    def test_explicit_custom_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        provider = resolve_provider(cli_override="custom")
        assert provider is LLMProvider.CUSTOM

    def test_custom_skipped_in_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        save_setup(
            enabled=[LLMProvider.CUSTOM, LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        provider = resolve_provider()
        assert provider is LLMProvider.ANTHROPIC

    def test_codex_can_be_selected_explicitly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        codex_home = tmp_path / "codex-home"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "auth.json").write_text(
            '{"tokens":{"access_token":"oauth-token","refresh_token":"refresh-token","account_id":"acct_123"}}'
        )
        provider = resolve_provider(cli_override="codex")
        assert provider is LLMProvider.CODEX


class TestValidateBaseUrl:
    def test_valid_http_url(self):
        assert validate_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_valid_https_url(self):
        assert validate_base_url("https://api.together.xyz/v1") == "https://api.together.xyz/v1"

    def test_strips_trailing_slash(self):
        assert validate_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"

    def test_rejects_missing_scheme(self):
        import pytest

        with pytest.raises(click.BadParameter):
            validate_base_url("localhost:8000/v1")


class TestSetupWizardCustom:
    @patch("kong.llm.probe.probe_endpoint", return_value=False)
    def test_setup_custom_provider(self, mock_probe, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["setup"],
            input="4\nhttp://localhost:11434/v1\nllama3:8b\n\n32000\n20\n4096\n",
        )
        assert result.exit_code == 0
        from kong.db import get_default_provider

        assert get_default_provider() == LLMProvider.CUSTOM
        cfg = get_custom_config()
        assert cfg["custom_base_url"] == "http://localhost:11434/v1"
        assert cfg["custom_model"] == "llama3:8b"
        assert cfg["custom_max_prompt_chars"] == "32000"

    def test_setup_option_5_is_anthropic_plus_openai(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="5\n1\n")
        assert result.exit_code == 0
        from kong.db import get_enabled_providers

        enabled = get_enabled_providers()
        assert LLMProvider.ANTHROPIC in enabled
        assert LLMProvider.OPENAI in enabled
        assert LLMProvider.CUSTOM not in enabled

    def test_setup_option_3_is_codex(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        codex_home = tmp_path / "codex-home"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "auth.json").write_text(
            '{"tokens":{"access_token":"oauth-token","refresh_token":"refresh-token","account_id":"acct_123"}}'
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="3\n")
        assert result.exit_code == 0
        from kong.db import get_default_provider

        assert get_default_provider() is LLMProvider.CODEX


class TestCustomProviderIntegration:
    @patch("kong.llm.probe.probe_endpoint", return_value=True)
    @patch("kong.llm.openai_client.openai.OpenAI")
    @patch("kong.config.find_ghidra_install", return_value=None)
    def test_analyze_with_base_url_flag(
        self, mock_ghidra, mock_openai_cls, mock_probe, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(enabled=[LLMProvider.CUSTOM], default=LLMProvider.CUSTOM)

        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", str(binary),
            "--base-url", "http://localhost:11434/v1",
            "--model", "llama3:8b",
            "--headless",
        ])

        assert "not installed" in result.output.lower() or "not found" in result.output.lower()
