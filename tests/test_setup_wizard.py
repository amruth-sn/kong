"""Tests for the interactive setup wizard and provider routing."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from kong.__main__ import cli
from kong.config import LLMProvider
from kong.db import get_default_provider, get_enabled_providers, is_setup_complete


class TestSetupWizard:
    def test_single_provider_anthropic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")

        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="1\n")

        assert result.exit_code == 0
        assert is_setup_complete()
        assert get_default_provider() is LLMProvider.ANTHROPIC
        assert get_enabled_providers() == [LLMProvider.ANTHROPIC]

    def test_single_provider_openai(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test1234")

        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="2\n")

        assert result.exit_code == 0
        assert get_default_provider() is LLMProvider.OPENAI
        assert get_enabled_providers() == [LLMProvider.OPENAI]

    def test_both_providers_default_anthropic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test1234")

        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="4\n1\n")

        assert result.exit_code == 0
        assert get_enabled_providers() == [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
        assert get_default_provider() is LLMProvider.ANTHROPIC

    def test_both_providers_default_openai(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test1234")

        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="4\n2\n")

        assert result.exit_code == 0
        assert get_default_provider() is LLMProvider.OPENAI

    def test_shows_missing_key_instructions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        runner = CliRunner()
        with patch("kong.banner._load_dotenv"):
            result = runner.invoke(cli, ["setup"], input="1\n")

        assert result.exit_code == 0
        assert "Not set" in result.output
        assert "console.anthropic.com" in result.output

    def test_shows_found_key_masked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-reallylong1234")

        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="1\n")

        assert result.exit_code == 0
        assert "Found" in result.output
        assert "sk-ant-" in result.output
        assert "1234" in result.output

    def test_rerun_setup_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test1234")

        runner = CliRunner()
        runner.invoke(cli, ["setup"], input="1\n")
        assert get_default_provider() is LLMProvider.ANTHROPIC

        runner.invoke(cli, ["setup"], input="2\n")
        assert get_default_provider() is LLMProvider.OPENAI


class TestAnalyzeSetupGate:
    def test_analyze_fails_without_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", str(binary)])

        assert result.exit_code != 0
        assert "setup" in result.output.lower()

    def test_analyze_with_explicit_provider(self, tmp_path, monkeypatch):
        """--provider flag works when setup is complete and key is available."""
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
        from kong.db import save_setup
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.ANTHROPIC,
        )

        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        with patch("kong.config.find_ghidra_install", return_value=None), \
             patch("kong.llm.probe.probe_endpoint", return_value=True):
            result = runner.invoke(
                cli, ["analyze", str(binary), "--provider", "openai"]
            )

        assert "not installed" in result.output.lower() or "not found" in result.output.lower()

    def test_analyze_falls_back_to_available_key(self, tmp_path, monkeypatch):
        """When default provider key is missing, falls back to another enabled provider."""
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
        from kong.db import save_setup
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.ANTHROPIC,
        )

        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        with patch("kong.config.find_ghidra_install", return_value=None), \
             patch("kong.llm.probe.probe_endpoint", return_value=True):
            result = runner.invoke(cli, ["analyze", str(binary)])

        assert "not installed" in result.output.lower() or "not found" in result.output.lower()

    def test_analyze_fails_when_no_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from kong.db import save_setup
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )

        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        with patch("kong.banner._load_dotenv"):
            result = runner.invoke(cli, ["analyze", str(binary)])

        assert result.exit_code != 0
        assert "no api keys" in result.output.lower() or "setup" in result.output.lower()
