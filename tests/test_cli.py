"""Tests for the Kong CLI."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from kong.__main__ import cli
from kong.config import LLMProvider
from kong.db import save_setup


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
         patch("kong.config.find_ghidra_install", return_value=None):
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
