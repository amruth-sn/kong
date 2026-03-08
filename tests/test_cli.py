"""Tests for the Kong CLI."""

from __future__ import annotations

from unittest.mock import patch


from click.testing import CliRunner

from kong.__main__ import cli


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "kong" in result.output.lower()
    assert "0.1.0" in result.output


def test_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "autonomous binary analysis" in result.output.lower()


def test_analyze_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "BINARY" in result.output


def test_analyze_missing_binary():
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "/nonexistent/binary"])
    assert result.exit_code != 0


def test_analyze_no_api_key(tmp_path):
    """When ANTHROPIC_API_KEY is not set, show setup instructions."""
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    with patch.dict("os.environ", {}, clear=True), \
         patch("kong.__main__.check_api_key", return_value=False):
        result = runner.invoke(cli, ["analyze", str(binary)])

    assert result.exit_code != 0
    assert "anthropic_api_key" in result.output.lower()
    assert "kong setup" in result.output.lower()


def test_analyze_no_ghidra_installed(tmp_path):
    """When Ghidra is not installed, show install instructions."""
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}), \
         patch("kong.config.find_ghidra_install", return_value=None):
        result = runner.invoke(cli, ["analyze", str(binary)])

    assert result.exit_code != 0
    assert "not installed" in result.output.lower() or "not found" in result.output.lower()
    assert "brew install ghidra" in result.output


def test_bare_kong_shows_banner():
    """Running 'kong' with no subcommand shows banner and usage."""
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert result.exit_code == 0
    assert "KONG" in result.output or "kong" in result.output.lower()
    assert "kong analyze" in result.output.lower()
    assert "kong setup" in result.output.lower()


def test_setup_with_key_set():
    """Setup command detects existing API key."""
    runner = CliRunner()
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test1234"}):
        result = runner.invoke(cli, ["setup"])

    assert result.exit_code == 0
    assert "sk-ant-" in result.output
    assert "..." in result.output


def test_info_requires_binary_arg():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code != 0


def test_eval_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "--help"])
    assert result.exit_code == 0
    assert "score" in result.output.lower() or "analysis" in result.output.lower()


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
