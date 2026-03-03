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


def test_analyze_no_ghidra_installed(tmp_path):
    """When Ghidra is not installed, show install instructions."""
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    with patch("kong.config.find_ghidra_install", return_value=None):
        result = runner.invoke(cli, ["analyze", str(binary)])

    assert result.exit_code != 0
    assert "not installed" in result.output.lower() or "not found" in result.output.lower()
    assert "brew install ghidra" in result.output


def test_info_requires_binary_arg():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code != 0
