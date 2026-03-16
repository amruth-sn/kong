from __future__ import annotations

import json
from pathlib import Path

import pytest

from kong.agent.models import AnalysisStats, FunctionResult
from kong.export.source import ExportData
from kong.export.structured import export_json
from kong.ghidra.types import BinaryInfo
from kong.llm.client import TokenUsage


@pytest.fixture
def binary_info() -> BinaryInfo:
    return BinaryInfo(
        arch="x86_64",
        format="ELF",
        endianness="little",
        word_size=64,
        compiler="gcc",
        name="test_binary",
        path="/tmp/test_binary",
    )


@pytest.fixture
def stats() -> AnalysisStats:
    s = AnalysisStats()
    s.total_functions = 10
    s.analyzed = 6
    s.renamed = 4
    s.confirmed = 1
    s.skipped = 3
    s.errors = 1
    s.llm_calls = 8
    s.high_confidence = 3
    s.medium_confidence = 2
    s.low_confidence = 1
    return s


@pytest.fixture
def token_usage() -> TokenUsage:
    return TokenUsage()


@pytest.fixture
def sample_results() -> dict[int, FunctionResult]:
    return {
        0x2000: FunctionResult(
            address=0x2000,
            original_name="FUN_00002000",
            name="parse_header",
            signature="int parse_header(uint8_t *data, size_t len)",
            confidence=75,
            classification="parser",
            comments="Parses the file header",
            reasoning="Structure matches typical header parsing",
        ),
        0x1000: FunctionResult(
            address=0x1000,
            original_name="FUN_00001000",
            name="rc4_init",
            signature="void rc4_init(uint8_t *s)",
            confidence=95,
            classification="crypto",
            comments="RC4 key schedule initialization",
            reasoning="S-box initialization pattern",
            obfuscation_techniques=["control_flow_flattening"],
        ),
        0x3000: FunctionResult(
            address=0x3000,
            original_name="FUN_00003000",
            skipped=True,
            skip_reason="thunk",
        ),
        0x4000: FunctionResult(
            address=0x4000,
            original_name="FUN_00004000",
            error="LLM timeout",
        ),
    }


def _make_data(
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    results: dict[int, FunctionResult],
    duration: float = 42.5,
) -> ExportData:
    return ExportData(
        binary_info=binary_info,
        stats=stats,
        results=results,
        decompilations={},
        token_usage=token_usage,
        duration_seconds=duration,
    )


def _export_and_load(data: ExportData, tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "output.json"
    export_json(data, output)
    return json.loads(output.read_text())


def test_file_creation(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
) -> None:
    data = _make_data(binary_info, stats, token_usage, sample_results)
    output = tmp_path / "output.json"
    result_path = export_json(data, output)
    assert result_path == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_valid_json(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
) -> None:
    data = _make_data(binary_info, stats, token_usage, sample_results)
    output = tmp_path / "output.json"
    export_json(data, output)
    parsed = json.loads(output.read_text())
    assert isinstance(parsed, dict)
    assert "binary" in parsed
    assert "stats" in parsed
    assert "functions" in parsed


def test_binary_section(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    data = _make_data(binary_info, stats, token_usage, {})
    parsed = _export_and_load(data, tmp_path)
    binary = parsed["binary"]
    assert binary["name"] == "test_binary"
    assert binary["path"] == "/tmp/test_binary"
    assert binary["arch"] == "x86_64"
    assert binary["format"] == "ELF"
    assert binary["endianness"] == "little"
    assert binary["word_size"] == 64
    assert binary["compiler"] == "gcc"


def test_stats_section(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    data = _make_data(binary_info, stats, token_usage, {}, duration=123.4)
    parsed = _export_and_load(data, tmp_path)
    s = parsed["stats"]
    assert s["total_functions"] == 10
    assert s["analyzed"] == 6
    assert s["named"] == 5  # renamed(4) + confirmed(1)
    assert s["renamed"] == 4
    assert s["confirmed"] == 1
    assert s["high_confidence"] == 3
    assert s["medium_confidence"] == 2
    assert s["low_confidence"] == 1
    assert s["skipped"] == 3
    assert s["errors"] == 1
    assert s["llm_calls"] == 8
    assert s["duration_seconds"] == 123.4
    assert "cost_usd" in s


def test_functions_structure(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
) -> None:
    data = _make_data(binary_info, stats, token_usage, sample_results)
    parsed = _export_and_load(data, tmp_path)
    funcs = parsed["functions"]
    assert len(funcs) == 2

    first = funcs[0]
    assert first["address"] == "0x00001000"
    assert first["original_name"] == "FUN_00001000"
    assert first["name"] == "rc4_init"
    assert first["signature"] == "void rc4_init(uint8_t *s)"
    assert first["confidence"] == 95
    assert first["classification"] == "crypto"
    assert first["comments"] == "RC4 key schedule initialization"
    assert first["reasoning"] == "S-box initialization pattern"
    assert first["obfuscation_techniques"] == ["control_flow_flattening"]

    second = funcs[1]
    assert second["address"] == "0x00002000"
    assert second["name"] == "parse_header"


def test_skipped_excluded(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
) -> None:
    data = _make_data(binary_info, stats, token_usage, sample_results)
    parsed = _export_and_load(data, tmp_path)
    addresses = [f["address"] for f in parsed["functions"]]
    assert "0x00003000" not in addresses


def test_errored_excluded(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
) -> None:
    data = _make_data(binary_info, stats, token_usage, sample_results)
    parsed = _export_and_load(data, tmp_path)
    addresses = [f["address"] for f in parsed["functions"]]
    assert "0x00004000" not in addresses


def test_functions_sorted_by_address(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    results = {
        0x5000: FunctionResult(
            address=0x5000,
            original_name="FUN_00005000",
            name="func_c",
            confidence=80,
            classification="utility",
        ),
        0x1000: FunctionResult(
            address=0x1000,
            original_name="FUN_00001000",
            name="func_a",
            confidence=90,
            classification="crypto",
        ),
        0x3000: FunctionResult(
            address=0x3000,
            original_name="FUN_00003000",
            name="func_b",
            confidence=70,
            classification="parser",
        ),
    }
    data = _make_data(binary_info, stats, token_usage, results)
    parsed = _export_and_load(data, tmp_path)
    addresses = [f["address"] for f in parsed["functions"]]
    assert addresses == ["0x00001000", "0x00003000", "0x00005000"]
