from __future__ import annotations

from pathlib import Path

import pytest

from kong.agent.models import AnalysisStats, FunctionResult
from kong.export.source import ExportData, export_source
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
        0x1000: FunctionResult(
            address=0x1000,
            original_name="FUN_00001000",
            name="aes_encrypt",
            confidence=92,
            classification="crypto",
            comments="AES-128 block encryption",
        ),
        0x2000: FunctionResult(
            address=0x2000,
            original_name="FUN_00002000",
            name="parse_header",
            confidence=75,
            classification="parser",
        ),
        0x3000: FunctionResult(
            address=0x3000,
            original_name="FUN_00003000",
            name="helper_func",
            confidence=60,
            classification="unknown",
        ),
        0x4000: FunctionResult(
            address=0x4000,
            original_name="FUN_00004000",
            skipped=True,
            skip_reason="thunk",
        ),
        0x5000: FunctionResult(
            address=0x5000,
            original_name="FUN_00005000",
            error="LLM timeout",
        ),
    }


@pytest.fixture
def sample_decompilations() -> dict[int, str]:
    return {
        0x1000: "void aes_encrypt(uint8_t *block, uint8_t *key) {\n  // ...\n}",
        0x2000: "int parse_header(uint8_t *data, size_t len) {\n  return 0;\n}",
        0x3000: "void helper_func(void) {\n  return;\n}",
    }


def test_file_creation(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    result_path = export_source(data, output)
    assert result_path == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_header_contains_binary_info(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "test_binary" in content
    assert "x86_64" in content
    assert "gcc" in content
    assert "ELF" in content


def test_header_contains_stats(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results={},
        decompilations={},
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "10" in content  # total_functions
    assert "6" in content  # analyzed


def test_functions_grouped_by_classification(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()

    crypto_pos = content.index("Crypto")
    parser_pos = content.index("Parser")
    general_pos = content.index("General")

    assert crypto_pos < parser_pos < general_pos


def test_function_doc_comments(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "aes_encrypt" in content
    assert "92%" in content
    assert "crypto" in content
    assert "AES-128 block encryption" in content


def test_skipped_functions_excluded(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "FUN_00004000" not in content


def test_errored_functions_excluded(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
    sample_results: dict[int, FunctionResult],
    sample_decompilations: dict[int, str],
) -> None:
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=sample_results,
        decompilations=sample_decompilations,
        token_usage=token_usage,
        duration_seconds=10.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "FUN_00005000" not in content


def test_functions_without_decompilations_excluded(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    results = {
        0x1000: FunctionResult(
            address=0x1000,
            original_name="FUN_00001000",
            name="some_func",
            confidence=80,
            classification="utility",
        ),
    }
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=results,
        decompilations={},
        token_usage=token_usage,
        duration_seconds=5.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()
    assert "some_func" not in content


def test_functions_ordered_by_address_within_section(
    tmp_path: Path,
    binary_info: BinaryInfo,
    stats: AnalysisStats,
    token_usage: TokenUsage,
) -> None:
    results = {
        0x3000: FunctionResult(
            address=0x3000,
            original_name="FUN_00003000",
            name="util_b",
            confidence=70,
            classification="utility",
        ),
        0x1000: FunctionResult(
            address=0x1000,
            original_name="FUN_00001000",
            name="util_a",
            confidence=80,
            classification="utility",
        ),
    }
    decompilations = {
        0x3000: "void util_b(void) {}",
        0x1000: "void util_a(void) {}",
    }
    data = ExportData(
        binary_info=binary_info,
        stats=stats,
        results=results,
        decompilations=decompilations,
        token_usage=token_usage,
        duration_seconds=5.0,
    )
    output = tmp_path / "output.c"
    export_source(data, output)
    content = output.read_text()

    pos_a = content.index("util_a")
    pos_b = content.index("util_b")
    assert pos_a < pos_b
