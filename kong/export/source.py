from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from kong.agent.models import FunctionResult
from kong.agent.models import AnalysisStats
from kong.ghidra.types import BinaryInfo
from kong.llm.usage import TokenUsage

SECTION_ORDER: list[tuple[str, str]] = [
    ("crypto", "Crypto"),
    ("networking", "Networking"),
    ("io", "I/O"),
    ("memory", "Memory Management"),
    ("string", "String Operations"),
    ("math", "Math"),
    ("init", "Initialization"),
    ("cleanup", "Cleanup"),
    ("handler", "Handlers"),
    ("parser", "Parsers"),
    ("utility", "Utilities"),
    ("unknown", "General"),
]

_SECTION_RANK: dict[str, int] = {key: i for i, (key, _) in enumerate(SECTION_ORDER)}
_SECTION_LABEL: dict[str, str] = dict(SECTION_ORDER)


@dataclass
class ExportData:
    binary_info: BinaryInfo
    stats: AnalysisStats
    results: dict[int, FunctionResult]
    decompilations: dict[int, str]
    token_usage: TokenUsage
    duration_seconds: float


def _format_header(data: ExportData) -> str:
    bi = data.binary_info
    s = data.stats
    minutes, seconds = divmod(data.duration_seconds, 60)
    lines = [
        f"/* {'=' * 60}",
        f" * Binary:   {bi.name}",
        f" * Arch:     {bi.arch}",
        f" * Format:   {bi.format}",
        f" * Compiler: {bi.compiler}",
        f" *",
        f" * Functions: {s.total_functions} total, {s.analyzed} analyzed, "
        f"{s.skipped} skipped, {s.errors} errors",
        f" * Renamed:   {s.renamed} | Confirmed: {s.confirmed}",
        f" * LLM calls: {s.llm_calls}",
        f" * Duration:  {int(minutes)}m {seconds:.1f}s",
        f" * Cost:      ${data.token_usage.total_cost_usd:.4f}",
        f" * {'=' * 60} */",
    ]
    return "\n".join(lines)


def _format_function(result: FunctionResult, decompilation: str) -> str:
    lines = [
        "/**",
        f" * @name  {result.name}",
    ]
    if result.comments:
        lines.append(f" * @brief {result.comments}")
    lines.extend([
        f" * @confidence {result.confidence}%",
        f" * @classification {result.classification}",
        f" * @address 0x{result.address:08x}",
        " */",
    ])
    lines.append(decompilation)
    return "\n".join(lines)


def _includable_results(data: ExportData) -> list[FunctionResult]:
    results: list[FunctionResult] = []
    for addr, result in data.results.items():
        if result.skipped or result.error:
            continue
        if addr not in data.decompilations:
            continue
        results.append(result)
    return results


def export_source(data: ExportData, output_path: Path) -> Path:
    parts: list[str] = [_format_header(data), ""]

    results = _includable_results(data)

    sections: dict[str, list[FunctionResult]] = {}
    for result in results:
        key = result.classification if result.classification in _SECTION_RANK else "unknown"
        sections.setdefault(key, []).append(result)

    for section_key, section_label in SECTION_ORDER:
        funcs = sections.get(section_key)
        if not funcs:
            continue
        funcs.sort(key=lambda r: r.address)
        parts.append(f"/* {'=' * 20} {section_label} {'=' * 20} */")
        parts.append("")
        for func in funcs:
            parts.append(_format_function(func, data.decompilations[func.address]))
            parts.append("")

    output_path.write_text("\n".join(parts))
    return output_path
