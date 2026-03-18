from __future__ import annotations

import json
from pathlib import Path

from kong.agent.models import FunctionResult
from kong.config import LLMProvider
from kong.export.source import ExportData


def _build_binary_section(data: ExportData) -> dict[str, str | int]:
    bi = data.binary_info
    return {
        "name": bi.name,
        "path": bi.path,
        "arch": bi.arch,
        "format": bi.format,
        "endianness": bi.endianness,
        "word_size": bi.word_size,
        "compiler": bi.compiler,
    }


def _build_stats_section(data: ExportData) -> dict[str, int | float | bool]:
    s = data.stats
    cost_tracking = data.provider is not LLMProvider.CUSTOM if data.provider else True
    return {
        "total_functions": s.total_functions,
        "analyzed": s.analyzed,
        "named": s.named,
        "renamed": s.renamed,
        "confirmed": s.confirmed,
        "high_confidence": s.high_confidence,
        "medium_confidence": s.medium_confidence,
        "low_confidence": s.low_confidence,
        "skipped": s.skipped,
        "errors": s.errors,
        "llm_calls": s.llm_calls,
        "duration_seconds": data.duration_seconds,
        "cost_usd": data.token_usage.total_cost_usd,
        "cost_tracking": cost_tracking,
    }


def _build_function_entry(result: FunctionResult) -> dict[str, str | int | list[str]]:
    return {
        "address": f"0x{result.address:08x}",
        "original_name": result.original_name,
        "name": result.name,
        "signature": result.signature,
        "confidence": result.confidence,
        "classification": result.classification,
        "comments": result.comments,
        "reasoning": result.reasoning,
        "obfuscation_techniques": result.obfuscation_techniques,
    }


def export_json(data: ExportData, output_path: Path) -> Path:
    includable = [
        result for result in data.results.values()
        if not result.skipped and not result.error
    ]
    includable.sort(key=lambda r: r.address)

    document = {
        "binary": _build_binary_section(data),
        "stats": _build_stats_section(data),
        "functions": [_build_function_entry(r) for r in includable],
    }

    output_path.write_text(json.dumps(document, indent=2))
    return output_path
