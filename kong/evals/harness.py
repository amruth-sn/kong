"""Evaluation harness for scoring Kong analysis against ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scorecard:
    binary: str = ""
    total_functions: int = 0
    functions_analyzed: int = 0
    symbol_accuracy: float = 0.0
    type_accuracy: float = 0.0
    per_function: list[dict[str, str | float]] = field(default_factory=list)
    llm_calls: int = 0
    duration_seconds: float = 0.0
    cost_usd: float = 0.0


def score(analysis_path: Path, source_path: Path) -> Scorecard:
    """Score a Kong analysis.json against ground truth C source code."""
    import json
    import re

    analysis = json.loads(analysis_path.read_text())

    source_text = source_path.read_text()
    source_functions = re.findall(
        r"(?:[\w\s\*]+)\s+(\w+)\s*\([^)]*\)",
        source_text,
    )
    source_func_set = set(source_functions)

    binary_name = analysis.get("binary", {}).get("name", "")
    stats = analysis.get("stats", {})
    analyzed_functions = analysis.get("functions", [])

    per_function: list[dict[str, str | float]] = []
    symbol_matches = 0
    type_matches = 0

    for func in analyzed_functions:
        predicted_name = func.get("name", "")
        best_match = predicted_name if predicted_name in source_func_set else ""
        sym_score = 1.0 if best_match else 0.0
        typ_score = 1.0 if best_match else 0.0

        if sym_score > 0:
            symbol_matches += 1
            type_matches += 1

        per_function.append({
            "predicted_name": predicted_name,
            "truth_name": best_match or "???",
            "symbol_score": sym_score,
            "type_score": typ_score,
        })

    total = max(len(source_func_set), 1)

    return Scorecard(
        binary=binary_name,
        total_functions=len(source_func_set),
        functions_analyzed=len(analyzed_functions),
        symbol_accuracy=symbol_matches / total,
        type_accuracy=type_matches / total,
        per_function=per_function,
        llm_calls=stats.get("llm_calls", 0),
        duration_seconds=stats.get("duration_seconds", 0.0),
        cost_usd=stats.get("cost_usd", 0.0),
    )
