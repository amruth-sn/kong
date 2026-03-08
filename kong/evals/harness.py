"""Eval harness: compare Kong analysis output against ground truth source code."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kong.evals.metrics import symbol_accuracy, type_accuracy


@dataclass
class GroundTruth:
    functions: list[dict[str, str]]


@dataclass
class Scorecard:
    binary: str
    total_functions: int
    functions_analyzed: int
    symbol_accuracy: float
    type_accuracy: float
    per_function: list[dict[str, str | float]]
    llm_calls: int
    duration_seconds: float
    cost_usd: float


_FUNC_PATTERN = re.compile(
    r"^(?:static\s+)?"
    r"([\w\s\*]+?)"
    r"\s+(\w+)\s*"
    r"\(([^)]*)\)\s*\{",
    re.MULTILINE,
)


def extract_ground_truth(source_path: Path) -> GroundTruth:
    """Parse a C source file to extract function names and signatures."""
    text = source_path.read_text()
    functions: list[dict[str, str]] = []

    for match in _FUNC_PATTERN.finditer(text):
        return_type = match.group(1).strip()
        name = match.group(2).strip()
        params = match.group(3).strip()
        signature = f"{return_type} {name}({params})"
        functions.append({"name": name, "signature": signature})

    return GroundTruth(functions=functions)


def load_analysis(analysis_path: Path) -> list[dict[str, str]]:
    """Load Kong's analysis.json and return the list of function entries."""
    data = json.loads(analysis_path.read_text())
    return data["functions"]


def match_functions(
    predicted: list[dict[str, str]],
    truth: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str] | None, float]]:
    """Match predicted functions to truth entries by best symbol_accuracy (greedy, no reuse)."""
    used_truth_indices: set[int] = set()
    results: list[tuple[dict[str, str], dict[str, str] | None, float]] = []

    scored_pairs: list[tuple[int, int, float]] = []
    for pi, pred in enumerate(predicted):
        for ti, tr in enumerate(truth):
            sc = symbol_accuracy(pred["name"], tr["name"])
            scored_pairs.append((pi, ti, sc))

    scored_pairs.sort(key=lambda x: x[2], reverse=True)

    matched_pred: dict[int, tuple[int, float]] = {}
    for pi, ti, sc in scored_pairs:
        if pi in matched_pred or ti in used_truth_indices:
            continue
        if sc > 0.0:
            matched_pred[pi] = (ti, sc)
            used_truth_indices.add(ti)

    for pi, pred in enumerate(predicted):
        if pi in matched_pred:
            ti, sc = matched_pred[pi]
            results.append((pred, truth[ti], sc))
        else:
            results.append((pred, None, 0.0))

    return results


def score(analysis_path: Path, source_path: Path) -> Scorecard:
    """Load analysis and source, match functions, compute scores."""
    raw = json.loads(analysis_path.read_text())
    predicted = load_analysis(analysis_path)
    ground_truth = extract_ground_truth(source_path)

    matches = match_functions(predicted, ground_truth.functions)

    per_function: list[dict[str, str | float]] = []
    sym_scores: list[float] = []
    type_scores: list[float] = []

    for pred, tr, _ in matches:
        if tr is not None:
            sym_sc = symbol_accuracy(pred["name"], tr["name"])
            type_sc = type_accuracy(pred["signature"], tr["signature"])
        else:
            sym_sc = 0.0
            type_sc = 0.0

        sym_scores.append(sym_sc)
        type_scores.append(type_sc)
        per_function.append({
            "predicted_name": pred["name"],
            "truth_name": tr["name"] if tr else "",
            "symbol_accuracy": sym_sc,
            "type_accuracy": type_sc,
        })

    avg_sym = sum(sym_scores) / len(sym_scores) if sym_scores else 0.0
    avg_type = sum(type_scores) / len(type_scores) if type_scores else 0.0

    stats = raw.get("stats", {})

    return Scorecard(
        binary=raw.get("binary", {}).get("name", ""),
        total_functions=stats.get("total_functions", 0),
        functions_analyzed=stats.get("analyzed", 0),
        symbol_accuracy=avg_sym,
        type_accuracy=avg_type,
        per_function=per_function,
        llm_calls=stats.get("llm_calls", 0),
        duration_seconds=stats.get("duration_seconds", 0.0),
        cost_usd=stats.get("cost_usd", 0.0),
    )
