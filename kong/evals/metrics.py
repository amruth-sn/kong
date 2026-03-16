"""Scoring functions for evaluating Kong analysis output against ground truth."""

from __future__ import annotations

import re

_SYNONYM_GROUPS: list[set[str]] = [
    {"search", "find", "lookup"},
    {"node", "entry", "element"},
    {"buffer", "buf"},
    {"encode", "encrypt"},
    {"decode", "decrypt"},
    {"delete", "remove"},
    {"create", "make", "new", "alloc"},
    {"insert", "add", "push"},
    {"string", "str"},
    {"print", "display", "show"},
    {"count", "num", "size", "length"},
]

_SYNONYM_MAP: dict[str, set[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_MAP[_word] = _group

_NOISE_WORDS = {"or", "from", "the", "a", "an", "of", "to", "and", "is", "in"}

_TYPE_ALIASES: dict[str, str] = {
    "uint": "int",
    "byte": "char",
    "undefined4": "int",
    "undefined8": "long",
    "undefined2": "short",
    "undefined1": "char",
    "bool": "int",
    "dword": "int",
    "qword": "long",
    "word": "short",
}


def _normalize_name(name: str) -> list[str]:
    """Split camelCase/snake_case/PascalCase into lowercase word tokens."""
    parts = name.replace("_", " ").strip().split()
    tokens: list[str] = []
    for part in parts:
        splits = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
        tokens.extend(s.lower() for s in splits if s)
    return [t for t in tokens if t not in _NOISE_WORDS]


def _is_synonym(a: str, b: str) -> bool:
    if a == b:
        return True
    group = _SYNONYM_MAP.get(a)
    return group is not None and b in group


def _synonym_recall(pred_set: set[str], truth_set: set[str]) -> float:
    """Fraction of truth tokens matched by a pred token (exact or synonym)."""
    if not truth_set:
        return 0.0
    matched = 0
    for t in truth_set:
        if any(_is_synonym(t, p) for p in pred_set):
            matched += 1
    return matched / len(truth_set)


def symbol_accuracy(predicted: str, truth: str) -> float:
    """Score predicted symbol name against ground truth.

    Recall-weighted: measures how well the prediction captures the ground
    truth concepts.  Extra descriptive words in the prediction are not
    penalized heavily — they add context in RE output.

    Returns:
        1.0  exact string match
        0.9  same word set, different order
        0.8  all truth words present (superset) — exact or synonym
        recall * 0.7  partial overlap, scaled by truth coverage
        0.0  no overlap even after synonym expansion
    """
    if predicted == truth:
        return 1.0

    pred_tokens = _normalize_name(predicted)
    truth_tokens = _normalize_name(truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    truth_set = set(truth_tokens)

    if pred_set == truth_set:
        return 0.9

    recall = _synonym_recall(pred_set, truth_set)

    if recall >= 1.0:
        return 0.8
    if recall > 0.0:
        return recall * 0.7
    return 0.0


def _normalize_type(type_str: str) -> str:
    """Normalize a C type string: strip qualifiers and resolve Ghidra aliases."""
    stripped = type_str
    for qualifier in ("const", "unsigned", "signed"):
        stripped = stripped.replace(qualifier, "")
    stripped = " ".join(stripped.split())

    base = stripped.rstrip("*").rstrip()
    stars = stripped[len(base) :]

    resolved = _TYPE_ALIASES.get(base, base)
    return (resolved + stars).strip()


def _parse_signature(sig: str) -> tuple[str, list[str]]:
    """Extract (return_type, [param_types]) from a C signature string."""
    paren_match = re.match(r"^(.*?)\s*\w+\s*\((.*)\)\s*$", sig)
    if not paren_match:
        return ("", [])

    return_type = paren_match.group(1).strip()
    params_str = paren_match.group(2).strip()

    if not params_str or params_str == "void":
        return (return_type, [])

    param_types: list[str] = []
    for param in params_str.split(","):
        param = param.strip()
        pointer_match = re.match(r"^(.*\*)\s*\w*$", param)
        if pointer_match:
            param_types.append(pointer_match.group(1).strip())
        else:
            parts = param.rsplit(None, 1)
            if len(parts) >= 2:
                param_types.append(parts[0].strip())
            else:
                param_types.append(param.strip())

    return (return_type, param_types)


def type_accuracy(predicted_sig: str, truth_sig: str) -> float:
    """Score predicted type signature against ground truth.

    Exact match: 1.0. Otherwise scored by components:
    return type match (0.4), param count match (0.3),
    individual param type matches (up to 0.3).

    Types are normalized through Ghidra alias resolution before comparison.
    """
    if predicted_sig == truth_sig:
        return 1.0

    pred_ret, pred_params = _parse_signature(predicted_sig)
    truth_ret, truth_params = _parse_signature(truth_sig)

    score = 0.0

    if _normalize_type(pred_ret) == _normalize_type(truth_ret):
        score += 0.4

    if len(pred_params) == len(truth_params):
        score += 0.3

    if pred_params and truth_params:
        match_count = sum(
            1
            for p, t in zip(pred_params, truth_params)
            if _normalize_type(p) == _normalize_type(t)
        )
        max_params = max(len(pred_params), len(truth_params))
        score += 0.3 * (match_count / max_params)
    elif not pred_params and not truth_params:
        score += 0.3

    return score


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


def overall_score(
    predicted: list[dict[str, str]],
    truth: list[dict[str, str]],
) -> dict[str, float]:
    """Compute aggregate symbol and type accuracy across matched functions."""

    matches = match_functions(predicted, truth)

    if not matches:
        return {"symbol_accuracy": 0.0, "type_accuracy": 0.0}

    sym_scores: list[float] = []
    type_scores: list[float] = []

    for pred, tr, _ in matches:
        if tr is not None:
            sym_scores.append(symbol_accuracy(pred["name"], tr["name"]))
            type_scores.append(type_accuracy(pred["signature"], tr["signature"]))
        else:
            sym_scores.append(0.0)
            type_scores.append(0.0)

    return {
        "symbol_accuracy": sum(sym_scores) / len(sym_scores),
        "type_accuracy": sum(type_scores) / len(type_scores),
    }
