"""Scoring functions for evaluating Kong analysis output against ground truth."""

from __future__ import annotations

import re


def _normalize_name(name: str) -> list[str]:
    """Split camelCase/snake_case/PascalCase into lowercase word tokens."""
    parts = name.replace("_", " ").strip().split()
    tokens: list[str] = []
    for part in parts:
        splits = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
        tokens.extend(s.lower() for s in splits if s)
    return tokens


def symbol_accuracy(predicted: str, truth: str) -> float:
    """Score predicted symbol name against ground truth.

    Returns 1.0 for exact match, 0.8 for same words different order,
    Jaccard-based (capped at 0.6) for partial overlap, 0.0 for no overlap.
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
        return 0.8

    intersection = pred_set & truth_set
    if not intersection:
        return 0.0

    union = pred_set | truth_set
    jaccard = len(intersection) / len(union)
    return min(jaccard, 0.6)


def _strip_qualifiers(type_str: str) -> str:
    """Remove const/unsigned/signed qualifiers from a type string."""
    stripped = type_str
    for qualifier in ("const", "unsigned", "signed"):
        stripped = stripped.replace(qualifier, "")
    return " ".join(stripped.split())


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
    """
    if predicted_sig == truth_sig:
        return 1.0

    pred_ret, pred_params = _parse_signature(predicted_sig)
    truth_ret, truth_params = _parse_signature(truth_sig)

    score = 0.0

    if _strip_qualifiers(pred_ret) == _strip_qualifiers(truth_ret):
        score += 0.4

    if len(pred_params) == len(truth_params):
        score += 0.3

    if pred_params and truth_params:
        match_count = sum(
            1 for p, t in zip(pred_params, truth_params)
            if _strip_qualifiers(p) == _strip_qualifiers(t)
        )
        max_params = max(len(pred_params), len(truth_params))
        score += 0.3 * (match_count / max_params)
    elif not pred_params and not truth_params:
        score += 0.3

    return score


def overall_score(
    predicted: list[dict[str, str]],
    truth: list[dict[str, str]],
) -> dict[str, float]:
    """Compute aggregate symbol and type accuracy across matched functions."""
    from kong.evals.harness import match_functions

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
