"""Dead code elimination on decompiled C text.

Given a set of resolved predicates (conditions proven always-true or
always-false by the z3 simplifier), rewrites the decompiled C to remove
unreachable branches and simplify constant conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Resolution(Enum):
    ALWAYS_TRUE = "always_true"
    ALWAYS_FALSE = "always_false"


@dataclass
class ResolvedPredicate:
    expression: str
    resolution: Resolution


def eliminate_dead_code(
    decompiled: str,
    resolved_predicates: list[ResolvedPredicate],
) -> str:
    """Remove dead branches from decompiled C based on resolved predicates.

    For each resolved predicate:
    - ALWAYS_TRUE: ``if (pred) { A } else { B }`` → ``A``
    - ALWAYS_FALSE: ``if (pred) { A } else { B }`` → ``B`` (or remove if no else)

    The matching is done on the condition text within ``if (...)`` statements.
    """
    result = decompiled
    for pred in resolved_predicates:
        result = _apply_predicate(result, pred)
    return result


def _apply_predicate(source: str, pred: ResolvedPredicate) -> str:
    """Apply a single resolved predicate to the source text."""
    escaped = re.escape(pred.expression)
    pattern = re.compile(
        r"if\s*\(\s*" + escaped + r"\s*\)",
        re.DOTALL,
    )

    result = source
    for match in reversed(list(pattern.finditer(source))):
        start = match.start()
        body_start = match.end()
        replacement = _rewrite_if(source, start, body_start, pred.resolution)
        if replacement is not None:
            end_pos = _find_if_else_end(source, body_start)
            result = result[:start] + replacement + result[end_pos:]
    return result


def _rewrite_if(
    source: str,
    if_start: int,
    body_start: int,
    resolution: Resolution,
) -> str | None:
    """Rewrite an if/else based on the resolution of its condition."""
    then_body, else_body, _ = _parse_if_else(source, body_start)
    if then_body is None:
        return None

    if resolution == Resolution.ALWAYS_TRUE:
        return _unwrap_body(then_body)

    if else_body is not None:
        return _unwrap_body(else_body)
    return ""


def _parse_if_else(
    source: str,
    body_start: int,
) -> tuple[str | None, str | None, int]:
    """Parse the then-body and optional else-body of an if statement.

    Returns (then_body, else_body, end_position).
    """
    then_body, then_end = _extract_braced_block(source, body_start)
    if then_body is None:
        return None, None, body_start

    rest = source[then_end:].lstrip()
    if rest.startswith("else"):
        else_start = source.index("else", then_end) + 4
        rest_after_else = source[else_start:].lstrip()
        if rest_after_else.startswith("{"):
            brace_pos = source.index("{", else_start)
            else_body, else_end = _extract_braced_block(source, brace_pos)
            return then_body, else_body, else_end
        if rest_after_else.startswith("if"):
            stmt_end = _find_if_else_end(source, source.index("if", else_start))
            else_body = source[else_start:stmt_end].strip()
            return then_body, else_body, stmt_end
        stmt_end = source.index(";", else_start) + 1
        else_body = source[else_start:stmt_end].strip()
        return then_body, else_body, stmt_end

    return then_body, None, then_end


def _extract_braced_block(source: str, start: int) -> tuple[str | None, int]:
    """Extract content between matching braces starting at or after ``start``."""
    idx = start
    while idx < len(source) and source[idx] != "{":
        if not source[idx].isspace():
            return None, start
        idx += 1

    if idx >= len(source):
        return None, start

    depth = 0
    block_start = idx + 1
    for i in range(idx, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[block_start:i], i + 1
    return None, start


def _find_if_else_end(source: str, body_start: int) -> int:
    """Find where an if/else statement ends (including any else clause)."""
    _, _, end = _parse_if_else(source, body_start)
    if end == body_start:
        result = _extract_braced_block(source, body_start)
        return result[1]
    return end


def _unwrap_body(body: str) -> str:
    """Clean up extracted body text, preserving indentation of inner lines."""
    lines = body.strip().splitlines()
    if not lines:
        return ""
    result_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            result_lines.append(stripped)
    return "\n".join(result_lines)
