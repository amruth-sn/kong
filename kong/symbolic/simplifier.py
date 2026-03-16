"""z3-based expression simplifier and opaque predicate detector.

Parses C-style arithmetic/boolean expressions (as emitted by Ghidra's
decompiler) into z3 ASTs, then checks whether expressions are tautologies
or contradictions (opaque predicates) and simplifies where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import z3


class PredicateKind(Enum):
    ALWAYS_TRUE = "always_true"
    ALWAYS_FALSE = "always_false"
    VARIABLE = "variable"


@dataclass
class SimplificationResult:
    original: str
    simplified: str
    is_opaque: bool
    predicate_kind: PredicateKind
    error: str = ""


def simplify_expression(expr: str, bit_width: int = 32) -> SimplificationResult:
    """Simplify a C-style expression and detect opaque predicates.

    Args:
        expr: C-style expression string (e.g. ``(x * (x + 1)) % 2 == 0``).
        bit_width: Bit width for integer variables (default 32).

    Returns:
        SimplificationResult with the simplified expression and opaque
        predicate classification.
    """
    expr = expr.strip()
    try:
        parser = _ExprParser(expr, bit_width)
        z3_expr = parser.parse()
    except _ParseError as e:
        return SimplificationResult(
            original=expr,
            simplified=expr,
            is_opaque=False,
            predicate_kind=PredicateKind.VARIABLE,
            error=str(e),
        )

    if z3.is_bool(z3_expr):
        return _classify_bool(expr, z3_expr)

    simplified = z3.simplify(z3_expr)
    return SimplificationResult(
        original=expr,
        simplified=_z3_to_c(simplified),
        is_opaque=False,
        predicate_kind=PredicateKind.VARIABLE,
    )


def _classify_bool(original: str, z3_expr: z3.ExprRef) -> SimplificationResult:
    """Check whether a boolean z3 expression is always true/false."""
    solver = z3.Solver()
    solver.set("timeout", 5000)

    solver.push()
    solver.add(z3.Not(z3_expr))
    not_sat = solver.check()
    solver.pop()

    if not_sat == z3.unsat:
        return SimplificationResult(
            original=original,
            simplified="true",
            is_opaque=True,
            predicate_kind=PredicateKind.ALWAYS_TRUE,
        )

    solver.push()
    solver.add(z3_expr)
    sat = solver.check()
    solver.pop()

    if sat == z3.unsat:
        return SimplificationResult(
            original=original,
            simplified="false",
            is_opaque=True,
            predicate_kind=PredicateKind.ALWAYS_FALSE,
        )

    simplified = z3.simplify(z3_expr)
    return SimplificationResult(
        original=original,
        simplified=_z3_to_c(simplified),
        is_opaque=False,
        predicate_kind=PredicateKind.VARIABLE,
    )


class _ParseError(Exception):
    pass


_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (0[xX][0-9a-fA-F]+)       # hex literal
      | (0[bB][01]+)               # binary literal
      | (\d+[uUlL]*)              # decimal literal (with optional suffix)
      | (&&)                       # logical and
      | (\|\|)                     # logical or
      | (<<|>>)                    # shifts
      | ([!=<>]=)                  # two-char comparisons
      | ([a-zA-Z_]\w*)            # identifier
      | (.)                        # single-char operator or paren
    )
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(expr):
        tok = m.group(0).strip()
        if tok:
            tokens.append(tok)
    return tokens


_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}

_BINARY_OPS = set(_PRECEDENCE.keys())


class _ExprParser:
    """Recursive-descent parser for C-style expressions into z3."""

    def __init__(self, expr: str, bit_width: int) -> None:
        self.tokens = _tokenize(expr)
        self.pos = 0
        self.bw = bit_width
        self._vars: dict[str, z3.BitVecRef] = {}

    def parse(self) -> z3.ExprRef:
        result = self._expr(0)
        if self.pos < len(self.tokens):
            raise _ParseError(
                f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}"
            )
        return result

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tok: str) -> None:
        if self._peek() != tok:
            raise _ParseError(
                f"Expected '{tok}', got '{self._peek()}' at position {self.pos}"
            )
        self._advance()

    def _expr(self, min_prec: int) -> z3.ExprRef:
        left = self._unary()
        while True:
            op = self._peek()
            if op is None or op not in _BINARY_OPS:
                break
            prec = _PRECEDENCE[op]
            if prec < min_prec:
                break
            self._advance()
            right = self._expr(prec + 1)
            left = self._apply_binary(op, left, right)
        return left

    def _unary(self) -> z3.ExprRef:
        tok = self._peek()
        if tok == "!":
            self._advance()
            operand = self._unary()
            return self._to_bool_not(operand)
        if tok == "~":
            self._advance()
            operand = self._unary()
            operand = self._to_bv(operand)
            return ~operand
        if tok == "-" and self._is_unary_minus():
            self._advance()
            operand = self._unary()
            operand = self._to_bv(operand)
            return -operand
        return self._primary()

    def _is_unary_minus(self) -> bool:
        if self.pos == 0:
            return True
        prev = self.tokens[self.pos - 1] if self.pos > 0 else None
        return prev in _BINARY_OPS or prev == "(" or prev is None

    def _primary(self) -> z3.ExprRef:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of expression")

        if tok == "(":
            self._advance()
            inner = self._expr(0)
            self._expect(")")
            return inner

        self._advance()

        if re.match(r"^0[xX][0-9a-fA-F]+$", tok):
            return z3.BitVecVal(int(tok, 16), self.bw)
        if re.match(r"^0[bB][01]+$", tok):
            return z3.BitVecVal(int(tok, 2), self.bw)
        if re.match(r"^\d+[uUlL]*$", tok):
            num_str = tok.rstrip("uUlL")
            return z3.BitVecVal(int(num_str), self.bw)

        if re.match(r"^[a-zA-Z_]\w*$", tok):
            if tok in ("true", "True"):
                return z3.BoolVal(True)
            if tok in ("false", "False"):
                return z3.BoolVal(False)
            return self._get_var(tok)

        raise _ParseError(f"Unexpected token '{tok}'")

    def _get_var(self, name: str) -> z3.BitVecRef:
        if name not in self._vars:
            self._vars[name] = z3.BitVec(name, self.bw)
        return self._vars[name]

    def _to_bv(self, e: z3.ExprRef) -> z3.ExprRef:
        """Convert a boolean expression to a bitvector (1/0) if needed."""
        if z3.is_bool(e):
            return z3.If(e, z3.BitVecVal(1, self.bw), z3.BitVecVal(0, self.bw))
        return e

    def _to_bool(self, e: z3.ExprRef) -> z3.ExprRef:
        """Convert a bitvector to boolean (nonzero = true) if needed."""
        if z3.is_bv(e):
            return e != z3.BitVecVal(0, self.bw)
        return e

    def _to_bool_not(self, e: z3.ExprRef) -> z3.ExprRef:
        if z3.is_bool(e):
            return z3.Not(e)
        return z3.If(
            e == z3.BitVecVal(0, self.bw),
            z3.BitVecVal(1, self.bw),
            z3.BitVecVal(0, self.bw),
        )

    def _apply_binary(self, op: str, left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
        if op in ("&&", "||"):
            lb = self._to_bool(left)
            rb = self._to_bool(right)
            if op == "&&":
                return z3.And(lb, rb)
            return z3.Or(lb, rb)

        if op in ("==", "!=", "<", ">", "<=", ">="):
            left = self._to_bv(left)
            right = self._to_bv(right)
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "<":
                return left < right
            if op == ">":
                return left > right
            if op == "<=":
                return left <= right
            return left >= right

        left = self._to_bv(left)
        right = self._to_bv(right)

        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return z3.UDiv(left, right)
        if op == "%":
            return z3.URem(left, right)
        if op == "&":
            return left & right
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "<<":
            return left << right
        if op == ">>":
            return z3.LShR(left, right)

        raise _ParseError(f"Unknown binary operator: {op}")


def _z3_to_c(expr: z3.ExprRef) -> str:
    """Best-effort conversion of a z3 expression back to C-like syntax."""
    if z3.is_true(expr):
        return "true"
    if z3.is_false(expr):
        return "false"

    if z3.is_bv_value(expr):
        val = expr.as_long()
        if val > 255:
            return hex(val)
        return str(val)

    if z3.is_int_value(expr):
        return str(expr.as_long())

    return str(z3.simplify(expr))
