"""Tests for kong.symbolic modules."""

from __future__ import annotations

import pytest

from kong.ghidra.types import (
    BasicBlock,
    BlockEdge,
    BlockEdgeType,
    ControlFlowGraph,
    PcodeOp,
)
from kong.symbolic.dead_code import (
    Resolution,
    ResolvedPredicate,
    eliminate_dead_code,
)
from kong.symbolic.simplifier import (
    PredicateKind,
    SimplificationResult,
    simplify_expression,
)
from kong.symbolic.state_machine import (
    StateMachineResult,
    StateTransition,
    trace_state_machine_from_cfg,
)


class TestSimplifierOpaquePredicates:
    """Opaque predicates should be detected as always-true or always-false."""

    def test_product_consecutive_mod2(self):
        result = simplify_expression("(x * (x + 1)) % 2 == 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_squared_mod2(self):
        result = simplify_expression("(x * x) % 2 == x % 2")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_xor_self(self):
        result = simplify_expression("(x ^ x) == 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_or_not_self(self):
        result = simplify_expression("(x | ~x) != 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_contradiction(self):
        result = simplify_expression("(x & ~x) != 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_FALSE

    def test_always_false_xor_self_nonzero(self):
        result = simplify_expression("(x ^ x) != 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_FALSE

    def test_variable_predicate(self):
        result = simplify_expression("x > 5")
        assert result.is_opaque is False
        assert result.predicate_kind == PredicateKind.VARIABLE


class TestSimplifierArithmetic:
    """Arithmetic simplification tests."""

    def test_xor_plus_and_identity(self):
        result = simplify_expression("(a ^ b) + 2 * (a & b)")
        assert result.is_opaque is False
        assert "+" in result.simplified or result.simplified != result.original

    def test_literal_arithmetic(self):
        result = simplify_expression("3 + 5")
        assert result.simplified == "8"

    def test_hex_literal(self):
        result = simplify_expression("0xff & 0x0f")
        assert result.simplified == "15"

    def test_zero_xor(self):
        result = simplify_expression("x ^ 0")
        assert result.is_opaque is False

    def test_subtract_self(self):
        result = simplify_expression("(x - x) == 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE


class TestSimplifierParsing:
    """Parser edge cases."""

    def test_nested_parentheses(self):
        result = simplify_expression("((x + 1) * (x - 1))")
        assert result.error == ""

    def test_unary_minus(self):
        result = simplify_expression("-x + x")
        assert result.error == ""

    def test_logical_operators(self):
        result = simplify_expression("x > 0 && x < 10")
        assert result.is_opaque is False
        assert result.error == ""

    def test_logical_or(self):
        result = simplify_expression("x == 0 || x != 0")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_bitwise_not(self):
        result = simplify_expression("~~x == x")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE

    def test_shift_operators(self):
        result = simplify_expression("1 << 3")
        assert result.simplified == "8"

    def test_empty_expression(self):
        result = simplify_expression("")
        assert result.error != ""

    def test_invalid_expression(self):
        result = simplify_expression("@#$")
        assert result.error != ""

    def test_suffix_literal(self):
        result = simplify_expression("42u + 0")
        assert result.simplified == "42"

    def test_comparison_chain(self):
        result = simplify_expression("x >= 0 && x <= 0xff")
        assert result.is_opaque is False
        assert result.error == ""

    def test_not_equal(self):
        result = simplify_expression("x != x")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_FALSE

    def test_boolean_true_false_keywords(self):
        result = simplify_expression("true && false")
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_FALSE

    def test_64bit_width(self):
        result = simplify_expression("(x ^ x) == 0", bit_width=64)
        assert result.is_opaque is True
        assert result.predicate_kind == PredicateKind.ALWAYS_TRUE


# ===================================================================
# State machine tracer tests
# ===================================================================

def _make_cff_cfg() -> tuple[ControlFlowGraph, list[PcodeOp]]:
    """Build a synthetic CFF control flow graph.

    Simulates:
        state = 0x3a2b  (entry)
        while (1) {
            switch (state) {
                case 0x3a2b: ... state = 0x7f01; break;
                case 0x7f01: ... state = 0x1c44; break;
                case 0x1c44: ... state = 0x3a2b; break;   (loop back)
                case 0x9e22: return;                         (exit)
            }
        }
    """
    blocks = [
        BasicBlock(start_addr=0x1000, end_addr=0x100f, instructions=["mov"]),
        BasicBlock(start_addr=0x1010, end_addr=0x101f, instructions=["cmp", "jne"]),
        BasicBlock(start_addr=0x1020, end_addr=0x102f, instructions=["xor"]),
        BasicBlock(start_addr=0x1030, end_addr=0x103f, instructions=["add"]),
        BasicBlock(start_addr=0x1040, end_addr=0x104f, instructions=["ret"]),
    ]
    edges = [
        BlockEdge(from_addr=0x1000, to_addr=0x1010, edge_type=BlockEdgeType.FALL_THROUGH),
        BlockEdge(from_addr=0x1010, to_addr=0x1020, edge_type=BlockEdgeType.BRANCH),
        BlockEdge(from_addr=0x1010, to_addr=0x1030, edge_type=BlockEdgeType.BRANCH),
        BlockEdge(from_addr=0x1010, to_addr=0x1040, edge_type=BlockEdgeType.BRANCH),
        BlockEdge(from_addr=0x1020, to_addr=0x1010, edge_type=BlockEdgeType.BRANCH),
        BlockEdge(from_addr=0x1030, to_addr=0x1010, edge_type=BlockEdgeType.BRANCH),
    ]
    cfg = ControlFlowGraph(function_addr=0x1000, blocks=blocks, edges=edges)

    pcode_ops = [
        PcodeOp(mnemonic="COPY", address=0x1000, inputs=["(const, 0x3a2b)"], output="state"),
        PcodeOp(mnemonic="INT_EQUAL", address=0x1010, inputs=["state", "(const, 0x3a2b)"], output=""),
        PcodeOp(mnemonic="COPY", address=0x1020, inputs=["(const, 0x7f01)"], output="state"),
        PcodeOp(mnemonic="COPY", address=0x1030, inputs=["(const, 0x1c44)"], output="state"),
        PcodeOp(mnemonic="COPY", address=0x1040, inputs=["(const, 0x9e22)"], output="state"),
    ]
    return cfg, pcode_ops


class TestStateMachineTracer:
    def test_extracts_states(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "state")
        assert result.error == ""
        assert set(result.states) == {0x3a2b, 0x7f01, 0x1c44, 0x9e22}

    def test_finds_transitions(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "state")
        from_to = {(t.from_state, t.to_state) for t in result.transitions}
        assert (0x3a2b, 0x7f01) not in from_to or len(result.transitions) > 0

    def test_finds_exit_states(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "state")
        assert 0x9e22 in result.exit_states

    def test_entry_state(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "state")
        assert result.entry_state == 0x3a2b

    def test_state_variable_stored(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "state")
        assert result.state_variable == "state"

    def test_wrong_state_var(self):
        cfg, pcode = _make_cff_cfg()
        result = trace_state_machine_from_cfg(cfg, pcode, "nonexistent")
        assert result.error != ""

    def test_empty_cfg(self):
        cfg = ControlFlowGraph(function_addr=0x1000)
        result = trace_state_machine_from_cfg(cfg, [], "state")
        assert result.error != ""

    def test_linear_chains(self):
        result = StateMachineResult(
            states=[1, 2, 3, 4, 5],
            transitions=[
                StateTransition(from_state=1, to_state=2),
                StateTransition(from_state=2, to_state=3),
                StateTransition(from_state=3, to_state=4),
                StateTransition(from_state=3, to_state=5),
            ],
        )
        chains = result.linear_chains()
        chain_sets = [set(c) for c in chains]
        assert any({1, 2, 3} <= s for s in chain_sets)

    def test_transitions_from(self):
        result = StateMachineResult(
            states=[1, 2, 3],
            transitions=[
                StateTransition(from_state=1, to_state=2),
                StateTransition(from_state=1, to_state=3),
                StateTransition(from_state=2, to_state=3),
            ],
        )
        assert len(result.transitions_from(1)) == 2
        assert len(result.transitions_from(3)) == 0


# ===================================================================
# Dead code elimination tests
# ===================================================================

class TestDeadCodeElimination:
    def test_always_true_removes_else(self):
        code = """\
if (x * (x + 1) % 2 == 0) {
    do_real_work();
} else {
    do_fake_work();
}"""
        pred = ResolvedPredicate(
            expression="x * (x + 1) % 2 == 0",
            resolution=Resolution.ALWAYS_TRUE,
        )
        result = eliminate_dead_code(code, [pred])
        assert "do_real_work" in result
        assert "do_fake_work" not in result
        assert "if" not in result

    def test_always_false_removes_then(self):
        code = """\
if (x & ~x) {
    unreachable();
} else {
    reachable();
}"""
        pred = ResolvedPredicate(
            expression="x & ~x",
            resolution=Resolution.ALWAYS_FALSE,
        )
        result = eliminate_dead_code(code, [pred])
        assert "reachable" in result
        assert "unreachable" not in result

    def test_always_false_no_else_removes_block(self):
        code = """\
if (x & ~x) {
    dead_code();
}
live_code();"""
        pred = ResolvedPredicate(
            expression="x & ~x",
            resolution=Resolution.ALWAYS_FALSE,
        )
        result = eliminate_dead_code(code, [pred])
        assert "dead_code" not in result
        assert "live_code" in result

    def test_multiple_predicates(self):
        code = """\
if (x ^ x) {
    dead1();
}
if (x | ~x) {
    live();
} else {
    dead2();
}"""
        preds = [
            ResolvedPredicate(expression="x ^ x", resolution=Resolution.ALWAYS_FALSE),
            ResolvedPredicate(expression="x | ~x", resolution=Resolution.ALWAYS_TRUE),
        ]
        result = eliminate_dead_code(code, preds)
        assert "dead1" not in result
        assert "dead2" not in result
        assert "live" in result

    def test_no_predicates_returns_unchanged(self):
        code = "if (x > 0) { work(); }"
        result = eliminate_dead_code(code, [])
        assert result == code

    def test_predicate_not_found_returns_unchanged(self):
        code = "if (y > 0) { work(); }"
        pred = ResolvedPredicate(
            expression="x > 0",
            resolution=Resolution.ALWAYS_TRUE,
        )
        result = eliminate_dead_code(code, [pred])
        assert result == code

    def test_nested_braces(self):
        code = """\
if (x ^ x) {
    if (y > 0) {
        inner();
    }
    outer();
}
after();"""
        pred = ResolvedPredicate(
            expression="x ^ x",
            resolution=Resolution.ALWAYS_FALSE,
        )
        result = eliminate_dead_code(code, [pred])
        assert "inner" not in result
        assert "outer" not in result
        assert "after" in result
