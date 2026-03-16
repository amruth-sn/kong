"""CFF state machine tracer using Ghidra's control flow graph.

Identifies control-flow-flattening (CFF) dispatch structures by analyzing
the function's basic blocks and pcode operations.  Extracts state values,
transitions, and classifies edges as unconditional or conditional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kong.ghidra.client import GhidraClient

from kong.ghidra.types import ControlFlowGraph, PcodeOp


@dataclass
class StateTransition:
    from_state: int
    to_state: int
    condition: str = ""


@dataclass
class StateMachineResult:
    states: list[int] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    entry_state: int = 0
    exit_states: list[int] = field(default_factory=list)
    dispatcher_addr: int = 0
    state_variable: str = ""
    error: str = ""

    @property
    def state_count(self) -> int:
        return len(self.states)

    def transitions_from(self, state: int) -> list[StateTransition]:
        return [t for t in self.transitions if t.from_state == state]

    def linear_chains(self) -> list[list[int]]:
        """Find maximal linear chains (sequences with single successor/predecessor)."""
        out_degree: dict[int, list[int]] = {}
        in_degree: dict[int, list[int]] = {}
        for t in self.transitions:
            out_degree.setdefault(t.from_state, []).append(t.to_state)
            in_degree.setdefault(t.to_state, []).append(t.from_state)

        visited: set[int] = set()
        chains: list[list[int]] = []
        for state in self.states:
            if state in visited:
                continue
            preds = in_degree.get(state, [])
            if len(preds) == 1 and len(out_degree.get(preds[0], [])) == 1:
                continue
            chain = [state]
            visited.add(state)
            current = state
            while True:
                succs = out_degree.get(current, [])
                if len(succs) != 1:
                    break
                nxt = succs[0]
                if nxt in visited:
                    break
                if len(in_degree.get(nxt, [])) != 1:
                    break
                chain.append(nxt)
                visited.add(nxt)
                current = nxt
            chains.append(chain)

        return chains


def trace_state_machine(
    client: GhidraClient,
    func_addr: int,
    state_var: str,
) -> StateMachineResult:
    """Trace CFF state transitions for a function using Ghidra IR.

    Args:
        client: An open GhidraClient instance.
        func_addr: Entry address of the function to analyze.
        state_var: Name of the state variable (e.g. ``"state"`` or ``"local_10"``).

    Returns:
        StateMachineResult with extracted states and transitions.
    """
    try:
        cfg = client.get_control_flow_graph(func_addr)
        pcode_ops = client.get_pcode_ops(func_addr)
    except Exception as e:
        return StateMachineResult(
            state_variable=state_var,
            error=f"Failed to get CFG/pcode: {e}",
        )

    return _analyze_cfg(cfg, pcode_ops, state_var)


def trace_state_machine_from_cfg(
    cfg: ControlFlowGraph,
    pcode_ops: list[PcodeOp],
    state_var: str,
) -> StateMachineResult:
    """Trace CFF state transitions from pre-built CFG data.

    Useful for testing without a live Ghidra instance.
    """
    return _analyze_cfg(cfg, pcode_ops, state_var)


def _analyze_cfg(
    cfg: ControlFlowGraph,
    pcode_ops: list[PcodeOp],
    state_var: str,
) -> StateMachineResult:
    """Core analysis: find the dispatcher, extract states and transitions."""
    if not cfg.blocks:
        return StateMachineResult(state_variable=state_var, error="Empty CFG")

    dispatcher_addr = _find_dispatcher(cfg, pcode_ops, state_var)

    state_assignments = _extract_state_assignments(pcode_ops, state_var)
    if not state_assignments:
        return StateMachineResult(
            state_variable=state_var,
            dispatcher_addr=dispatcher_addr,
            error="No state assignments found for variable",
        )

    all_states = sorted(set(state_assignments.values()))
    block_to_state = _map_blocks_to_states(cfg, state_assignments)
    transitions = _build_transitions(cfg, block_to_state, state_assignments)

    entry_state = _find_entry_state(cfg, block_to_state)
    exit_states = _find_exit_states(all_states, transitions)

    return StateMachineResult(
        states=all_states,
        transitions=transitions,
        entry_state=entry_state,
        exit_states=exit_states,
        dispatcher_addr=dispatcher_addr,
        state_variable=state_var,
    )


def _find_dispatcher(
    cfg: ControlFlowGraph,
    pcode_ops: list[PcodeOp],
    state_var: str,
) -> int:
    """Find the dispatcher block — the one with the most outgoing edges
    that also references the state variable in a comparison/branch."""
    max_out = 0
    dispatcher = cfg.function_addr
    for block in cfg.blocks:
        out_count = len(cfg.successors(block.start_addr))
        if out_count > max_out:
            has_state_ref = any(
                state_var in str(op.inputs)
                for op in pcode_ops
                if op.address >= block.start_addr
                and op.address <= block.end_addr
                and op.mnemonic
                in ("INT_EQUAL", "INT_NOTEQUAL", "CBRANCH", "MULTIEQUAL")
            )
            if has_state_ref or out_count > max_out:
                max_out = out_count
                dispatcher = block.start_addr
    return dispatcher


def _extract_state_assignments(
    pcode_ops: list[PcodeOp],
    state_var: str,
) -> dict[int, int]:
    """Extract address → state_value mappings from COPY/INT_* ops that write to the state var."""
    assignments: dict[int, int] = {}
    for op in pcode_ops:
        if state_var not in op.output:
            continue
        if op.mnemonic not in ("COPY", "INT_ADD", "INT_AND", "INT_OR", "INT_XOR"):
            continue
        for inp in op.inputs:
            val = _try_parse_constant(inp)
            if val is not None:
                assignments[op.address] = val
                break
    return assignments


def _try_parse_constant(token: str) -> int | None:
    """Try to extract an integer constant from a pcode operand string."""
    token = token.strip()
    match = re.search(r"(?:0x([0-9a-fA-F]+))|(?:^#?(-?\d+)$)", token)
    if match:
        if match.group(1):
            return int(match.group(1), 16)
        if match.group(2):
            return int(match.group(2))

    match = re.search(r"\(const,\s*(0x[0-9a-fA-F]+|\d+)", token)
    if match:
        val = match.group(1)
        return int(val, 16) if val.startswith("0x") else int(val)

    return None


def _map_blocks_to_states(
    cfg: ControlFlowGraph,
    state_assignments: dict[int, int],
) -> dict[int, int]:
    """Map block start addresses to the state value they represent
    (i.e., what state value leads execution to this block)."""
    block_to_state: dict[int, int] = {}
    for block in cfg.blocks:
        for addr, state_val in state_assignments.items():
            if block.start_addr <= addr <= block.end_addr:
                block_to_state[block.start_addr] = state_val
                break
    return block_to_state


def _build_transitions(
    cfg: ControlFlowGraph,
    block_to_state: dict[int, int],
    state_assignments: dict[int, int],
) -> list[StateTransition]:
    """Build state transitions from the CFG edges and state assignments."""
    transitions: list[StateTransition] = []
    seen: set[tuple[int, int]] = set()

    for block in cfg.blocks:
        block_state = block_to_state.get(block.start_addr)
        if block_state is None:
            continue

        for assign_addr, next_state in state_assignments.items():
            if block.start_addr <= assign_addr <= block.end_addr:
                if next_state == block_state:
                    continue
                key = (block_state, next_state)
                if key not in seen:
                    seen.add(key)
                    transitions.append(
                        StateTransition(
                            from_state=block_state,
                            to_state=next_state,
                        )
                    )

    return transitions


def _find_entry_state(
    cfg: ControlFlowGraph,
    block_to_state: dict[int, int],
) -> int:
    """The entry state is typically assigned in the block closest to the function entry."""
    if not block_to_state:
        return 0
    entry_addr = cfg.function_addr
    closest_addr = min(block_to_state.keys(), key=lambda a: abs(a - entry_addr))
    return block_to_state[closest_addr]


def _find_exit_states(
    all_states: list[int],
    transitions: list[StateTransition],
) -> list[int]:
    """Exit states are those that don't transition to any other state."""
    sources = {t.from_state for t in transitions}
    return [s for s in all_states if s not in sources]
