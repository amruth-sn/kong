"""Tool definitions and executor for LLM-driven deobfuscation.

Defines the Anthropic tool schemas exposed to the LLM during deobfuscation
analysis, and the ToolExecutor that dispatches calls to the symbolic tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from kong.symbolic.dead_code import ResolvedPredicate, Resolution, eliminate_dead_code
from kong.symbolic.simplifier import simplify_expression

if TYPE_CHECKING:
    from kong.ghidra.client import GhidraClient

logger = logging.getLogger(__name__)


ToolSchema = dict[str, Any]


DEOBFUSCATION_TOOLS: list[ToolSchema] = [
    {
        "name": "simplify_expression",
        "description": (
            "Simplify a C-style arithmetic or boolean expression using z3. "
            "Detects opaque predicates (expressions that are always true or "
            "always false for all inputs). Use this on suspicious branch "
            "conditions to identify dead code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A C-style expression to simplify, e.g. "
                        "'(x * (x + 1)) % 2 == 0'"
                    ),
                },
                "bit_width": {
                    "type": "integer",
                    "description": "Bit width for integer variables (default: 32).",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "eliminate_dead_code",
        "description": (
            "Remove dead branches from decompiled C code given a list of "
            "resolved opaque predicates. For always-true predicates, the else "
            "branch is removed. For always-false predicates, the if body is "
            "removed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "decompiled_code": {
                    "type": "string",
                    "description": "The decompiled C source code to clean up.",
                },
                "resolved_predicates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The predicate expression text.",
                            },
                            "resolution": {
                                "type": "string",
                                "enum": ["always_true", "always_false"],
                                "description": "Whether the predicate is always true or always false.",
                            },
                        },
                        "required": ["expression", "resolution"],
                    },
                    "description": "List of predicates that have been resolved.",
                },
            },
            "required": ["decompiled_code", "resolved_predicates"],
        },
    },
    {
        "name": "trace_state_machine",
        "description": (
            "Trace CFF (control flow flattening) state transitions in a "
            "function using Ghidra's IR. Returns the state transition graph "
            "including states, transitions, entry state, and exit states. "
            "Use this when you detect a while(1)/switch(state) pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_address": {
                    "type": "integer",
                    "description": "Address of the function to trace (as integer).",
                },
                "state_variable": {
                    "type": "string",
                    "description": (
                        "Name of the state variable in the decompiled code "
                        "(e.g. 'local_10', 'state')."
                    ),
                },
            },
            "required": ["function_address", "state_variable"],
        },
    },
    {
        "name": "identify_crypto_constants",
        "description": (
            "Scan a function's decompilation for known cryptographic constants "
            "(AES S-box values, RC4 initialization, SHA-256 initial hash values, "
            "etc.). Returns any matches found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_address": {
                    "type": "integer",
                    "description": "Address of the function to scan.",
                },
            },
            "required": ["function_address"],
        },
    },
    {
        "name": "get_decompilation",
        "description": (
            "Fetch the current decompiled C source for a function from Ghidra. "
            "Useful to re-read decompilation after Ghidra state changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_address": {
                    "type": "integer",
                    "description": "Address of the function to decompile.",
                },
            },
            "required": ["function_address"],
        },
    },
    {
        "name": "get_basic_blocks",
        "description": (
            "Get the basic blocks and control flow graph of a function. "
            "Returns blocks with their instructions and edges between blocks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_address": {
                    "type": "integer",
                    "description": "Address of the function to analyze.",
                },
            },
            "required": ["function_address"],
        },
    },
]


_CRYPTO_CONSTANTS: dict[str, list[str]] = {
    "AES S-box": ["0x63", "0x7c", "0x77", "0x7b", "0xf2", "0x6b", "0x6f", "0xc5"],
    "AES inverse S-box": ["0x52", "0x09", "0x6a", "0xd5", "0x30", "0x36", "0xa5", "0x38"],
    "AES round constants": ["0x01", "0x02", "0x04", "0x08", "0x10", "0x20", "0x40", "0x80", "0x1b", "0x36"],
    "RC4 KSA": ["0x100"],
    "SHA-256 initial hash": [
        "0x6a09e667", "0xbb67ae85", "0x3c6ef372", "0xa54ff53a",
        "0x510e527f", "0x9b05688c", "0x1f83d9ab", "0x5be0cd19",
    ],
    "SHA-256 round constants": [
        "0x428a2f98", "0x71374491", "0xb5c0fbcf", "0xe9b5dba5",
    ],
    "MD5 T-values": ["0xd76aa478", "0xe8c7b756", "0x242070db", "0xc1bdceee"],
    "CRC32 polynomial": ["0xedb88320", "0x04c11db7"],
    "TEA delta": ["0x9e3779b9"],
}


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: dict[str, Any]
    result: str


class ToolExecutor:
    """Executes tool calls from the LLM, dispatching to symbolic tools."""

    def __init__(self, ghidra_client: GhidraClient) -> None:
        self.client = ghidra_client
        self.call_log: list[ToolCallRecord] = field(default_factory=list)
        self.call_log = []

    @property
    def call_count(self) -> int:
        return len(self.call_log)

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        logger.debug("Tool call: %s(%s)", tool_name, tool_input)
        try:
            result = self._dispatch(tool_name, tool_input)
        except Exception as e:
            result = f"Error executing {tool_name}: {e}"
            logger.warning("Tool execution error: %s", result)

        self.call_log.append(ToolCallRecord(
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
        ))
        return result

    def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "simplify_expression":
            return self._simplify_expression(tool_input)
        if tool_name == "eliminate_dead_code":
            return self._eliminate_dead_code(tool_input)
        if tool_name == "trace_state_machine":
            return self._trace_state_machine(tool_input)
        if tool_name == "identify_crypto_constants":
            return self._identify_crypto_constants(tool_input)
        if tool_name == "get_decompilation":
            return self._get_decompilation(tool_input)
        if tool_name == "get_basic_blocks":
            return self._get_basic_blocks(tool_input)
        return f"Unknown tool: {tool_name}"

    def _simplify_expression(self, inputs: dict[str, Any]) -> str:
        expr = inputs["expression"]
        bit_width = inputs.get("bit_width", 32)
        result = simplify_expression(expr, bit_width=bit_width)
        return json.dumps({
            "original": result.original,
            "simplified": result.simplified,
            "is_opaque_predicate": result.is_opaque,
            "predicate_kind": result.predicate_kind.value,
            "error": result.error,
        })

    def _eliminate_dead_code(self, inputs: dict[str, Any]) -> str:
        code = inputs["decompiled_code"]
        preds = [
            ResolvedPredicate(
                expression=p["expression"],
                resolution=Resolution(p["resolution"]),
            )
            for p in inputs["resolved_predicates"]
        ]
        cleaned = eliminate_dead_code(code, preds)
        return cleaned

    def _trace_state_machine(self, inputs: dict[str, Any]) -> str:
        from kong.symbolic.state_machine import trace_state_machine

        func_addr = inputs["function_address"]
        state_var = inputs["state_variable"]
        result = trace_state_machine(self.client, func_addr, state_var)
        return json.dumps({
            "states": [hex(s) for s in result.states],
            "transitions": [
                {
                    "from": hex(t.from_state),
                    "to": hex(t.to_state),
                    "condition": t.condition,
                }
                for t in result.transitions
            ],
            "entry_state": hex(result.entry_state),
            "exit_states": [hex(s) for s in result.exit_states],
            "state_count": result.state_count,
            "linear_chains": [
                [hex(s) for s in chain]
                for chain in result.linear_chains()
            ],
            "error": result.error,
        })

    def _identify_crypto_constants(self, inputs: dict[str, Any]) -> str:
        func_addr = inputs["function_address"]
        try:
            decomp = self.client.get_decompilation(func_addr)
        except Exception as e:
            return json.dumps({"error": str(e), "matches": []})

        matches: list[dict[str, str]] = []
        decomp_lower = decomp.lower()
        for algo, constants in _CRYPTO_CONSTANTS.items():
            found = [c for c in constants if c.lower() in decomp_lower]
            if len(found) >= 2 or (len(found) >= 1 and len(constants) <= 2):
                matches.append({
                    "algorithm": algo,
                    "matched_constants": found,
                    "total_constants": len(constants),
                })
        return json.dumps({"matches": matches})

    def _get_decompilation(self, inputs: dict[str, Any]) -> str:
        func_addr = inputs["function_address"]
        try:
            return self.client.get_decompilation(func_addr)
        except Exception as e:
            return f"Error: {e}"

    def _get_basic_blocks(self, inputs: dict[str, Any]) -> str:
        func_addr = inputs["function_address"]
        try:
            cfg = self.client.get_control_flow_graph(func_addr)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({
            "function_address": hex(cfg.function_addr),
            "block_count": cfg.block_count,
            "blocks": [
                {
                    "start": hex(b.start_addr),
                    "end": hex(b.end_addr),
                    "instructions": b.instructions,
                    "successors": [hex(s) for s in cfg.successors(b.start_addr)],
                    "predecessors": [hex(p) for p in cfg.predecessors(b.start_addr)],
                }
                for b in cfg.blocks
            ],
            "edges": [
                {
                    "from": hex(e.from_addr),
                    "to": hex(e.to_addr),
                    "type": e.edge_type.value,
                }
                for e in cfg.edges
            ],
        })
