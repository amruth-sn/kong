"""Triage agent.

Enumerates functions, matches signatures, detects compiler/language hints,
builds the call graph, and produces the ordered work queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kong.agent.queue import WorkQueue
from kong.agent.signatures import SignatureDB, SignatureMatch
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry

logger = logging.getLogger(__name__)


@dataclass
class CallGraph:
    """Adjacency lists for the call graph."""

    callers: dict[int, list[int]] = field(
        default_factory=dict
    )  # addr -> [caller addrs]
    callees: dict[int, list[int]] = field(
        default_factory=dict
    )  # addr -> [callee addrs]

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.callees.values())


@dataclass
class LanguageHints:
    """Detected language and compiler hints from binary analysis."""

    compiler: str = "unknown"
    language: str = "C"  # C, C++, Go, Rust
    indicators: list[str] = field(default_factory=list)


@dataclass
class TriageResult:
    """Complete output of the triage phase."""

    binary_info: BinaryInfo
    functions: list[FunctionInfo]
    strings: list[StringEntry]
    call_graph: CallGraph
    signature_matches: list[SignatureMatch]
    language_hints: LanguageHints
    queue: WorkQueue

    @property
    def total_functions(self) -> int:
        return len(self.functions)

    @property
    def matched_count(self) -> int:
        return len(self.signature_matches)

    @property
    def queue_size(self) -> int:
        return self.queue.total

    def classification_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.functions:
            cls = f.classification.value if f.classification else "unknown"
            counts[cls] = counts.get(cls, 0) + 1
        return counts


class TriageAgent:
    """Runs Phase 1: enumerate, classify, match, build queue.

    Usage:
        triage = TriageAgent(client)
        result = triage.run()
        # result.queue is ready for the analysis phase
    """

    def __init__(
        self,
        client: GhidraClient,
        signature_db: SignatureDB | None = None,
    ) -> None:
        self.client = client
        self.sig_db = signature_db or SignatureDB()

    def run(self) -> TriageResult:
        """Execute the full triage phase."""
        binary_info = self.client.get_binary_info()
        functions = self.client.list_functions()
        strings = self.client.get_strings()

        call_graph = self._build_call_graph(functions)
        signature_matches = self._match_signatures(functions)
        language_hints = self._detect_language(binary_info, functions, strings)
        queue = self._build_queue(functions, call_graph)

        return TriageResult(
            binary_info=binary_info,
            functions=functions,
            strings=strings,
            call_graph=call_graph,
            signature_matches=signature_matches,
            language_hints=language_hints,
            queue=queue,
        )

    def _build_call_graph(self, functions: list[FunctionInfo]) -> CallGraph:
        """Build caller/callee adjacency lists from Ghidra xrefs."""
        graph = CallGraph()
        for func in functions:
            addr = func.address
            graph.callers[addr] = self.client.get_callers(addr)
            graph.callees[addr] = self.client.get_callees(addr)
        return graph

    def _match_signatures(self, functions: list[FunctionInfo]) -> list[SignatureMatch]:
        """Match functions against the signature database."""
        if self.sig_db.size == 0:
            self.sig_db.load_directory()

        return self.sig_db.match_functions(functions)

    def _detect_language(
        self,
        binary_info: BinaryInfo,
        functions: list[FunctionInfo],
        strings: list[StringEntry],
    ) -> LanguageHints:
        """Detect compiler and source language from binary artifacts."""
        hints = LanguageHints(compiler=binary_info.compiler)
        names = {f.name for f in functions}
        string_values = {s.value for s in strings}

        # Go detection
        go_indicators = [
            n for n in names if n.startswith("runtime.") or n.startswith("main.main")
        ]
        if go_indicators:
            hints.language = "Go"
            hints.indicators.append(f"Go runtime functions: {len(go_indicators)}")

        # Rust detection
        rust_mangled = [n for n in names if "_ZN" in n and "17h" in n]
        if rust_mangled:
            hints.language = "Rust"
            hints.indicators.append(f"Rust mangled symbols: {len(rust_mangled)}")

        # C++ detection (if not Go/Rust)
        if hints.language == "C":
            cpp_indicators = [
                n
                for n in names
                if n.startswith("_ZN") or n.startswith("_ZSt") or n.startswith("std::")
            ]
            vtable_strings = [s for s in string_values if "vtable" in s.lower()]
            if cpp_indicators or vtable_strings:
                hints.language = "C++"
                if cpp_indicators:
                    hints.indicators.append(
                        f"C++ mangled symbols: {len(cpp_indicators)}"
                    )
                if vtable_strings:
                    hints.indicators.append(f"vtable references: {len(vtable_strings)}")

        # Compiler hints from strings
        for s in string_values:
            if "GCC:" in s or "gcc" in s.lower():
                hints.compiler = "GCC"
                hints.indicators.append(f"GCC string: {s[:60]}")
                break
            if "clang" in s.lower():
                hints.compiler = "Clang"
                hints.indicators.append(f"Clang string: {s[:60]}")
                break

        return hints

    def _build_queue(
        self,
        functions: list[FunctionInfo],
        call_graph: CallGraph,
    ) -> WorkQueue:
        """Build the priority work queue from functions and call graph."""
        queue = WorkQueue()
        queue.build(
            functions,
            callers_map=call_graph.callers,
            callees_map=call_graph.callees,
        )
        return queue
