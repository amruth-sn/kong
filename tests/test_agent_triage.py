"""Tests for the triage agent and signature matching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from kong.agent.signatures import SignatureDB, SignatureMatch
from kong.agent.triage import CallGraph, LanguageHints, TriageAgent, TriageResult
from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    StringEntry,
)

SIGNATURES_DIR = Path(__file__).parent.parent / "kong" / "signatures"


def _func(addr, name, size=100, cls=FunctionClassification.MEDIUM):
    return FunctionInfo(address=addr, name=name, size=size, classification=cls)


def _make_client(functions=None, binary_info=None, strings=None):
    client = MagicMock()
    client.get_binary_info.return_value = binary_info or BinaryInfo(
        arch="x86-64",
        format="ELF",
        endianness="little",
        word_size=8,
        compiler="GCC",
        name="test_binary",
    )
    client.list_functions.return_value = functions or []
    client.get_strings.return_value = strings or []
    client.get_callers.return_value = []
    client.get_callees.return_value = []
    return client


class TestSignatureDB:
    def test_load_stdlib(self):
        db = SignatureDB()
        count = db.load_file(SIGNATURES_DIR / "stdlib.json")
        assert count > 30
        assert db.size > 30

    def test_load_crypto(self):
        db = SignatureDB()
        count = db.load_file(SIGNATURES_DIR / "crypto.json")
        assert count > 5

    def test_load_directory(self):
        db = SignatureDB()
        total = db.load_directory(SIGNATURES_DIR)
        assert total > 40  # stdlib + crypto

    def test_lookup_exact(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        entry = db.lookup("malloc")
        assert entry is not None
        assert entry.name == "malloc"

    def test_lookup_alias(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        entry = db.lookup("__libc_malloc")
        assert entry is not None
        assert entry.name == "malloc"

    def test_lookup_normalized_underscores(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        # __memcpy -> stripped to memcpy
        entry = db.lookup("__memcpy")
        assert entry is not None
        assert entry.name == "memcpy"

    def test_lookup_case_insensitive(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        entry = db.lookup("MALLOC")
        assert entry is not None
        assert entry.name == "malloc"

    def test_lookup_miss(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        assert db.lookup("my_custom_function") is None

    def test_match_functions(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")
        funcs = [
            _func(0x1000, "malloc"),
            _func(0x2000, "FUN_00002000"),  # no match
            _func(0x3000, "strlen"),
        ]
        matches = db.match_functions(funcs)
        assert len(matches) == 2
        names = {m.matched_name for m in matches}
        assert names == {"malloc", "strlen"}

    def test_crypto_constants(self):
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "crypto.json")
        entry = db.lookup("sha256_init")
        assert entry is not None
        assert "0x6a09e667" in entry.indicators["constants"]

    def test_load_nonexistent_file(self):
        db = SignatureDB()
        count = db.load_file(Path("/nonexistent/file.json"))
        assert count == 0


class TestTriageAgent:
    def test_run_returns_triage_result(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        assert isinstance(result, TriageResult)
        assert result.total_functions == 2
        assert isinstance(result.call_graph, CallGraph)
        assert isinstance(result.queue, object)

    def test_call_graph_built(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        client = _make_client(functions=funcs)
        client.get_callees.side_effect = lambda addr: [0x2000] if addr == 0x1000 else []
        client.get_callers.side_effect = lambda addr: [0x1000] if addr == 0x2000 else []

        triage = TriageAgent(client)
        result = triage.run()

        assert result.call_graph.callees[0x1000] == [0x2000]
        assert result.call_graph.callers[0x2000] == [0x1000]

    def test_signature_matching(self):
        funcs = [_func(0x1000, "malloc"), _func(0x2000, "my_func")]
        client = _make_client(functions=funcs)
        db = SignatureDB()
        db.load_file(SIGNATURES_DIR / "stdlib.json")

        triage = TriageAgent(client, signature_db=db)
        result = triage.run()

        assert result.matched_count == 1
        assert result.signature_matches[0].matched_name == "malloc"

    def test_queue_built_with_correct_size(self):
        funcs = [
            _func(0x1000, "a"),
            _func(0x2000, "b", cls=FunctionClassification.THUNK),
            _func(0x3000, "c"),
        ]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        # Thunks excluded from queue
        assert result.queue_size == 2


class TestLanguageDetection:
    def test_detect_go(self):
        funcs = [
            _func(0x1000, "runtime.newproc"),
            _func(0x2000, "runtime.goexit"),
            _func(0x3000, "main.main"),
        ]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.language == "Go"

    def test_detect_rust(self):
        funcs = [
            _func(0x1000, "_ZN4core3ptr17h8a2e1234abcdef00E"),
            _func(0x2000, "some_func"),
        ]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.language == "Rust"

    def test_detect_cpp(self):
        funcs = [
            _func(0x1000, "_ZN5MyApp4initEv"),
            _func(0x2000, "_ZSt4cout"),
        ]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.language == "C++"

    def test_detect_cpp_from_vtable_strings(self):
        funcs = [_func(0x1000, "func_a")]
        strings = [StringEntry(address=0x5000, value="vtable for MyClass")]
        client = _make_client(functions=funcs, strings=strings)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.language == "C++"

    def test_detect_c_default(self):
        funcs = [_func(0x1000, "func_a"), _func(0x2000, "func_b")]
        client = _make_client(functions=funcs)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.language == "C"

    def test_detect_compiler_from_strings(self):
        funcs = [_func(0x1000, "a")]
        strings = [StringEntry(address=0x5000, value="GCC: (Ubuntu 12.2) 12.2.0")]
        client = _make_client(functions=funcs, strings=strings)
        triage = TriageAgent(client)
        result = triage.run()

        assert result.language_hints.compiler == "GCC"


class TestCallGraph:
    def test_edge_count(self):
        g = CallGraph(
            callers={0x1000: [], 0x2000: [0x1000]},
            callees={0x1000: [0x2000], 0x2000: []},
        )
        assert g.edge_count == 1


class TestTriageResultCounts:
    def test_classification_counts(self):
        result = TriageResult(
            binary_info=BinaryInfo(
                arch="x86", format="ELF", endianness="little", word_size=8
            ),
            functions=[
                _func(1, "a", cls=FunctionClassification.SMALL),
                _func(2, "b", cls=FunctionClassification.SMALL),
                _func(3, "c", cls=FunctionClassification.LARGE),
                _func(4, "d", cls=FunctionClassification.THUNK),
            ],
            strings=[],
            call_graph=CallGraph(),
            signature_matches=[],
            language_hints=LanguageHints(),
            queue=MagicMock(total=3),
        )
        counts = result.classification_counts()
        assert counts["small"] == 2
        assert counts["large"] == 1
        assert counts["thunk"] == 1
