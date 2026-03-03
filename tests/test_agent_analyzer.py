"""Tests for the analyzer agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kong.agent.analyzer import (
    Analyzer,
    AnalysisContext,
    CallerSnippet,
    CalleeSnippet,
    LLMResponse,
    VariableRename,
)
from kong.agent.queue import WorkItem
from kong.agent.models import FunctionResult
from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    StringEntry,
    XRef,
)


def _func(addr=0x1000, name="FUN_00001000", size=100):
    return FunctionInfo(
        address=addr, name=name, size=size,
        classification=FunctionClassification.MEDIUM,
    )


def _item(addr=0x1000, name="FUN_00001000", callers=None, callees=None):
    return WorkItem(
        function=_func(addr, name),
        callers=callers or [],
        callees=callees or [],
    )


def _binary_info():
    return BinaryInfo(
        arch="x86-64", format="ELF", endianness="little",
        word_size=8, compiler="GCC",
    )


def _mock_llm(response: LLMResponse):
    llm = MagicMock()
    llm.analyze_function.return_value = response
    return llm


def _mock_client():
    client = MagicMock()
    client.get_decompilation.return_value = "void FUN_00001000(void) { return; }"
    client.get_function_info.return_value = _func()
    client.get_xrefs_from.return_value = []
    return client


class TestAnalyzerAnalyze:
    def test_basic_analysis(self):
        response = LLMResponse(
            name="init_module",
            signature="void init_module(void)",
            confidence=85,
            classification="init",
            comments="Initializes the module",
            reasoning="Single initialization call pattern",
        )
        llm = _mock_llm(response)
        client = _mock_client()

        analyzer = Analyzer(client, llm)
        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        assert result.name == "init_module"
        assert result.signature == "void init_module(void)"
        assert result.confidence == 85
        assert result.classification == "init"
        assert result.llm_calls == 1

    def test_writes_back_to_ghidra(self):
        response = LLMResponse(
            name="decrypt_data",
            signature="void decrypt_data(char *buf, int len)",
            comments="Decrypts buffer using XOR",
        )
        llm = _mock_llm(response)
        client = _mock_client()

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, [])

        client.rename_function.assert_called_once_with(0x1000, "decrypt_data")
        client.set_function_signature.assert_called_once_with(
            0x1000, "void decrypt_data(char *buf, int len)"
        )
        client.add_comment.assert_called_once_with(0x1000, "Decrypts buffer using XOR")

    def test_skips_empty_writeback_fields(self):
        response = LLMResponse(name="foo")
        llm = _mock_llm(response)
        client = _mock_client()

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, [])

        client.rename_function.assert_called_once()
        client.set_function_signature.assert_not_called()
        client.add_comment.assert_not_called()

    def test_writeback_failure_doesnt_crash(self):
        response = LLMResponse(name="bad_name!@#")
        llm = _mock_llm(response)
        client = _mock_client()
        client.rename_function.side_effect = Exception("Invalid name")

        analyzer = Analyzer(client, llm)
        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        assert result.name == "bad_name!@#"


class TestAnalyzerContext:
    def test_includes_decompilation(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.return_value = "int main() { return 0; }"

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "int main() { return 0; }" in prompt

    def test_includes_binary_info(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "x86-64" in prompt
        assert "ELF" in prompt

    def test_includes_callee_snippets(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.side_effect = lambda addr: (
            "void callee() { stuff; }" if addr == 0x2000
            else "void target() { callee(); }"
        )

        item = _item(callees=[0x2000])
        analyzer = Analyzer(client, llm)
        analyzer.analyze(item, _binary_info(), {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "Called Functions" in prompt
        assert "callee" in prompt

    def test_includes_caller_snippets(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.side_effect = lambda addr: (
            "void caller() { target(); }" if addr == 0x3000
            else "void target() { return; }"
        )

        item = _item(callers=[0x3000])
        analyzer = Analyzer(client, llm)
        analyzer.analyze(item, _binary_info(), {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "Calling Functions" in prompt

    def test_includes_referenced_strings(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_xrefs_from.return_value = [
            XRef(from_addr=0x1000, to_addr=0x5000, ref_type="DATA"),
        ]
        strings = [StringEntry(address=0x5000, value="Hello, World!")]

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, strings)

        prompt = llm.analyze_function.call_args[0][0]
        assert "Hello, World!" in prompt

    def test_includes_known_functions(self):
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()

        known = {
            0x2000: FunctionResult(
                address=0x2000, original_name="FUN_2000",
                name="rc4_init", confidence=90,
            ),
        }

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), known, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "rc4_init" in prompt
        assert "Already Identified" in prompt

    def test_uses_known_name_for_callee(self):
        """When a callee has already been analyzed, use its resolved name."""
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.return_value = "void f() {}"

        known = {
            0x2000: FunctionResult(
                address=0x2000, original_name="FUN_2000",
                name="malloc_wrapper", confidence=80,
            ),
        }

        item = _item(callees=[0x2000])
        analyzer = Analyzer(client, llm)
        ctx = analyzer._build_context(item, _binary_info(), known, [])

        callee_names = [c.name for c in ctx.callee_snippets]
        assert "malloc_wrapper" in callee_names


class TestParseLLMJson:
    def test_parse_clean_json(self):
        raw = '{"name": "init", "signature": "void init()", "confidence": 90, "classification": "init", "comments": "init func", "reasoning": "obvious"}'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.name == "init"
        assert resp.confidence == 90

    def test_parse_fenced_json(self):
        raw = 'Here is my analysis:\n```json\n{"name": "parse_args", "confidence": 75}\n```\nDone.'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.name == "parse_args"
        assert resp.confidence == 75

    def test_parse_bare_fenced(self):
        raw = '```\n{"name": "foo"}\n```'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.name == "foo"

    def test_parse_with_variables(self):
        raw = '{"name": "f", "variables": [{"old_name": "var1", "new_name": "count"}, {"old_name": "var2", "new_name": "buffer"}]}'
        resp = Analyzer.parse_llm_json(raw)
        assert len(resp.variables) == 2
        assert resp.variables[0].old_name == "var1"
        assert resp.variables[0].new_name == "count"

    def test_parse_invalid_json(self):
        raw = "This is not JSON at all"
        resp = Analyzer.parse_llm_json(raw)
        assert resp.name == ""
        assert "Failed to parse" in resp.reasoning

    def test_parse_missing_fields(self):
        raw = '{"name": "f"}'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.name == "f"
        assert resp.confidence == 0
        assert resp.classification == ""

    def test_parse_preserves_raw(self):
        raw = '{"name": "test"}'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.raw == raw
