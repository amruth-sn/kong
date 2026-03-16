"""Tests for the analyzer agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kong.agent.analyzer import (
    AnalysisContext,
    Analyzer,
    LLMResponse,
)
from kong.agent.models import FunctionResult
from kong.agent.queue import WorkItem
from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    StringEntry,
    XRef,
)


def _func(addr=0x1000, name="FUN_00001000", size=100):
    return FunctionInfo(
        address=addr,
        name=name,
        size=size,
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
        arch="x86-64",
        format="ELF",
        endianness="little",
        word_size=8,
        compiler="GCC",
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

    def test_signature_applied_tracks_success(self):
        response = LLMResponse(name="f", signature="void f(int x)")
        llm = _mock_llm(response)
        client = _mock_client()

        analyzer = Analyzer(client, llm)
        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        assert result.signature_applied is True

    def test_signature_applied_tracks_failure(self):
        response = LLMResponse(name="f", signature="void f(cJSON *item)")
        llm = _mock_llm(response)
        client = _mock_client()
        client.set_function_signature.side_effect = Exception("Can't resolve datatype")

        analyzer = Analyzer(client, llm)
        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        assert result.signature_applied is False
        assert result.signature == "void f(cJSON *item)"


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
            "void callee() { stuff; }"
            if addr == 0x2000
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
            "void caller() { target(); }"
            if addr == 0x3000
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
                address=0x2000,
                original_name="FUN_2000",
                name="rc4_init",
                confidence=90,
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
                address=0x2000,
                original_name="FUN_2000",
                name="malloc_wrapper",
                confidence=80,
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


class TestNormalizerIntegration:
    def test_decompilation_normalized_before_prompt(self):
        """Decompilation artifacts (undefined4, + -) are cleaned before reaching the LLM."""
        raw_decomp = (
            "undefined4 counter;\n"
            "for (counter = 0; counter < n; counter = counter + 1) {\n"
            "  x = x + -0x10;\n"
            "}\n"
        )
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.return_value = raw_decomp

        analyzer = Analyzer(client, llm)
        analyzer.analyze(_item(), _binary_info(), {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "undefined4" not in prompt
        assert "int counter" in prompt
        assert "+ -" not in prompt
        assert "- 0x10" in prompt

    def test_caller_snippets_normalized(self):
        raw_caller = "undefined4 i;\nfor (i = 0; i < n; i = i + 1) { x = x + -5; }\n"
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.side_effect = lambda addr: (
            raw_caller if addr == 0x3000 else "void target() { return; }"
        )

        item = _item(callers=[0x3000])
        analyzer = Analyzer(client, llm)
        ctx = analyzer._build_context(item, _binary_info(), {}, [])

        assert len(ctx.caller_snippets) == 1
        assert "undefined4" not in ctx.caller_snippets[0].snippet
        assert "int i" in ctx.caller_snippets[0].snippet

    def test_callee_snippets_normalized(self):
        raw_callee = "undefined4 j;\nfor (j = 0; j < m; j = j + 1) { y = y + -3; }\n"
        llm = _mock_llm(LLMResponse(name="f"))
        client = _mock_client()
        client.get_decompilation.side_effect = lambda addr: (
            raw_callee if addr == 0x2000 else "void target() { callee(); }"
        )

        item = _item(callees=[0x2000])
        analyzer = Analyzer(client, llm)
        ctx = analyzer._build_context(item, _binary_info(), {}, [])

        assert len(ctx.callee_snippets) == 1
        assert "undefined4" not in ctx.callee_snippets[0].snippet
        assert "- 3" in ctx.callee_snippets[0].snippet


class TestBuildBatchPrompt:
    def test_batch_prompt_contains_all_functions(self) -> None:
        """Batch prompt should contain separator and decompilation for each context."""
        client = MagicMock()
        llm = MagicMock()
        analyzer = Analyzer(client, llm)

        binary_info = BinaryInfo(
            arch="x86",
            format="ELF",
            endianness="little",
            word_size=8,
            compiler="gcc",
            name="test",
        )

        contexts = []
        for i in range(3):
            ctx = AnalysisContext(
                function=FunctionInfo(
                    address=0x1000 + i * 0x100,
                    name=f"FUN_{0x1000 + i * 0x100:08x}",
                    size=64,
                ),
                decompilation=f"void func_{i}(void) {{ return; }}",
                binary_info=binary_info,
            )
            contexts.append(ctx)

        prompt = analyzer.build_batch_prompt(contexts)
        assert "=== Function 1" in prompt
        assert "=== Function 2" in prompt
        assert "=== Function 3" in prompt
        assert "func_0" in prompt
        assert "func_2" in prompt
        assert "0x00001000" in prompt


class TestParseBatchJson:
    def test_parses_json_array(self) -> None:
        raw = json.dumps(
            [
                {
                    "address": "0x1000",
                    "name": "foo",
                    "signature": "void foo(void)",
                    "confidence": 85,
                    "classification": "utility",
                    "comments": "",
                    "reasoning": "",
                },
                {
                    "address": "0x2000",
                    "name": "bar",
                    "signature": "int bar(int x)",
                    "confidence": 70,
                    "classification": "math",
                    "comments": "",
                    "reasoning": "",
                },
            ]
        )
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"
        assert results[0].confidence == 85

    def test_handles_markdown_fences(self) -> None:
        raw = '```json\n[{"address": "0x1000", "name": "test", "confidence": 50}]\n```'
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) == 1
        assert results[0].name == "test"

    def test_returns_empty_on_parse_failure(self) -> None:
        results = Analyzer.parse_llm_json_batch("not json at all")
        assert results == []

    def test_recovers_truncated_json_array(self) -> None:
        raw = '[{"name": "foo", "confidence": 80}, {"name": "bar", "confid'
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) >= 1
        assert results[0].name == "foo"

    def test_recovers_trailing_comma(self) -> None:
        raw = '[{"name": "foo", "confidence": 80,}]'
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) == 1
        assert results[0].name == "foo"


class TestParseLlmJsonRepair:
    def test_recovers_trailing_comma_single(self) -> None:
        raw = '{"name": "init_state", "confidence": 85, "classification": "init",}'
        result = Analyzer.parse_llm_json(raw)
        assert result.name == "init_state"
        assert result.confidence == 85

    def test_recovers_truncated_object(self) -> None:
        raw = '{"name": "process_data", "confidence": 70, "classification": "par'
        result = Analyzer.parse_llm_json(raw)
        assert result.name == "process_data"
