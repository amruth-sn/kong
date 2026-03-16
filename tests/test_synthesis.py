"""Tests for the semantic synthesis module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kong.agent.models import FunctionResult
from kong.synthesis.semantic import (
    SYNTHESIS_FUNCTION_CAP,
    SemanticSynthesizer,
    SynthesisResult,
)


@dataclass
class FakeLLMResponse:
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class FakeLLMClient:
    """Minimal mock satisfying the LLMClient protocol for synthesis tests."""

    def __init__(self, raw_response: str = "{}") -> None:
        self._raw_response = raw_response
        self.last_prompt: str = ""
        self.last_model: str | None = None

    def analyze_function(
        self, prompt: str, *, model: str | None = None
    ) -> FakeLLMResponse:
        self.last_prompt = prompt
        self.last_model = model
        return FakeLLMResponse(raw=self._raw_response)

    def analyze_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, object]],
        tool_executor: object,
        max_rounds: int = 10,
    ) -> FakeLLMResponse:
        return FakeLLMResponse(raw=self._raw_response)


SAMPLE_DECOMP_A = """\
void FUN_00401000(void) {
    int local_8 = DAT_100008000;
    int local_c = DAT_100004000;
    printf("%d", local_8 + local_c);
}
"""

SAMPLE_DECOMP_B = """\
void FUN_00402000(void) {
    int val = DAT_100008000;
    return val * 2;
}
"""

SAMPLE_DECOMP_NO_GLOBALS = """\
int add(int a, int b) {
    return a + b;
}
"""


def _make_result(
    address: int,
    name: str = "",
    classification: str = "utility",
    confidence: int = 70,
) -> FunctionResult:
    return FunctionResult(
        address=address,
        original_name=f"FUN_{address:08x}",
        name=name or f"func_{address:x}",
        classification=classification,
        confidence=confidence,
    )


class TestGlobalExtraction:
    def test_extracts_globals_from_multiple_functions(self) -> None:
        decompilations = {
            0x401000: SAMPLE_DECOMP_A,
            0x402000: SAMPLE_DECOMP_B,
        }
        synth = SemanticSynthesizer(FakeLLMClient())
        globals_map, xref_counts = synth._extract_globals(decompilations)

        assert "DAT_100008000" in globals_map
        assert globals_map["DAT_100008000"] == {0x401000, 0x402000}

        assert "DAT_100004000" in globals_map
        assert globals_map["DAT_100004000"] == {0x401000}

    def test_no_globals_when_none_present(self) -> None:
        decompilations = {
            0x401000: SAMPLE_DECOMP_NO_GLOBALS,
        }
        synth = SemanticSynthesizer(FakeLLMClient())
        globals_map, xref_counts = synth._extract_globals(decompilations)
        assert globals_map == {}


class TestPromptBuilding:
    def test_includes_multi_use_globals(self) -> None:
        results = [
            _make_result(0x401000, "process_data"),
            _make_result(0x402000, "compute"),
        ]
        decompilations = {0x401000: SAMPLE_DECOMP_A, 0x402000: SAMPLE_DECOMP_B}
        synth = SemanticSynthesizer(FakeLLMClient())
        prompt = synth._build_synthesis_prompt(results, decompilations)

        assert "DAT_100008000" in prompt
        assert "process_data" in prompt or "compute" in prompt

    def test_excludes_single_use_globals(self) -> None:
        results = [
            _make_result(0x401000, "process_data"),
            _make_result(0x402000, "compute"),
        ]
        decompilations = {0x401000: SAMPLE_DECOMP_A, 0x402000: SAMPLE_DECOMP_B}
        synth = SemanticSynthesizer(FakeLLMClient())
        prompt = synth._build_synthesis_prompt(results, decompilations)

        global_section_start = prompt.find("## Global Variables")
        functions_section_start = prompt.find("## Functions and Decompilations")
        if global_section_start != -1 and functions_section_start != -1:
            global_section = prompt[global_section_start:functions_section_start]
            assert "DAT_100004000" not in global_section

    def test_includes_all_functions(self) -> None:
        results = [_make_result(0x401000, "alpha"), _make_result(0x402000, "beta")]
        decompilations = {
            0x401000: SAMPLE_DECOMP_NO_GLOBALS,
            0x402000: SAMPLE_DECOMP_NO_GLOBALS,
        }
        synth = SemanticSynthesizer(FakeLLMClient())
        prompt = synth._build_synthesis_prompt(results, decompilations)

        assert "alpha" in prompt
        assert "beta" in prompt
        assert "0x00401000" in prompt
        assert "0x00402000" in prompt


class TestResponseParsing:
    def test_parse_valid_json(self) -> None:
        raw = """{
            "globals": {"DAT_100008000": "g_counter"},
            "structs": [{"name": "Point", "fields": [{"name": "x", "type": "int", "offset": 0}]}],
            "name_refinements": {"0x00401000": "init_counter"}
        }"""
        result = SemanticSynthesizer._parse_response(raw)
        assert result.globals == {"DAT_100008000": "g_counter"}
        assert len(result.structs) == 1
        assert result.structs[0]["name"] == "Point"
        assert result.name_refinements == {"0x00401000": "init_counter"}

    def test_parse_json_in_markdown_fence(self) -> None:
        raw = """Some preamble text
```json
{
    "globals": {"DAT_100008000": "g_config"},
    "structs": [],
    "name_refinements": {}
}
```
Trailing text"""
        result = SemanticSynthesizer._parse_response(raw)
        assert result.globals == {"DAT_100008000": "g_config"}

    def test_parse_empty_response(self) -> None:
        result = SemanticSynthesizer._parse_response("not valid json at all")
        assert result.globals == {}
        assert result.structs == []
        assert result.name_refinements == {}

    def test_parse_partial_response(self) -> None:
        raw = '{"globals": {"DAT_100008000": "g_flag"}}'
        result = SemanticSynthesizer._parse_response(raw)
        assert result.globals == {"DAT_100008000": "g_flag"}
        assert result.structs == []
        assert result.name_refinements == {}


class TestSynthesisCap:
    def test_synthesis_caps_at_50_functions(self) -> None:

        results = [_make_result(i, f"func_{i}") for i in range(100)]
        decomps = {i: f"void func_{i}(void) {{ DAT_1000 = 1; }}" for i in range(100)}

        synth = SemanticSynthesizer(FakeLLMClient())
        prompt = synth._build_synthesis_prompt(results, decomps)

        func_count = prompt.count("### func_")
        assert func_count <= SYNTHESIS_FUNCTION_CAP

    def test_prioritizes_functions_with_most_xrefs(self) -> None:
        results = [_make_result(i, f"func_{i}") for i in range(60)]
        decomps = {}
        for i in range(60):
            if i < 5:
                decomps[i] = (
                    f"void func_{i}(void) {{ DAT_A = 1; DAT_B = 2; DAT_C = 3; }}"
                )
            else:
                decomps[i] = f"void func_{i}(void) {{ return; }}"

        synth = SemanticSynthesizer(FakeLLMClient())
        prompt = synth._build_synthesis_prompt(results, decomps)

        for i in range(5):
            assert f"func_{i}" in prompt


class TestGlobalApplication:
    def test_apply_globals_replaces_dat_names(self) -> None:
        code = "int val = DAT_100008000 + DAT_100008000;"
        globals_map = {"DAT_100008000": "g_counter"}
        result = SemanticSynthesizer._apply_globals(code, globals_map)
        assert result == "int val = g_counter + g_counter;"

    def test_apply_globals_preserves_unmapped(self) -> None:
        code = "int a = DAT_100004000; int b = DAT_100008000;"
        globals_map = {"DAT_100008000": "g_counter"}
        result = SemanticSynthesizer._apply_globals(code, globals_map)
        assert "DAT_100004000" in result
        assert "g_counter" in result

    def test_apply_to_decompilations_all_functions(self) -> None:
        decompilations = {
            0x401000: "use DAT_100008000;",
            0x402000: "read DAT_100008000;",
        }
        synthesis_result = SynthesisResult(
            globals={"DAT_100008000": "g_value"},
            structs=[],
            name_refinements={},
        )
        synth = SemanticSynthesizer(FakeLLMClient())
        updated = synth.apply_to_decompilations(synthesis_result, decompilations)
        assert updated[0x401000] == "use g_value;"
        assert updated[0x402000] == "read g_value;"


class TestSynthesizeIntegration:
    def test_synthesize_calls_llm_and_returns_result(self) -> None:
        raw_response = """{
            "globals": {"DAT_100008000": "g_state"},
            "structs": [],
            "name_refinements": {"0x00401000": "initialize_state"}
        }"""
        llm = FakeLLMClient(raw_response=raw_response)
        synth = SemanticSynthesizer(llm)

        results = [
            _make_result(0x401000, "process_data"),
            _make_result(0x402000, "compute"),
        ]
        decompilations = {0x401000: SAMPLE_DECOMP_A, 0x402000: SAMPLE_DECOMP_B}

        synthesis = synth.synthesize(results, decompilations)

        assert llm.last_prompt != ""
        assert "DAT_100008000" in llm.last_prompt
        assert synthesis.globals == {"DAT_100008000": "g_state"}
        assert synthesis.name_refinements == {"0x00401000": "initialize_state"}

    def test_synthesize_with_model_override(self) -> None:
        llm = FakeLLMClient(
            raw_response='{"globals": {}, "structs": [], "name_refinements": {}}'
        )
        synth = SemanticSynthesizer(llm)

        results = [_make_result(0x401000)]
        decompilations = {0x401000: SAMPLE_DECOMP_NO_GLOBALS}

        synth.synthesize(results, decompilations, model="claude-haiku-4-5-20251001")
        assert llm.last_model == "claude-haiku-4-5-20251001"
