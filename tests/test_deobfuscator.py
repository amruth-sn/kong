"""Tests for kong.agent.deobfuscator — classification heuristics, pattern loading,
and the Deobfuscator orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kong.agent.analyzer import AnalysisContext, Analyzer, LLMResponse
from kong.agent.deobfuscator import (
    Deobfuscator,
    ObfuscationType,
    classify_obfuscation,
    load_patterns,
)
from kong.agent.models import FunctionResult
from kong.agent.prompts import DEOBFUSCATION_SYSTEM_PROMPT
from kong.agent.queue import WorkItem
from kong.ghidra.types import BinaryInfo, FunctionInfo
from kong.llm.tools import DEOBFUSCATION_TOOLS

_CFF_CODE = """\
void FUN_004015a0(int *data, int len) {
    int state = 0x3a2b;
    while (1) {
        switch (state) {
            case 0x3a2b:
                if (len <= 0) { state = 0x9e22; }
                else { state = 0x7f01; }
                break;
            case 0x7f01:
                data[0] ^= 0x55;
                state = 0x1c44;
                break;
            case 0x1c44:
                data++;
                len--;
                state = 0x3a2b;
                break;
            case 0x9e22:
                return;
        }
    }
}
"""

_BOGUS_CF_CODE = """\
int FUN_00401000(int a, int b) {
    int result;
    if ((a * (a + 1)) % 2 == 0) {
        result = a + b;
    } else {
        result = a ^ b ^ 0xDEAD;
    }
    if ((b & ~b) != 0) {
        result ^= 0xBEEF;
    }
    if ((a | ~a) == -1) {
        result += 1;
    }
    return result;
}
"""

_INSTRUCTION_SUB_CODE = """\
int FUN_00403000(int a, int b) {
    int t1 = (~a & b) | (a & ~b);
    int t2 = (a ^ b) + 2 * (a & b);
    return ~t1 + 1;
}
"""

_STRING_ENCRYPT_CODE = """\
void FUN_00404000(void) {
    char buf[14] = { 0x2a, 0x27, 0x2e, 0x2e, 0x29, 0x4e, 0x42, 0x3f, 0x29, 0x36, 0x2e, 0x26, 0x43, 0x42 };
    for (int i = 0; i < 14; i++) {
        buf[i] ^= 0x42;
    }
    puts(buf);
}
"""

_CLEAN_CODE = """\
int add(int a, int b) {
    return a + b;
}
"""


class TestClassifyObfuscation:
    def test_detects_cff(self):
        techniques = classify_obfuscation(_CFF_CODE)
        assert ObfuscationType.CONTROL_FLOW_FLATTENING in techniques

    def test_detects_bogus_cf(self):
        techniques = classify_obfuscation(_BOGUS_CF_CODE)
        assert ObfuscationType.BOGUS_CONTROL_FLOW in techniques

    def test_detects_instruction_sub(self):
        techniques = classify_obfuscation(_INSTRUCTION_SUB_CODE)
        assert ObfuscationType.INSTRUCTION_SUBSTITUTION in techniques

    def test_detects_string_encrypt(self):
        techniques = classify_obfuscation(_STRING_ENCRYPT_CODE)
        assert ObfuscationType.STRING_ENCRYPTION in techniques

    def test_clean_code_no_obfuscation(self):
        techniques = classify_obfuscation(_CLEAN_CODE)
        assert techniques == []

    def test_combined_cff_and_bogus(self):
        combined = _CFF_CODE + "\n" + _BOGUS_CF_CODE
        techniques = classify_obfuscation(combined)
        assert ObfuscationType.CONTROL_FLOW_FLATTENING in techniques
        assert ObfuscationType.BOGUS_CONTROL_FLOW in techniques

    def test_vmprotect_needs_many_cases(self):
        code = "void dispatch(void) {\n"
        code += "while (1) {\n"
        code += "  unsigned char opcode = *vip++;\n"
        code += "  switch (opcode) {\n"
        for i in range(25):
            code += f"    case {i}: vstack[vsp++] = reg[{i % 8}]; break;\n"
        code += "  }\n}\n}\n"
        padding_lines = 210 - code.count("\n")
        code += "\n".join(f"// padding line {i}" for i in range(padding_lines))
        techniques = classify_obfuscation(code)
        assert ObfuscationType.VM_PROTECTION in techniques


class TestPatternLibrary:
    def test_load_cff_pattern(self):
        content = load_patterns([ObfuscationType.CONTROL_FLOW_FLATTENING])
        assert "Control Flow Flattening" in content
        assert "while(1)" in content or "while (1)" in content

    def test_load_bogus_cf_pattern(self):
        content = load_patterns([ObfuscationType.BOGUS_CONTROL_FLOW])
        assert "Bogus Control Flow" in content

    def test_load_multiple_patterns(self):
        content = load_patterns(
            [
                ObfuscationType.CONTROL_FLOW_FLATTENING,
                ObfuscationType.BOGUS_CONTROL_FLOW,
            ]
        )
        assert "Control Flow Flattening" in content
        assert "Bogus Control Flow" in content
        assert "---" in content

    def test_load_all_patterns(self):
        all_types = list(ObfuscationType)
        content = load_patterns(all_types)
        assert "Control Flow Flattening" in content
        assert "Instruction Substitution" in content
        assert "VM-Based Protection" in content


def _make_context(decompilation: str = "void foo() {}") -> AnalysisContext:
    return AnalysisContext(
        function=FunctionInfo(address=0x401000, name="FUN_00401000", size=200),
        decompilation=decompilation,
        binary_info=BinaryInfo(
            arch="x86",
            format="ELF",
            endianness="little",
            word_size=8,
            compiler="gcc",
        ),
    )


class TestDeobfuscator:
    @pytest.fixture
    def mock_ghidra(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.analyze_with_tools.return_value = LLMResponse(
            name="deobfuscated_func",
            signature="int deobfuscated_func(int x)",
            confidence=75,
            classification="crypto",
            comments="Deobfuscated CFF function",
            reasoning="Used state machine tracer",
        )
        return llm

    def test_deobfuscate_returns_response(self, mock_ghidra, mock_llm):
        deob = Deobfuscator(mock_ghidra, mock_llm)
        context = _make_context(_CFF_CODE)
        response, tool_calls = deob.deobfuscate(
            context,
            [ObfuscationType.CONTROL_FLOW_FLATTENING],
        )
        assert response.name == "deobfuscated_func"
        assert response.confidence == 75
        mock_llm.analyze_with_tools.assert_called_once()

    def test_prompt_includes_pattern_context(self, mock_ghidra, mock_llm):
        deob = Deobfuscator(mock_ghidra, mock_llm)
        context = _make_context(_CFF_CODE)
        deob.deobfuscate(context, [ObfuscationType.CONTROL_FLOW_FLATTENING])
        call_kwargs = mock_llm.analyze_with_tools.call_args
        prompt = (
            call_kwargs.kwargs.get("prompt")
            or call_kwargs[1].get("prompt")
            or call_kwargs[0][0]
        )
        assert "Control Flow Flattening" in prompt or "cff" in prompt.lower()

    def test_prompt_includes_technique_list(self, mock_ghidra, mock_llm):
        deob = Deobfuscator(mock_ghidra, mock_llm)
        context = _make_context(_CFF_CODE)
        deob.deobfuscate(
            context,
            [
                ObfuscationType.CONTROL_FLOW_FLATTENING,
                ObfuscationType.BOGUS_CONTROL_FLOW,
            ],
        )
        call_kwargs = mock_llm.analyze_with_tools.call_args
        prompt = (
            call_kwargs.kwargs.get("prompt")
            or call_kwargs[1].get("prompt")
            or call_kwargs[0][0]
        )
        assert "cff" in prompt
        assert "bogus_cf" in prompt

    def test_passes_deobfuscation_system_prompt(self, mock_ghidra, mock_llm):
        deob = Deobfuscator(mock_ghidra, mock_llm)
        context = _make_context()
        deob.deobfuscate(context, [ObfuscationType.BOGUS_CONTROL_FLOW])
        call_kwargs = mock_llm.analyze_with_tools.call_args
        system = (
            call_kwargs.kwargs.get("system")
            or call_kwargs[1].get("system")
            or call_kwargs[0][1]
        )
        assert system == DEOBFUSCATION_SYSTEM_PROMPT

    def test_passes_tools(self, mock_ghidra, mock_llm):
        deob = Deobfuscator(mock_ghidra, mock_llm)
        context = _make_context()
        deob.deobfuscate(context, [ObfuscationType.INSTRUCTION_SUBSTITUTION])
        call_kwargs = mock_llm.analyze_with_tools.call_args
        tools = (
            call_kwargs.kwargs.get("tools")
            or call_kwargs[1].get("tools")
            or call_kwargs[0][2]
        )
        assert tools == DEOBFUSCATION_TOOLS


class TestAnalyzerDeobfuscationIntegration:
    def test_obfuscated_function_delegates(self):
        mock_ghidra = MagicMock()
        mock_ghidra.get_decompilation.return_value = _CFF_CODE
        mock_ghidra.get_xrefs_from.return_value = []

        mock_llm = MagicMock()
        mock_llm.analyze_with_tools.return_value = LLMResponse(
            name="xor_decrypt",
            confidence=80,
            classification="crypto",
        )

        deob = Deobfuscator(mock_ghidra, mock_llm)

        analyzer = Analyzer(mock_ghidra, mock_llm, deobfuscator=deob)
        item = WorkItem(
            function=FunctionInfo(address=0x401000, name="FUN_00401000", size=200),
            callers=[],
            callees=[],
        )
        result = analyzer.analyze(
            item,
            binary_info=BinaryInfo(
                arch="x86",
                format="ELF",
                endianness="little",
                word_size=8,
                compiler="gcc",
            ),
            known_results={},
            strings=[],
        )
        assert result.name == "xor_decrypt"
        assert "cff" in result.obfuscation_techniques
        mock_llm.analyze_with_tools.assert_called_once()

    def test_clean_function_uses_normal_path(self):

        mock_ghidra = MagicMock()
        mock_ghidra.get_decompilation.return_value = _CLEAN_CODE
        mock_ghidra.get_xrefs_from.return_value = []

        mock_llm = MagicMock()
        mock_llm.analyze_function.return_value = LLMResponse(
            name="add",
            confidence=95,
            classification="math",
        )

        deob = Deobfuscator(mock_ghidra, mock_llm)

        analyzer = Analyzer(mock_ghidra, mock_llm, deobfuscator=deob)
        item = WorkItem(
            function=FunctionInfo(address=0x402000, name="FUN_00402000", size=20),
            callers=[],
            callees=[],
        )
        result = analyzer.analyze(
            item,
            binary_info=BinaryInfo(
                arch="x86",
                format="ELF",
                endianness="little",
                word_size=8,
                compiler="gcc",
            ),
            known_results={},
            strings=[],
        )
        assert result.name == "add"
        assert result.obfuscation_techniques == []
        mock_llm.analyze_function.assert_called_once()
        mock_llm.analyze_with_tools.assert_not_called()
