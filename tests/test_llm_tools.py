"""Tests for kong.llm.tools — tool schemas, executor, and tool use loop."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kong.llm.client import AnthropicClient
from kong.llm.tools import (
    DEOBFUSCATION_TOOLS,
    ToolExecutor,
)


class TestToolSchemas:
    def test_all_tools_have_required_fields(self):
        for tool in DEOBFUSCATION_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]
            assert "required" in tool["input_schema"]

    def test_expected_tools_present(self):
        names = {t["name"] for t in DEOBFUSCATION_TOOLS}
        assert "simplify_expression" in names
        assert "eliminate_dead_code" in names
        assert "trace_state_machine" in names
        assert "identify_crypto_constants" in names
        assert "get_decompilation" in names
        assert "get_basic_blocks" in names


class TestToolExecutor:
    @pytest.fixture
    def mock_ghidra(self):
        return MagicMock()

    @pytest.fixture
    def executor(self, mock_ghidra):
        return ToolExecutor(mock_ghidra)

    def test_simplify_expression(self, executor):
        result = executor.execute(
            "simplify_expression",
            {
                "expression": "(x ^ x) == 0",
            },
        )
        data = json.loads(result)
        assert data["is_opaque_predicate"] is True
        assert data["predicate_kind"] == "always_true"

    def test_simplify_expression_with_bit_width(self, executor):
        result = executor.execute(
            "simplify_expression",
            {
                "expression": "3 + 5",
                "bit_width": 64,
            },
        )
        data = json.loads(result)
        assert data["simplified"] == "8"

    def test_eliminate_dead_code(self, executor):
        result = executor.execute(
            "eliminate_dead_code",
            {
                "decompiled_code": "if (x ^ x) { dead(); } live();",
                "resolved_predicates": [
                    {"expression": "x ^ x", "resolution": "always_false"},
                ],
            },
        )
        assert "dead" not in result
        assert "live" in result

    def test_get_decompilation(self, executor, mock_ghidra):
        mock_ghidra.get_decompilation.return_value = "void main() {}"
        result = executor.execute(
            "get_decompilation",
            {
                "function_address": 0x401000,
            },
        )
        assert "void main" in result
        mock_ghidra.get_decompilation.assert_called_once_with(0x401000)

    def test_identify_crypto_constants(self, executor, mock_ghidra):
        mock_ghidra.get_decompilation.return_value = (
            "int sbox[] = { 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5 };"
        )
        result = executor.execute(
            "identify_crypto_constants",
            {
                "function_address": 0x401000,
            },
        )
        data = json.loads(result)
        assert len(data["matches"]) > 0
        assert any("AES" in m["algorithm"] for m in data["matches"])

    def test_identify_crypto_constants_no_match(self, executor, mock_ghidra):
        mock_ghidra.get_decompilation.return_value = "void foo() { return; }"
        result = executor.execute(
            "identify_crypto_constants",
            {
                "function_address": 0x401000,
            },
        )
        data = json.loads(result)
        assert data["matches"] == []

    def test_unknown_tool(self, executor):
        result = executor.execute("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_handles_execution_error(self, executor, mock_ghidra):
        mock_ghidra.get_decompilation.side_effect = Exception("Ghidra crashed")
        result = executor.execute(
            "get_decompilation",
            {
                "function_address": 0xDEAD,
            },
        )
        assert "Error" in result
        assert executor.call_count == 1


class TestToolUseLoop:
    def _mock_text_message(self, text, input_tokens=100, output_tokens=50):
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = text
        msg.content = [block]
        msg.stop_reason = "end_turn"
        msg.usage.input_tokens = input_tokens
        msg.usage.output_tokens = output_tokens
        return msg

    def _mock_tool_use_message(self, tool_name, tool_input, tool_id="tool_1"):
        msg = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_input
        tool_block.id = tool_id
        msg.content = [tool_block]
        msg.stop_reason = "tool_use"
        msg.usage.input_tokens = 200
        msg.usage.output_tokens = 100
        return msg

    def test_no_tool_calls(self):

        json_response = json.dumps(
            {
                "name": "simple_func",
                "signature": "void simple_func(void)",
                "confidence": 85,
                "classification": "utility",
                "comments": "A simple function",
                "reasoning": "No obfuscation",
            }
        )

        with patch("kong.llm.client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_text_message(
                json_response
            )

            client = AnthropicClient(api_key="test")
            executor = ToolExecutor(MagicMock())
            response = client.analyze_with_tools(
                prompt="test",
                system="test system",
                tools=DEOBFUSCATION_TOOLS,
                tool_executor=executor,
            )
            assert response.name == "simple_func"
            assert response.confidence == 85

    def test_single_tool_call(self):

        json_response = json.dumps(
            {
                "name": "deobfuscated_func",
                "signature": "int deobfuscated_func(int x)",
                "confidence": 72,
                "classification": "crypto",
                "comments": "Deobfuscated",
                "reasoning": "Used simplifier",
            }
        )

        tool_msg = self._mock_tool_use_message(
            "simplify_expression",
            {"expression": "(x ^ x) == 0"},
        )
        final_msg = self._mock_text_message(json_response)

        with patch("kong.llm.client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = [tool_msg, final_msg]

            client = AnthropicClient(api_key="test")
            executor = ToolExecutor(MagicMock())
            response = client.analyze_with_tools(
                prompt="test",
                system="test system",
                tools=DEOBFUSCATION_TOOLS,
                tool_executor=executor,
            )
            assert response.name == "deobfuscated_func"
            assert executor.call_count == 1
            assert mock_client.messages.create.call_count == 2

    def test_max_rounds_limit(self):

        tool_msg = self._mock_tool_use_message(
            "simplify_expression",
            {"expression": "x + 1"},
        )

        with patch("kong.llm.client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = tool_msg

            client = AnthropicClient(api_key="test")
            executor = ToolExecutor(MagicMock())
            _ = client.analyze_with_tools(
                prompt="test",
                system="test system",
                tools=DEOBFUSCATION_TOOLS,
                tool_executor=executor,
                max_rounds=3,
            )
            assert mock_client.messages.create.call_count == 3
