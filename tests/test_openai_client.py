"""Tests for the OpenAI LLM client wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from kong.llm.openai_client import OpenAIClient, _convert_tools_to_openai
from kong.llm.usage import ModelTokenUsage, TokenUsage


def _mock_response(text: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Create a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = None

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestOpenAIClient:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_analyze_function_parses_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "parse_config", "confidence": 85, "classification": "parser"}'
        )

        client = OpenAIClient(api_key="test-key")
        response = client.analyze_function("analyze this function")

        assert response.name == "parse_config"
        assert response.confidence == 85
        assert response.classification == "parser"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_tracks_token_usage(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}', prompt_tokens=200, completion_tokens=80
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("prompt1")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 80
        assert client.usage.calls == 1

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_accumulates_across_calls(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}', prompt_tokens=100, completion_tokens=50
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 100
        assert client.usage.calls == 2

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_sends_system_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}'
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("test prompt")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "reverse engineer" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test prompt"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_uses_json_mode_for_single_analysis(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}'
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("prompt")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_model_override(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}'
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("prompt", model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_per_model_cost_tracking(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "f"}', prompt_tokens=100, completion_tokens=50
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2", model="gpt-4o-mini")

        assert len(client.usage.by_model) == 2
        assert "gpt-4o" in client.usage.by_model
        assert "gpt-4o-mini" in client.usage.by_model
        assert client.usage.calls == 2

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_handles_malformed_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            "I cannot analyze this function because reasons."
        )

        client = OpenAIClient(api_key="test-key")
        response = client.analyze_function("prompt")

        assert response.name == ""
        assert "Failed to parse" in response.reasoning

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_handles_empty_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = _mock_response("")
        resp.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = resp

        client = OpenAIClient(api_key="test-key")
        response = client.analyze_function("prompt")

        assert response.name == ""

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_records_cached_tokens(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = _mock_response('{"name": "f"}', prompt_tokens=200, completion_tokens=50)
        details = MagicMock()
        details.cached_tokens = 100
        resp.usage.prompt_tokens_details = details
        mock_client.chat.completions.create.return_value = resp

        client = OpenAIClient(api_key="test-key")
        client.analyze_function("prompt")

        mu = client.usage.by_model["gpt-4o"]
        assert mu.cache_read_tokens == 100


class TestOpenAIBatch:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_batch_returns_list_of_responses(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            json.dumps(
                [
                    {"address": "0x1000", "name": "foo", "confidence": 80},
                    {"address": "0x2000", "name": "bar", "confidence": 70},
                ]
            )
        )

        client = OpenAIClient(api_key="test-key")
        results = client.analyze_function_batch("batch prompt")

        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_batch_uses_batch_system_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '[{"name": "f", "confidence": 50}]'
        )

        client = OpenAIClient(api_key="test-key")
        client.analyze_function_batch("prompt")

        call_kwargs = mock_client.chat.completions.create.call_args
        system_text = call_kwargs.kwargs["messages"][0]["content"]
        assert "multiple" in system_text.lower() or "batch" in system_text.lower()


class TestOpenAIToolUse:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_tool_loop_executes_tools(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "simplify_expression"
        tool_call.function.arguments = '{"expression": "x + 0"}'

        tool_response = _mock_response("", prompt_tokens=100, completion_tokens=20)
        tool_response.choices[0].finish_reason = "tool_calls"
        tool_response.choices[0].message.content = None
        tool_response.choices[0].message.tool_calls = [tool_call]

        final_response = _mock_response(
            '{"name": "clean_func", "confidence": 75}',
            prompt_tokens=150,
            completion_tokens=40,
        )

        mock_client.chat.completions.create.side_effect = [
            tool_response,
            final_response,
        ]

        executor = MagicMock()
        executor.execute.return_value = '{"simplified": "x"}'

        tools = [
            {
                "name": "simplify_expression",
                "description": "Simplify an expression.",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            }
        ]

        client = OpenAIClient(api_key="test-key")
        result = client.analyze_with_tools("prompt", "system", tools, executor)

        assert result.name == "clean_func"
        assert result.input_tokens == 250
        assert result.output_tokens == 60
        executor.execute.assert_called_once_with(
            "simplify_expression", {"expression": "x + 0"}
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_max_rounds_exhausted_uses_last_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tool_call = MagicMock()
        tool_call.id = "call_abc"
        tool_call.function.name = "lookup"
        tool_call.function.arguments = '{"key": "val"}'

        def make_tool_response(content: str | None):
            resp = _mock_response("", prompt_tokens=50, completion_tokens=20)
            resp.choices[0].finish_reason = "tool_calls"
            resp.choices[0].message.content = content
            resp.choices[0].message.tool_calls = [tool_call]
            return resp

        mock_client.chat.completions.create.side_effect = [
            make_tool_response(None),
            make_tool_response('{"name": "last_round", "confidence": 60}'),
        ]

        executor = MagicMock()
        executor.execute.return_value = '{"result": "ok"}'

        tools = [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "input_schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            }
        ]

        client = OpenAIClient(api_key="test-key")
        result = client.analyze_with_tools(
            "prompt", "system", tools, executor, max_rounds=2
        )

        assert result.name == "last_round"
        assert result.confidence == 60
        assert result.input_tokens == 100
        assert result.output_tokens == 40


class TestConvertTools:
    def test_converts_anthropic_to_openai_format(self):
        anthropic_tools = [
            {
                "name": "test_tool",
                "description": "A test tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]

        openai_tools = _convert_tools_to_openai(anthropic_tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "test_tool"
        assert openai_tools[0]["function"]["description"] == "A test tool."
        assert openai_tools[0]["function"]["parameters"]["type"] == "object"
        assert "x" in openai_tools[0]["function"]["parameters"]["properties"]


class TestOpenAICostCalculation:
    def test_gpt4o_cost(self):
        mu = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = mu.cost_usd("gpt-4o")
        assert cost == 2.50 + 10.00

    def test_gpt4o_mini_cost(self):
        mu = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = mu.cost_usd("gpt-4o-mini")
        assert cost == 0.15 + 0.60

    def test_gpt4o_cached_tokens_cost(self):
        mu = ModelTokenUsage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        cost = mu.cost_usd("gpt-4o")
        assert cost == 1.25
