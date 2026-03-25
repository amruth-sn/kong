from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from kong.llm.codex_client import CodexClient


class _FakeStreamingResponse:
    def __init__(self, body: str):
        self._lines = iter(body.encode("utf-8").splitlines(keepends=True))

    def readline(self) -> bytes:
        return next(self._lines, b"")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _completed_event(output: list[dict[str, object]], *, input_tokens: int = 10, output_tokens: int = 5) -> str:
    return (
        "event: response.completed\n"
        f"data: {json.dumps({'response': {'output': output, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}})}\n\n"
        "data: [DONE]\n\n"
    )


@patch("kong.llm.codex_client.resolve_codex_credential")
@patch("kong.llm.codex_client.urllib.request.urlopen")
def test_analyze_function_parses_json(mock_urlopen, mock_resolve_credential):
    mock_resolve_credential.return_value = MagicMock(access_token="oauth", account_id="acct", refresh_token="refresh")
    mock_urlopen.return_value = _FakeStreamingResponse(_completed_event([
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": json.dumps({
                        "name": "hash_string",
                        "confidence": 91,
                        "classification": "string",
                        "comments": "Hashes a string",
                    }),
                }
            ],
        }
    ]))

    client = CodexClient(model="gpt-5-codex")
    response = client.analyze_function("analyze this function")

    assert response.name == "hash_string"
    assert response.confidence == 91
    assert response.input_tokens == 10
    assert response.output_tokens == 5


@patch("kong.llm.codex_client.resolve_codex_credential")
@patch("kong.llm.codex_client.urllib.request.urlopen")
def test_analyze_with_tools_executes_tool_calls(mock_urlopen, mock_resolve_credential):
    mock_resolve_credential.return_value = MagicMock(access_token="oauth", account_id="acct", refresh_token="refresh")
    mock_urlopen.side_effect = [
        _FakeStreamingResponse(_completed_event([
            {
                "type": "function_call",
                "name": "lookup_symbol",
                "call_id": "call_123",
                "arguments": json.dumps({"name": "Foo"}),
            }
        ], input_tokens=20, output_tokens=4)),
        _FakeStreamingResponse(_completed_event([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({
                            "name": "resolve_foo",
                            "confidence": 88,
                            "classification": "utility",
                            "comments": "Uses the looked up symbol",
                        }),
                    }
                ],
            }
        ], input_tokens=15, output_tokens=7)),
    ]

    executor = MagicMock()
    executor.execute.return_value = '{"symbol":"Foo"}'

    client = CodexClient(model="gpt-5-codex")
    response = client.analyze_with_tools(
        prompt="use tools",
        system="system",
        tools=[
            {
                "name": "lookup_symbol",
                "description": "Lookup a symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ],
        tool_executor=executor,
    )

    executor.execute.assert_called_once_with("lookup_symbol", {"name": "Foo"})
    assert response.name == "resolve_foo"
    assert response.input_tokens == 35
    assert response.output_tokens == 11
