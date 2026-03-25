"""Codex ChatGPT-backend client for function analysis."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT, OUTPUT_SCHEMA, SYSTEM_PROMPT
from kong.llm.codex_auth import CodexCredential, refresh_codex_credential, resolve_codex_credential
from kong.llm.tools import ToolExecutor
from kong.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-codex"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


class CodexClientError(RuntimeError):
    """Raised when the Codex backend rejects a request."""


@dataclass
class _CodexEnvelope:
    output_text: str
    response: dict[str, Any]
    input_tokens: int
    output_tokens: int
    outputs: list[dict[str, Any]]


class CodexClient:
    """Concrete LLM client using the ChatGPT Codex backend."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.usage = TokenUsage()

    def analyze_function(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        effective_model = model or self.model
        envelope = self._request(
            instructions=f"{SYSTEM_PROMPT}\n\n{OUTPUT_SCHEMA}",
            input_items=[self._user_message(prompt)],
            model=effective_model,
        )

        result = Analyzer.parse_llm_json(envelope.output_text)
        result.input_tokens = envelope.input_tokens
        result.output_tokens = envelope.output_tokens
        result.raw = envelope.output_text
        return result

    def analyze_function_batch(self, prompt: str, *, model: str | None = None) -> list[LLMResponse]:
        effective_model = model or self.model
        envelope = self._request(
            instructions=f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}",
            input_items=[self._user_message(prompt)],
            model=effective_model,
        )

        responses = Analyzer.parse_llm_json_batch(envelope.output_text)
        for resp in responses:
            resp.input_tokens = envelope.input_tokens
            resp.output_tokens = envelope.output_tokens
        return responses

    def analyze_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        max_rounds: int = 10,
    ) -> LLMResponse:
        input_items: list[dict[str, Any]] = [self._user_message(prompt)]
        total_input = 0
        total_output = 0
        last_text = ""

        for _ in range(max_rounds):
            envelope = self._request(
                instructions=f"{system}\n\n{OUTPUT_SCHEMA}",
                input_items=input_items,
                tools=self._convert_tools(tools),
                model=self.model,
            )

            total_input += envelope.input_tokens
            total_output += envelope.output_tokens
            last_text = envelope.output_text

            tool_calls = [item for item in envelope.outputs if item.get("type") == "function_call"]
            if not tool_calls:
                result = Analyzer.parse_llm_json(last_text)
                result.input_tokens = total_input
                result.output_tokens = total_output
                result.raw = last_text
                return result

            for tool_call in tool_calls:
                raw_args = tool_call.get("arguments", "") or "{}"
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}

                result_str = tool_executor.execute(tool_call["name"], arguments)
                input_items.append({
                    "type": "function_call",
                    "call_id": tool_call["call_id"],
                    "name": tool_call["name"],
                    "arguments": raw_args,
                })
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": result_str,
                })

        result = Analyzer.parse_llm_json(last_text)
        result.input_tokens = total_input
        result.output_tokens = total_output
        result.raw = last_text
        return result

    @property
    def total_cost_usd(self) -> float:
        return 0.0

    def probe(self) -> bool:
        envelope = self._request(
            instructions='Reply with exactly this JSON object: {"ok": true}',
            input_items=[self._user_message("ping")],
            model=self.model,
        )
        return '"ok"' in envelope.output_text

    def _request(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> _CodexEnvelope:
        credential = resolve_codex_credential()
        if credential is None:
            raise CodexClientError("Codex OAuth credentials not found. Run `codex` to sign in.")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._perform_request(
                    credential=credential,
                    instructions=instructions,
                    input_items=input_items,
                    model=model,
                    tools=tools,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 401 and credential.refresh_token and attempt == 0:
                    refreshed = refresh_codex_credential(credential)
                    if refreshed is not None:
                        credential = refreshed
                        continue
                body = exc.read().decode("utf-8", "replace")
                raise CodexClientError(self._format_http_error(exc.code, body)) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                raise CodexClientError(f"Could not connect to Codex backend: {exc.reason}") from exc

        raise CodexClientError(str(last_error) if last_error else "Codex request failed")

    def _perform_request(
        self,
        *,
        credential: CodexCredential,
        instructions: str,
        input_items: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> _CodexEnvelope:
        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": instructions,
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "text": {"verbosity": "medium"},
            "reasoning": {"effort": "medium"},
        }
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if tools:
            payload["tools"] = tools

        request = urllib.request.Request(
            CODEX_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        for key, value in {
            "Authorization": f"Bearer {credential.access_token}",
            "chatgpt-account-id": credential.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }.items():
            request.add_header(key, value)

        with urllib.request.urlopen(request, timeout=120) as response:
            events = self._read_sse_events(response)

        completed_response: dict[str, Any] | None = None
        output_text_parts: list[str] = []
        for event_name, payload_data in events:
            if event_name == "response.output_text.done":
                text = payload_data.get("text")
                if isinstance(text, str):
                    output_text_parts.append(text)
            elif event_name == "response.completed":
                completed_response = payload_data.get("response")

        if completed_response is None:
            raise CodexClientError("Codex backend did not return a completed response.")

        outputs = completed_response.get("output", [])
        if not isinstance(outputs, list):
            outputs = []

        if not output_text_parts:
            for item in outputs:
                if item.get("type") != "message":
                    continue
                for part in item.get("content", []):
                    text = part.get("text")
                    if isinstance(text, str):
                        output_text_parts.append(text)

        usage = completed_response.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self._record_usage(model, input_tokens, output_tokens)

        return _CodexEnvelope(
            output_text="".join(output_text_parts),
            response=completed_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            outputs=[item for item in outputs if isinstance(item, dict)],
        )

    @staticmethod
    def _user_message(prompt: str) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
            ],
        }

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": True,
            })
        return converted

    @staticmethod
    def _read_sse_events(response: Any) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        event_name = ""
        data_lines: list[str] = []

        while True:
            raw_line = response.readline()
            if not raw_line:
                break

            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    if data != "[DONE]":
                        try:
                            events.append((event_name, json.loads(data)))
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON SSE payload: %s", data[:200])
                event_name = ""
                data_lines = []
                continue

            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        return events

    def _record_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        usage = self.usage._get(model)
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.calls += 1
        logger.debug(
            "Codex [%s]: %d in / %d out tokens (total: %d calls)",
            model,
            input_tokens,
            output_tokens,
            self.usage.calls,
        )

    @staticmethod
    def _format_http_error(status: int, body: str) -> str:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return f"HTTP {status}: {message}"
            detail = parsed.get("detail")
            if isinstance(detail, str) and detail:
                return f"HTTP {status}: {detail}"

        body = body.strip()
        if body:
            return f"HTTP {status}: {body[:200]}"
        return f"HTTP {status}"
