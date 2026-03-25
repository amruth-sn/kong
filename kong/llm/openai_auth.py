"""Helpers for resolving OpenAI credentials, including Codex OAuth."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CredentialSource = Literal["env", "codex_api_key", "codex_oauth"]


@dataclass(frozen=True)
class OpenAICredential:
    token: str
    source: CredentialSource

    @property
    def source_label(self) -> str:
        return {
            "env": "OPENAI_API_KEY",
            "codex_api_key": "Codex API key",
            "codex_oauth": "Codex OAuth",
        }[self.source]

    @property
    def masked_value(self) -> str | None:
        if self.source == "codex_oauth":
            return None
        if len(self.token) <= 11:
            return "***"
        return self.token[:7] + "..." + self.token[-4:]


def codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def resolve_openai_credential() -> OpenAICredential | None:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return OpenAICredential(token=env_key, source="env")

    auth_path = codex_auth_path()
    if not auth_path.is_file():
        return None

    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    codex_api_key = auth.get("OPENAI_API_KEY")
    if isinstance(codex_api_key, str) and codex_api_key:
        return OpenAICredential(token=codex_api_key, source="codex_api_key")

    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access_token = tokens.get("access_token")
    if isinstance(access_token, str) and access_token:
        return OpenAICredential(token=access_token, source="codex_oauth")

    return None
