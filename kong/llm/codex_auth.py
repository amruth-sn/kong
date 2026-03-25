"""Helpers for resolving and refreshing Codex OAuth credentials."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass(frozen=True)
class CodexCredential:
    access_token: str
    account_id: str
    refresh_token: str | None = None
    auth_path: Path | None = None

    @property
    def source_label(self) -> str:
        return "Codex OAuth"


def codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def resolve_codex_credential() -> CodexCredential | None:
    auth_path = codex_auth_path()
    if not auth_path.is_file():
        return None

    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return _credential_from_auth(auth, auth_path=auth_path)


def refresh_codex_credential(credential: CodexCredential | None = None) -> CodexCredential | None:
    current = credential or resolve_codex_credential()
    if current is None or not current.refresh_token:
        return None

    payload = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": current.refresh_token,
        "client_id": CLIENT_ID,
    }).encode("utf-8")
    request = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        return None
    if not isinstance(refresh_token, str) or not refresh_token:
        return None

    account_id = current.account_id or _extract_account_id(access_token) or ""
    if not account_id:
        return None

    updated = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": int(time.time()),
    }

    auth_path = current.auth_path or codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")

    return CodexCredential(
        access_token=access_token,
        account_id=account_id,
        refresh_token=refresh_token,
        auth_path=auth_path,
    )


def _credential_from_auth(auth: dict[str, Any], *, auth_path: Path) -> CodexCredential | None:
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return None

    refresh_token = tokens.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = None

    account_id = tokens.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        account_id = _extract_account_id(access_token)
    if not account_id:
        return None

    return CodexCredential(
        access_token=access_token,
        account_id=account_id,
        refresh_token=refresh_token,
        auth_path=auth_path,
    )


def _extract_account_id(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return None

    direct = payload.get("account_id")
    if isinstance(direct, str) and direct:
        return direct

    nested = payload.get(JWT_CLAIM_PATH)
    if isinstance(nested, dict):
        for key in ("chatgpt_account_id", "account_id"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        parsed = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed
