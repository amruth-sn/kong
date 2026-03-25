from __future__ import annotations

import json

from kong.llm.openai_auth import codex_auth_path, resolve_openai_credential


def test_resolve_openai_credential_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    auth_path = codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-codex-test",
        "tokens": {"access_token": "oauth-token"},
    }))

    credential = resolve_openai_credential()

    assert credential is not None
    assert credential.source == "env"
    assert credential.token == "sk-env-test"


def test_resolve_openai_credential_uses_codex_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    auth_path = codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-codex-test",
        "tokens": {"access_token": "oauth-token"},
    }))

    credential = resolve_openai_credential()

    assert credential is not None
    assert credential.source == "codex_api_key"
    assert credential.token == "sk-codex-test"


def test_resolve_openai_credential_uses_codex_oauth(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    auth_path = codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({
        "OPENAI_API_KEY": None,
        "tokens": {"access_token": "oauth-token"},
    }))

    credential = resolve_openai_credential()

    assert credential is not None
    assert credential.source == "codex_oauth"
    assert credential.token == "oauth-token"
