from __future__ import annotations

import json
from unittest.mock import patch

from kong.llm.codex_auth import refresh_codex_credential, resolve_codex_credential


def test_resolve_codex_credential_reads_auth_file(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "oauth-token",
            "refresh_token": "refresh-token",
            "account_id": "acct_123",
        },
    }))

    credential = resolve_codex_credential()

    assert credential is not None
    assert credential.access_token == "oauth-token"
    assert credential.refresh_token == "refresh-token"
    assert credential.account_id == "acct_123"


@patch("kong.llm.codex_auth.os.chmod")
@patch("kong.llm.codex_auth.urllib.request.urlopen")
def test_refresh_codex_credential_updates_auth_file(mock_urlopen, mock_chmod, tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({
        "tokens": {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "account_id": "acct_123",
        },
    }))

    mock_response = mock_urlopen.return_value.__enter__.return_value
    mock_response.read.return_value = json.dumps({
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
    }).encode("utf-8")

    credential = refresh_codex_credential()

    assert credential is not None
    assert credential.access_token == "new-access"
    assert credential.refresh_token == "new-refresh"
    assert credential.account_id == "acct_123"
    mock_chmod.assert_called_once_with(auth_path, 0o600)
    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == "new-access"
    assert saved["tokens"]["refresh_token"] == "new-refresh"
    assert saved["tokens"]["account_id"] == "acct_123"
