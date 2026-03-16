"""Persistent configuration store backed by SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from kong.config import LLMProvider

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "kong"
_DB_FILENAME = "config.db"


def _config_dir() -> Path:
    return Path(os.environ.get("KONG_CONFIG_DIR", str(_DEFAULT_CONFIG_DIR)))


def get_config_db() -> Path:
    directory = _config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _DB_FILENAME


def _connect() -> sqlite3.Connection:
    db_path = get_config_db()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def read_config(key: str) -> str | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def write_config(key: str, value: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def write_configs(pairs: dict[str, str]) -> None:
    conn = _connect()
    try:
        conn.executemany(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(pairs.items()),
        )
        conn.commit()
    finally:
        conn.close()


def is_setup_complete() -> bool:
    return read_config("setup_complete") == "true"


def get_default_provider() -> LLMProvider | None:
    raw = read_config("default_provider")
    if raw is None:
        return None
    try:
        return LLMProvider(raw)
    except ValueError:
        return None


def get_enabled_providers() -> list[LLMProvider]:
    raw = read_config("enabled_providers")
    if raw is None:
        return []
    try:
        return [LLMProvider(v) for v in json.loads(raw)]
    except (json.JSONDecodeError, ValueError):
        return []


def save_setup(
    enabled: list[LLMProvider],
    default: LLMProvider,
) -> None:
    write_configs(
        {
            "enabled_providers": json.dumps([p.value for p in enabled]),
            "default_provider": default.value,
            "setup_complete": "true",
        }
    )
