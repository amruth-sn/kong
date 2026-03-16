"""Tests for the SQLite configuration store."""

from __future__ import annotations

from kong.config import LLMProvider
from kong.db import (
    get_default_provider,
    get_enabled_providers,
    is_setup_complete,
    read_config,
    save_setup,
    write_config,
)


class TestConfigReadWrite:
    def test_read_missing_key_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert read_config("nonexistent") is None

    def test_write_then_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        write_config("foo", "bar")
        assert read_config("foo") == "bar"

    def test_upsert_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        write_config("key", "v1")
        write_config("key", "v2")
        assert read_config("key") == "v2"

    def test_creates_config_dir(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "nested"
        monkeypatch.setenv("KONG_CONFIG_DIR", str(nested))
        write_config("x", "y")
        assert nested.exists()
        assert read_config("x") == "y"


class TestSetupHelpers:
    def test_not_setup_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert not is_setup_complete()

    def test_save_setup_marks_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        assert is_setup_complete()

    def test_get_default_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.OPENAI,
        )
        assert get_default_provider() is LLMProvider.OPENAI

    def test_get_default_provider_returns_none_before_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert get_default_provider() is None

    def test_get_enabled_providers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.ANTHROPIC,
        )
        assert get_enabled_providers() == [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

    def test_get_enabled_providers_empty_before_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert get_enabled_providers() == []

    def test_save_setup_overwrites_previous(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        save_setup(
            enabled=[LLMProvider.OPENAI],
            default=LLMProvider.OPENAI,
        )
        assert get_default_provider() is LLMProvider.OPENAI
        assert get_enabled_providers() == [LLMProvider.OPENAI]
