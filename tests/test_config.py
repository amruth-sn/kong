from __future__ import annotations

from kong.config import LLMConfig, LLMProvider


class TestLLMProviderCustom:
    def test_custom_variant_exists(self):
        assert LLMProvider.CUSTOM.value == "custom"

    def test_custom_display_name(self):
        assert LLMProvider.CUSTOM.display_name == "Custom"

    def test_existing_display_names_unchanged(self):
        assert LLMProvider.ANTHROPIC.display_name == "Anthropic"
        assert LLMProvider.OPENAI.display_name == "OpenAI"


class TestLLMConfigCustomFields:
    def test_defaults_are_none(self):
        cfg = LLMConfig()
        assert cfg.base_url is None
        assert cfg.max_prompt_chars is None
        assert cfg.max_chunk_functions is None
        assert cfg.max_output_tokens is None

    def test_custom_config_with_all_fields(self):
        cfg = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            max_prompt_chars=32000,
            max_chunk_functions=20,
            max_output_tokens=4096,
        )
        assert cfg.provider is LLMProvider.CUSTOM
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.max_prompt_chars == 32000
