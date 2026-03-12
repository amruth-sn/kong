"""Tests for prompt templates."""

from __future__ import annotations

from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT, OUTPUT_SCHEMA, SYSTEM_PROMPT


class TestPrompts:
    def test_system_prompt_establishes_role(self):
        assert "reverse engineer" in SYSTEM_PROMPT.lower()

    def test_system_prompt_requests_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_output_schema_has_required_fields(self):
        for field in ["name", "signature", "confidence", "classification", "comments", "reasoning", "variables"]:
            assert field in OUTPUT_SCHEMA

    def test_output_schema_has_classification_values(self):
        for cls in ["crypto", "networking", "io", "memory", "string", "math", "init", "cleanup", "handler", "parser", "utility", "unknown"]:
            assert cls in OUTPUT_SCHEMA

    def test_output_schema_has_confidence_guidelines(self):
        assert "90-100" in OUTPUT_SCHEMA
        assert "0-29" in OUTPUT_SCHEMA


class TestBatchPrompts:
    def test_batch_output_schema_is_json_array(self) -> None:
        assert '"address"' in BATCH_OUTPUT_SCHEMA
        assert "[" in BATCH_OUTPUT_SCHEMA

    def test_batch_system_prompt_mentions_multiple(self) -> None:
        assert "multiple" in BATCH_SYSTEM_PROMPT.lower() or "batch" in BATCH_SYSTEM_PROMPT.lower()
