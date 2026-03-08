"""Tests for the eval harness and scoring metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kong.evals.metrics import (
    _normalize_name,
    _parse_signature,
    overall_score,
    symbol_accuracy,
    type_accuracy,
)
from kong.evals.harness import (
    GroundTruth,
    Scorecard,
    extract_ground_truth,
    load_analysis,
    match_functions,
    score,
)


class TestNormalizeName:
    def test_snake_case(self) -> None:
        assert _normalize_name("init_key") == ["init", "key"]

    def test_camel_case(self) -> None:
        assert _normalize_name("initKey") == ["init", "key"]

    def test_upper_camel_case(self) -> None:
        assert _normalize_name("InitKey") == ["init", "key"]

    def test_mixed(self) -> None:
        assert _normalize_name("my_initKey") == ["my", "init", "key"]

    def test_all_caps(self) -> None:
        tokens = _normalize_name("SSL")
        assert tokens == ["ssl"]


class TestSymbolAccuracy:
    def test_exact_match(self) -> None:
        assert symbol_accuracy("init_key", "init_key") == 1.0

    def test_word_reorder(self) -> None:
        result = symbol_accuracy("init_key", "key_init")
        assert 0.7 <= result <= 0.9

    def test_partial_overlap(self) -> None:
        result = symbol_accuracy("init_key_value", "init_buffer_size")
        assert 0.1 <= result <= 0.6

    def test_no_match(self) -> None:
        assert symbol_accuracy("foo", "bar") == 0.0

    def test_case_insensitive(self) -> None:
        result = symbol_accuracy("InitKey", "init_key")
        assert result >= 0.8


class TestParseSignature:
    def test_simple_signature(self) -> None:
        ret, params = _parse_signature("int add(int a, int b)")
        assert ret == "int"
        assert params == ["int", "int"]

    def test_void_return_void_params(self) -> None:
        ret, params = _parse_signature("void cleanup(void)")
        assert ret == "void"
        assert params == []

    def test_pointer_params(self) -> None:
        ret, params = _parse_signature("char * strdup(const char * src)")
        assert ret == "char *"
        assert params == ["const char *"]

    def test_no_params(self) -> None:
        ret, params = _parse_signature("int getpid()")
        assert ret == "int"
        assert params == []


class TestTypeAccuracy:
    def test_exact_match(self) -> None:
        result = type_accuracy("int add(int a, int b)", "int add(int a, int b)")
        assert result == 1.0

    def test_return_type_match_params_differ(self) -> None:
        result = type_accuracy(
            "int foo(float x, float y)",
            "int bar(int a, int b)",
        )
        assert 0.4 <= result <= 0.8

    def test_param_count_and_types_match(self) -> None:
        result = type_accuracy(
            "int process(int x, char * buf)",
            "int handle(int count, char * data)",
        )
        assert result >= 0.8

    def test_void_params(self) -> None:
        result = type_accuracy("void init(void)", "void init(void)")
        assert result == 1.0

    def test_ignores_qualifiers(self) -> None:
        result = type_accuracy(
            "unsigned int get_count(const int x)",
            "int get_count(int x)",
        )
        assert result >= 0.8


class TestExtractGroundTruth:
    def test_extracts_from_c_source(self, tmp_path: Path) -> None:
        c_source = tmp_path / "test.c"
        c_source.write_text(
            '#include <stdio.h>\n'
            '\n'
            'int add(int a, int b) {\n'
            '    return a + b;\n'
            '}\n'
            '\n'
            'void greet(const char *name) {\n'
            '    printf("Hello, %s\\n", name);\n'
            '}\n'
        )
        truth = extract_ground_truth(c_source)
        assert isinstance(truth, GroundTruth)
        assert len(truth.functions) == 2
        names = [f["name"] for f in truth.functions]
        assert "add" in names
        assert "greet" in names

    def test_extracts_static_functions(self, tmp_path: Path) -> None:
        c_source = tmp_path / "static.c"
        c_source.write_text(
            'static int helper(int x) {\n'
            '    return x * 2;\n'
            '}\n'
        )
        truth = extract_ground_truth(c_source)
        assert len(truth.functions) == 1
        assert truth.functions[0]["name"] == "helper"

    def test_extracts_signature(self, tmp_path: Path) -> None:
        c_source = tmp_path / "sig.c"
        c_source.write_text(
            'char * strdup(const char * src) {\n'
            '    return NULL;\n'
            '}\n'
        )
        truth = extract_ground_truth(c_source)
        assert len(truth.functions) == 1
        assert "char" in truth.functions[0]["signature"]
        assert "strdup" in truth.functions[0]["signature"]


class TestMatchFunctions:
    def test_matches_by_best_score(self) -> None:
        predicted = [
            {"name": "init_key", "signature": "void init_key(void)"},
            {"name": "process_data", "signature": "int process_data(int x)"},
        ]
        truth = [
            {"name": "init_key", "signature": "void init_key(void)"},
            {"name": "process_data", "signature": "int process_data(int x)"},
        ]
        matches = match_functions(predicted, truth)
        assert len(matches) == 2
        for pred, tr, sc in matches:
            assert sc == 1.0

    def test_no_reuse_of_truth_entries(self) -> None:
        predicted = [
            {"name": "foo", "signature": "void foo(void)"},
            {"name": "foo", "signature": "void foo(void)"},
        ]
        truth = [
            {"name": "foo", "signature": "void foo(void)"},
        ]
        matches = match_functions(predicted, truth)
        assert len(matches) == 2
        matched_truths = [tr for _, tr, _ in matches if tr is not None]
        assert len(matched_truths) == 1


class TestOverallScore:
    def test_returns_both_metrics(self) -> None:
        predicted = [
            {"name": "add", "signature": "int add(int a, int b)"},
        ]
        truth = [
            {"name": "add", "signature": "int add(int a, int b)"},
        ]
        result = overall_score(predicted, truth)
        assert "symbol_accuracy" in result
        assert "type_accuracy" in result
        assert result["symbol_accuracy"] == 1.0
        assert result["type_accuracy"] == 1.0


class TestLoadAnalysis:
    def test_loads_analysis_json(self, tmp_path: Path) -> None:
        analysis = {
            "binary": {"name": "test"},
            "stats": {"llm_calls": 5, "duration_seconds": 10.0, "cost_usd": 0.5},
            "functions": [
                {
                    "address": "0x00001000",
                    "original_name": "FUN_00001000",
                    "name": "add",
                    "signature": "int add(int a, int b)",
                    "confidence": 90,
                },
            ],
        }
        path = tmp_path / "analysis.json"
        path.write_text(json.dumps(analysis))
        entries = load_analysis(path)
        assert len(entries) == 1
        assert entries[0]["name"] == "add"


class TestScorecard:
    def test_scorecard_structure(self, tmp_path: Path) -> None:
        analysis = {
            "binary": {"name": "test_binary"},
            "stats": {
                "total_functions": 2,
                "analyzed": 1,
                "llm_calls": 3,
                "duration_seconds": 5.0,
                "cost_usd": 0.25,
            },
            "functions": [
                {
                    "address": "0x00001000",
                    "original_name": "FUN_00001000",
                    "name": "add",
                    "signature": "int add(int a, int b)",
                    "confidence": 95,
                },
            ],
        }
        analysis_path = tmp_path / "analysis.json"
        analysis_path.write_text(json.dumps(analysis))

        c_source = tmp_path / "source.c"
        c_source.write_text(
            'int add(int a, int b) {\n'
            '    return a + b;\n'
            '}\n'
            '\n'
            'int sub(int a, int b) {\n'
            '    return a - b;\n'
            '}\n'
        )

        result = score(analysis_path, c_source)
        assert isinstance(result, Scorecard)
        assert result.binary == "test_binary"
        assert result.total_functions == 2
        assert result.functions_analyzed == 1
        assert result.symbol_accuracy == 1.0
        assert result.type_accuracy == 1.0
        assert result.llm_calls == 3
        assert result.duration_seconds == 5.0
        assert result.cost_usd == 0.25
        assert len(result.per_function) == 1
