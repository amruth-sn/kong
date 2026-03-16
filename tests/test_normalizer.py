from __future__ import annotations


from kong.normalizer.syntactic import normalize


class TestNegativeLiteralCleanup:
    def test_simple_plus_negative(self) -> None:
        assert normalize("x + -5") == "x - 5"

    def test_hex_negative(self) -> None:
        assert normalize("*local_8 + -0x54") == "*local_8 - 0x54"

    def test_multiple_occurrences(self) -> None:
        code = "a + -1 + b + -2"
        assert normalize(code) == "a - 1 + b - 2"

    def test_no_false_positive_minus_minus(self) -> None:
        code = "x - -5"
        assert normalize(code) == "x - -5"

    def test_negative_in_parenthesized_expression(self) -> None:
        assert normalize("(*local_8 + -0x54)") == "(*local_8 - 0x54)"

    def test_negative_literal_with_cast(self) -> None:
        code = "(char)(*local_8 + -0x54)"
        assert normalize(code) == "(char)(*local_8 - 0x54)"


class TestModuloRecovery:
    def test_simple_modulo_subtraction_form(self) -> None:
        assert normalize("x - (x / 5) * 5") == "x % 5"

    def test_modulo_negative_multiply_form(self) -> None:
        assert normalize("x + (x / 10) * -10") == "x % 10"

    def test_modulo_hex_values(self) -> None:
        assert normalize("x + (x / 0x1a) * -0x1a") == "x % 0x1a"

    def test_no_false_positive_different_variables(self) -> None:
        code = "x - (y / 5) * 5"
        assert normalize(code) == "x - (y / 5) * 5"

    def test_no_false_positive_different_divisor_multiplier(self) -> None:
        code = "x - (x / 5) * 6"
        assert normalize(code) == "x - (x / 5) * 6"

    def test_complex_expression_modulo(self) -> None:
        code = "(*local_8 - 0x54) + ((*local_8 - 0x54) / 0x1a) * -0x1a"
        result = normalize(code)
        assert "% 0x1a" in result
        assert "* -0x1a" not in result

    def test_modulo_with_surrounding_code(self) -> None:
        code = "(char)((*local_8 - 0x54) % 0x1a) + 'a'"
        assert normalize(code) == code

    def test_rot13_full_line(self) -> None:
        code = "*local_8 = (char)(*local_8 + -0x54) + (char)((*local_8 + -0x54) / 0x1a) * -0x1a + 'a';"
        result = normalize(code)
        assert "+ -" not in result
        assert "% 0x1a" in result


class TestUndefinedTypeInference:
    def test_loop_counter_for_loop(self) -> None:
        code = "undefined4 local_14;\nfor (local_14 = 0; local_14 < length; local_14 = local_14 + 1)"
        result = normalize(code)
        assert "int local_14" in result
        assert "undefined4" not in result

    def test_loop_counter_increment_pp(self) -> None:
        code = "undefined4 i;\nfor (i = 0; i < n; i++)"
        result = normalize(code)
        assert "int i" in result

    def test_accumulator_pattern(self) -> None:
        code = "undefined4 sum;\nsum = 0;\nsum += x;"
        result = normalize(code)
        assert "int sum" in result

    def test_accumulator_init_zero_and_add(self) -> None:
        code = "undefined4 total;\ntotal = 0;\ntotal = total + item;"
        result = normalize(code)
        assert "int total" in result

    def test_undefined8_null_comparison(self) -> None:
        code = "undefined8 ptr;\nif (ptr == 0)"
        result = normalize(code)
        assert "long ptr" in result

    def test_undefined8_null_literal(self) -> None:
        code = "undefined8 ptr;\nif (ptr == NULL)"
        result = normalize(code)
        assert "long ptr" in result

    def test_undefined8_dat_global(self) -> None:
        code = "undefined8 val;\nval = DAT_00100040;"
        result = normalize(code)
        assert "long val" in result

    def test_undefined8_pointer_cast(self) -> None:
        code = "undefined8 mem;\n*(int *)mem = 5;"
        result = normalize(code)
        assert "long mem" in result

    def test_no_inference_without_usage_pattern(self) -> None:
        code = "undefined4 mystery;\nfoo(mystery);"
        result = normalize(code)
        assert "undefined4 mystery" in result

    def test_multiple_variables_inferred(self) -> None:
        code = "undefined4 i;\nundefined4 sum;\nfor (i = 0; i < n; i++)\nsum = 0;\nsum += val;"
        result = normalize(code)
        assert "int i" in result
        assert "int sum" in result


class TestDeadNullAssignmentRemoval:
    def test_simple_dead_null_assignment(self) -> None:
        code = (
            "if (local_18 == (undefined4 *)0x0) {\n    local_18 = (undefined4 *)0x0;\n}"
        )
        result = normalize(code)
        assert "local_18 = (undefined4 *)0x0;" not in result
        assert "if (local_18 == (undefined4 *)0x0) {" in result

    def test_dead_null_different_type(self) -> None:
        code = "if (ptr == (char *)0x0) {\n    ptr = (char *)0x0;\n}"
        result = normalize(code)
        assert "ptr = (char *)0x0;" not in result

    def test_non_dead_assignment_preserved(self) -> None:
        code = "if (ptr == (char *)0x0) {\n    ptr = malloc(10);\n}"
        result = normalize(code)
        assert "ptr = malloc(10);" in result

    def test_dead_null_with_other_statements(self) -> None:
        code = 'if (var == (int *)0x0) {\n    var = (int *)0x0;\n    printf("null");\n}'
        result = normalize(code)
        assert "var = (int *)0x0;" not in result
        assert 'printf("null");' in result


class TestPipelineOrdering:
    def test_negative_literal_before_modulo(self) -> None:
        code = "x + (x / 0x1a) * -0x1a"
        result = normalize(code)
        assert "% 0x1a" in result

    def test_full_rot13_pipeline(self) -> None:
        code = (
            "*local_8 = (char)(*local_8 + -0x54) + "
            "(char)((*local_8 + -0x54) / 0x1a) * -0x1a + 'a';"
        )
        result = normalize(code)
        assert "+ -" not in result
        assert "% 0x1a" in result

    def test_create_node_dead_assignment(self) -> None:
        code = (
            "if (local_18 == (undefined4 *)0x0) {\n    local_18 = (undefined4 *)0x0;\n}"
        )
        result = normalize(code)
        assert "local_18 = (undefined4 *)0x0;" not in result
