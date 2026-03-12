"""Tests for the type recovery pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from kong.agent.analyzer import (
    Analyzer,
    LLMResponse,
    StructFieldProposal,
    StructProposal,
)
from kong.agent.models import FunctionResult
from kong.agent.type_recovery import (
    ParamTypeApplication,
    StructAccumulator,
    UnifiedStruct,
    apply_unified_structs,
    _pick_best_field,
    _resolve_param_ordinal,
)
from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    ParameterInfo,
    StructDefinition,
    StructField,
)


def _field(name: str, dtype: str, offset: int, size: int) -> StructFieldProposal:
    return StructFieldProposal(name=name, data_type=dtype, offset=offset, size=size)


def _proposal(
    name: str,
    total_size: int,
    fields: list[StructFieldProposal],
    used_by_param: str = "",
    source_function: int = 0,
) -> StructProposal:
    return StructProposal(
        name=name,
        total_size=total_size,
        fields=fields,
        used_by_param=used_by_param,
        source_function=source_function,
    )


# ---------------------------------------------------------------------------
# StructAccumulator
# ---------------------------------------------------------------------------

class TestStructAccumulator:
    def test_empty_accumulator(self):
        acc = StructAccumulator()
        assert acc.proposal_count == 0
        assert acc.unify() == []

    def test_add_proposals(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("conn_t", 32, [_field("fd", "int", 0, 4)]),
        ])
        assert acc.proposal_count == 1

    def test_tags_source_function(self):
        acc = StructAccumulator()
        p = _proposal("conn_t", 32, [_field("fd", "int", 0, 4)])
        acc.add_proposals(0x1000, [p])
        assert acc._proposals[0].source_function == 0x1000

    def test_single_proposal_unifies_to_one(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("config_t", 24, [
                _field("flags", "int", 0, 4),
                _field("timeout", "int", 4, 4),
            ]),
        ])
        unified = acc.unify()
        assert len(unified) == 1
        assert unified[0].definition.name == "config_t"
        assert unified[0].definition.size == 24
        assert len(unified[0].definition.fields) == 2

    def test_same_name_merges_fields(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("conn_t", 32, [
                _field("fd", "int", 0, 4),
                _field("flags", "int", 4, 4),
            ]),
        ])
        acc.add_proposals(0x2000, [
            _proposal("conn_t", 32, [
                _field("socket_fd", "int", 0, 4),
                _field("port", "uint16_t", 8, 2),
            ]),
        ])
        unified = acc.unify()
        assert len(unified) == 1
        assert unified[0].definition.size == 32
        assert len(unified[0].definition.fields) == 3

    def test_same_name_different_sizes_merge_to_max(self):
        """The cJSON scenario: same struct seen partially by different functions."""
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("cJSON", 24, [
                _field("type", "int", 0, 4),
                _field("valueint", "int", 8, 4),
            ]),
        ])
        acc.add_proposals(0x2000, [
            _proposal("cJSON", 64, [
                _field("type", "int", 0, 4),
                _field("valueint", "int", 8, 4),
                _field("valuedouble", "double", 16, 8),
                _field("string", "char *", 24, 8),
                _field("child", "void *", 40, 8),
                _field("next", "void *", 48, 8),
            ]),
        ])
        unified = acc.unify()
        assert len(unified) == 1
        assert unified[0].definition.name == "cJSON"
        assert unified[0].definition.size == 64
        assert len(unified[0].definition.fields) == 6

    def test_different_names_stay_separate(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("type_a", 16, [_field("x", "int", 0, 4)]),
        ])
        acc.add_proposals(0x2000, [
            _proposal("type_b", 16, [_field("y", "int", 8, 4)]),
        ])
        unified = acc.unify()
        assert len(unified) == 2

    def test_name_used_directly(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal("config_t", 16, [_field("a", "int", 0, 4)]),
        ])
        acc.add_proposals(0x2000, [
            _proposal("config_t", 32, [_field("a", "int", 0, 4), _field("b", "int", 8, 4)]),
        ])
        unified = acc.unify()
        assert len(unified) == 1
        assert unified[0].definition.name == "config_t"

    def test_applications_tracked(self):
        acc = StructAccumulator()
        acc.add_proposals(0x1000, [
            _proposal(
                "conn_t", 32,
                [_field("fd", "int", 0, 4)],
                used_by_param="param_1",
                source_function=0x1000,
            ),
        ])
        unified = acc.unify()
        assert len(unified[0].applications) == 1
        app = unified[0].applications[0]
        assert app.func_addr == 0x1000
        assert app.param_name == "param_1"
        assert app.struct_name == "conn_t"


# ---------------------------------------------------------------------------
# _pick_best_field
# ---------------------------------------------------------------------------

class TestPickBestField:
    def test_prefers_descriptive_name(self):
        candidates = [
            _field("field_0x0", "int", 0, 4),
            _field("socket_fd", "int", 0, 4),
        ]
        best = _pick_best_field(candidates)
        assert best.name == "socket_fd"

    def test_prefers_specific_type(self):
        candidates = [
            _field("count", "undefined4", 0, 4),
            _field("count", "int", 0, 4),
        ]
        best = _pick_best_field(candidates)
        assert best.data_type == "int"

    def test_single_candidate(self):
        candidates = [_field("x", "int", 0, 4)]
        best = _pick_best_field(candidates)
        assert best.name == "x"


# ---------------------------------------------------------------------------
# apply_unified_structs
# ---------------------------------------------------------------------------

class TestApplyUnifiedStructs:
    def test_creates_structs_and_applies_types(self):
        client = MagicMock()
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="process", size=100,
            params=[ParameterInfo(name="param_1", data_type="undefined *", ordinal=0)],
        )

        us = UnifiedStruct(
            definition=StructDefinition(
                name="conn_t",
                size=32,
                fields=[StructField(name="fd", data_type="int", offset=0, size=4)],
            ),
            applications=[
                ParamTypeApplication(func_addr=0x1000, param_name="param_1", struct_name="conn_t"),
            ],
        )

        affected = apply_unified_structs(client, [us])

        client.create_struct.assert_called_once_with(us.definition)
        client.apply_type_to_param.assert_called_once_with(
            0x1000, 0, "conn_t", as_pointer=True,
        )
        assert 0x1000 in affected

    def test_handles_create_failure(self):
        client = MagicMock()
        client.create_struct.side_effect = Exception("DTM error")

        us = UnifiedStruct(
            definition=StructDefinition(name="bad_t", size=8, fields=[]),
            applications=[
                ParamTypeApplication(func_addr=0x1000, param_name="param_1", struct_name="bad_t"),
            ],
        )

        affected = apply_unified_structs(client, [us])
        assert affected == []
        client.apply_type_to_param.assert_not_called()

    def test_handles_apply_failure(self):
        client = MagicMock()
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="f", size=100,
            params=[ParameterInfo(name="param_1", data_type="void *", ordinal=0)],
        )
        client.apply_type_to_param.side_effect = Exception("Apply failed")

        us = UnifiedStruct(
            definition=StructDefinition(name="t", size=8, fields=[]),
            applications=[
                ParamTypeApplication(func_addr=0x1000, param_name="param_1", struct_name="t"),
            ],
        )

        affected = apply_unified_structs(client, [us])
        assert affected == []

    def test_no_applications(self):
        client = MagicMock()
        us = UnifiedStruct(
            definition=StructDefinition(name="orphan_t", size=16, fields=[]),
        )
        affected = apply_unified_structs(client, [us])
        client.create_struct.assert_called_once()
        assert affected == []


# ---------------------------------------------------------------------------
# _resolve_param_ordinal
# ---------------------------------------------------------------------------

class TestResolveParamOrdinal:
    def test_resolves_by_name(self):
        client = MagicMock()
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="f", size=100,
            params=[
                ParameterInfo(name="param_1", data_type="int", ordinal=0),
                ParameterInfo(name="param_2", data_type="char *", ordinal=1),
            ],
        )
        assert _resolve_param_ordinal(client, 0x1000, "param_2") == 1

    def test_returns_none_for_unknown_param(self):
        client = MagicMock()
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="f", size=100,
            params=[ParameterInfo(name="param_1", data_type="int", ordinal=0)],
        )
        assert _resolve_param_ordinal(client, 0x1000, "param_99") is None

    def test_returns_none_on_error(self):
        client = MagicMock()
        client.get_function_info.side_effect = Exception("No function")
        assert _resolve_param_ordinal(client, 0x1000, "param_1") is None


# ---------------------------------------------------------------------------
# LLM JSON parsing — struct_proposals
# ---------------------------------------------------------------------------

class TestParseLLMJsonStructs:
    def test_parse_with_struct_proposals(self):
        raw = '''{
            "name": "process_request",
            "confidence": 75,
            "struct_proposals": [
                {
                    "name": "request_t",
                    "total_size": 24,
                    "fields": [
                        {"name": "method", "data_type": "int", "offset": 0, "size": 4},
                        {"name": "url", "data_type": "char *", "offset": 8, "size": 8}
                    ],
                    "used_by_param": "param_1"
                }
            ]
        }'''
        resp = Analyzer.parse_llm_json(raw)
        assert len(resp.struct_proposals) == 1
        sp = resp.struct_proposals[0]
        assert sp.name == "request_t"
        assert sp.total_size == 24
        assert len(sp.fields) == 2
        assert sp.fields[0].name == "method"
        assert sp.fields[1].data_type == "char *"
        assert sp.used_by_param == "param_1"

    def test_parse_without_struct_proposals(self):
        raw = '{"name": "foo", "confidence": 50}'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.struct_proposals == []

    def test_parse_malformed_struct_proposals_skipped(self):
        raw = '{"name": "foo", "struct_proposals": [{"not_a_name": true}]}'
        resp = Analyzer.parse_llm_json(raw)
        assert resp.struct_proposals == []

    def test_parse_struct_fields_with_defaults(self):
        raw = '''{
            "name": "f",
            "struct_proposals": [{
                "name": "s",
                "total_size": 8,
                "fields": [{"offset": 0}]
            }]
        }'''
        resp = Analyzer.parse_llm_json(raw)
        assert len(resp.struct_proposals) == 1
        f = resp.struct_proposals[0].fields[0]
        assert f.name == "field_0"
        assert f.data_type == "undefined"
        assert f.size == 4


# ---------------------------------------------------------------------------
# Analyzer context — known types in prompt
# ---------------------------------------------------------------------------

class TestAnalyzerContextTypes:
    def test_known_types_included_in_prompt(self):
        llm = MagicMock()
        llm.analyze_function.return_value = LLMResponse(name="f")
        client = MagicMock()
        client.get_decompilation.return_value = "void f(void *p) { return; }"
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="FUN_00001000", size=100,
            classification=FunctionClassification.MEDIUM,
        )
        client.get_xrefs_from.return_value = []

        from kong.agent.queue import WorkItem
        item = WorkItem(
            function=FunctionInfo(
                address=0x1000, name="FUN_00001000", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
            callers=[], callees=[],
        )
        binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC",
        )
        known_types = [
            StructDefinition(
                name="config_t",
                size=16,
                fields=[
                    StructField(name="flags", data_type="int", offset=0, size=4),
                    StructField(name="timeout", data_type="int", offset=4, size=4),
                ],
            ),
        ]

        analyzer = Analyzer(client, llm)
        analyzer.analyze(item, binary_info, {}, [], known_types=known_types)

        prompt = llm.analyze_function.call_args[0][0]
        assert "Known Struct Types" in prompt
        assert "config_t" in prompt
        assert "flags" in prompt
        assert "timeout" in prompt

    def test_no_known_types_section_when_empty(self):
        llm = MagicMock()
        llm.analyze_function.return_value = LLMResponse(name="f")
        client = MagicMock()
        client.get_decompilation.return_value = "void f() {}"
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="f", size=100,
            classification=FunctionClassification.MEDIUM,
        )
        client.get_xrefs_from.return_value = []

        from kong.agent.queue import WorkItem
        item = WorkItem(
            function=FunctionInfo(
                address=0x1000, name="f", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
            callers=[], callees=[],
        )
        binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC",
        )

        analyzer = Analyzer(client, llm)
        analyzer.analyze(item, binary_info, {}, [])

        prompt = llm.analyze_function.call_args[0][0]
        assert "Known Struct Types" not in prompt


# ---------------------------------------------------------------------------
# Analyzer — struct proposals flow through to FunctionResult
# ---------------------------------------------------------------------------

class TestAnalyzerStructProposalPassthrough:
    def test_struct_proposals_in_result(self):
        response = LLMResponse(
            name="process",
            confidence=70,
            struct_proposals=[
                StructProposal(
                    name="req_t",
                    total_size=16,
                    fields=[_field("method", "int", 0, 4)],
                    used_by_param="param_1",
                ),
            ],
        )
        llm = MagicMock()
        llm.analyze_function.return_value = response
        client = MagicMock()
        client.get_decompilation.return_value = "void f() {}"
        client.get_function_info.return_value = FunctionInfo(
            address=0x1000, name="f", size=100,
            classification=FunctionClassification.MEDIUM,
        )
        client.get_xrefs_from.return_value = []

        from kong.agent.queue import WorkItem
        item = WorkItem(
            function=FunctionInfo(
                address=0x1000, name="f", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
            callers=[], callees=[],
        )
        binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC",
        )

        analyzer = Analyzer(client, llm)
        result = analyzer.analyze(item, binary_info, {}, [])

        assert len(result.struct_proposals) == 1
        assert result.struct_proposals[0].name == "req_t"


# ---------------------------------------------------------------------------
# Ghidra types dataclasses
# ---------------------------------------------------------------------------

class TestStructDefinition:
    def test_field_count(self):
        sd = StructDefinition(
            name="test_t", size=16,
            fields=[
                StructField(name="a", data_type="int", offset=0, size=4),
                StructField(name="b", data_type="int", offset=4, size=4),
            ],
        )
        assert sd.field_count == 2

    def test_field_at_offset(self):
        sd = StructDefinition(
            name="test_t", size=16,
            fields=[
                StructField(name="a", data_type="int", offset=0, size=4),
                StructField(name="b", data_type="long", offset=8, size=8),
            ],
        )
        assert sd.field_at_offset(0).name == "a"
        assert sd.field_at_offset(8).name == "b"
        assert sd.field_at_offset(4) is None

    def test_struct_field_end_offset(self):
        f = StructField(name="x", data_type="int", offset=4, size=4)
        assert f.end_offset == 8


# ---------------------------------------------------------------------------
# Supervisor cleanup integration
# ---------------------------------------------------------------------------

class TestSupervisorCleanupIntegration:
    def test_cleanup_with_struct_proposals(self, tmp_path):
        from kong.agent.events import EventType, Phase
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        funcs = [
            FunctionInfo(
                address=0x1000, name="process", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
        ]
        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="test",
        )
        client.list_functions.return_value = funcs
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []
        client.get_decompilation.return_value = "void f() {}"
        client.get_function_info.return_value = funcs[0]
        client.get_xrefs_from.return_value = []
        client.list_custom_types.return_value = []

        response = LLMResponse(
            name="process_request",
            confidence=75,
            struct_proposals=[
                StructProposal(
                    name="req_t",
                    total_size=16,
                    fields=[_field("method", "int", 0, 4)],
                    used_by_param="param_1",
                ),
            ],
        )
        llm = MagicMock()
        llm.analyze_function.return_value = response

        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config, llm_client=llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        assert sup.struct_accumulator.proposal_count == 1

        event_types = [e.type for e in events]
        assert EventType.CLEANUP_TYPES_UNIFIED in event_types
        assert EventType.CLEANUP_TYPE_CREATED in event_types

    def test_cleanup_without_struct_proposals(self, tmp_path):
        from kong.agent.events import EventType
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        funcs = [
            FunctionInfo(
                address=0x1000, name="f", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
        ]
        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="test",
        )
        client.list_functions.return_value = funcs
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []
        client.get_decompilation.return_value = "void f() {}"
        client.get_xrefs_from.return_value = []

        response = LLMResponse(name="helper", confidence=80)
        llm = MagicMock()
        llm.analyze_function.return_value = response

        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config, llm_client=llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        assert sup.struct_accumulator.proposal_count == 0
        event_types = [e.type for e in events]
        assert EventType.CLEANUP_TYPES_UNIFIED not in event_types

    def test_cleanup_reanalyzes_low_confidence(self, tmp_path):
        from kong.agent.events import EventType, Phase
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        funcs = [
            FunctionInfo(
                address=0x1000, name="FUN_00001000", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
        ]
        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="test",
        )
        client.list_functions.return_value = funcs
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []
        client.get_decompilation.return_value = "void f() {}"
        client.get_function_info.return_value = funcs[0]
        client.get_xrefs_from.return_value = []
        client.list_custom_types.return_value = []

        batch_call_count = 0

        def varying_batch_response(prompt: str, **kwargs: object) -> list[LLMResponse]:
            nonlocal batch_call_count
            batch_call_count += 1
            if batch_call_count == 1:
                return [LLMResponse(name="unknown_func", confidence=30)]
            return [LLMResponse(name="decode_buffer", confidence=70)]

        llm = MagicMock()
        llm.analyze_function_batch.side_effect = varying_batch_response

        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config, llm_client=llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        assert sup.results[0x1000].name == "decode_buffer"
        assert sup.results[0x1000].confidence == 70

        cleanup_completes = [
            e for e in events
            if e.type == EventType.FUNCTION_COMPLETE and e.phase == Phase.CLEANUP
        ]
        assert len(cleanup_completes) == 1

    def test_cleanup_retries_failed_signatures(self, tmp_path):
        """Signatures that fail during Phase 2 (type not yet created) get
        retried after struct creation in cleanup."""
        from kong.agent.events import EventType
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        funcs = [
            FunctionInfo(
                address=0x1000, name="cJSON_IsTrue", size=100,
                classification=FunctionClassification.MEDIUM,
            ),
        ]
        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="cjson_test",
        )
        client.list_functions.return_value = funcs
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []
        client.get_decompilation.return_value = "int f(void *p) { return *(int *)(p + 4); }"
        client.get_function_info.return_value = funcs[0]
        client.get_xrefs_from.return_value = []
        client.list_custom_types.return_value = []

        sig_call_count = 0

        def set_sig_side_effect(addr, sig):
            nonlocal sig_call_count
            sig_call_count += 1
            if sig_call_count == 1:
                raise Exception("Can't resolve datatype: cJSON *")

        client.set_function_signature.side_effect = set_sig_side_effect

        response = LLMResponse(
            name="cJSON_IsTrue",
            signature="int cJSON_IsTrue(cJSON *item)",
            confidence=95,
            struct_proposals=[
                StructProposal(
                    name="cJSON",
                    total_size=64,
                    fields=[
                        _field("type", "int", 0, 4),
                        _field("valueint", "int", 4, 4),
                    ],
                    used_by_param="param_1",
                ),
            ],
        )
        llm = MagicMock()
        llm.analyze_function.return_value = response

        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config, llm_client=llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        assert sup.results[0x1000].signature == "int cJSON_IsTrue(cJSON *item)"
        assert sup.results[0x1000].signature_applied is True

        event_types = [e.type for e in events]
        assert EventType.CLEANUP_SIGNATURES_RETRIED in event_types

        retry_event = next(
            e for e in events if e.type == EventType.CLEANUP_SIGNATURES_RETRIED
        )
        assert retry_event.data["succeeded"] == 1
