"""Tests for the supervisor agent loop."""

from __future__ import annotations

from unittest.mock import MagicMock

from kong.agent.analyzer import LLMResponse
from kong.agent.events import EventType, Phase
from kong.agent.models import AnalysisStats, FunctionResult
from kong.agent.supervisor import Supervisor
from kong.config import KongConfig, OutputConfig
from kong.ghidra.types import BinaryInfo, FunctionClassification, FunctionInfo


def _make_client(functions=None, binary_info=None):
    """Create a mock GhidraClient."""
    client = MagicMock()
    client.get_binary_info.return_value = binary_info or BinaryInfo(
        arch="x86-64",
        format="ELF",
        endianness="little",
        word_size=8,
        compiler="GCC",
        name="test_binary",
    )
    client.list_functions.return_value = functions or []
    client.get_strings.return_value = []
    client.get_callers.return_value = []
    client.get_callees.return_value = []
    client.get_decompilation.return_value = "void stub(void) { return; }"
    return client


def _func(addr, name, size=100, cls=FunctionClassification.MEDIUM):
    return FunctionInfo(address=addr, name=name, size=size, classification=cls)


def _batch_response_with_addresses(prompt: str, **kwargs: object) -> list[LLMResponse]:
    """Mock batch response that extracts addresses from chunk prompt headers."""
    import re

    addresses = re.findall(r"### (0x[0-9a-fA-F]+):", prompt)
    return [
        LLMResponse(
            name=f"func_{i}",
            confidence=80,
            classification="utility",
            address=int(addr, 16),
        )
        for i, addr in enumerate(addresses)
    ]


class TestSupervisorLifecycle:
    def test_run_emits_start_and_complete(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        types = [e.type for e in events]
        assert types[0] == EventType.RUN_START
        assert types[-1] == EventType.RUN_COMPLETE

    def test_all_five_phases_run(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        phase_starts = [e.phase for e in events if e.type == EventType.PHASE_START]
        expected = [
            Phase.TRIAGE,
            Phase.ANALYSIS,
            Phase.CLEANUP,
            Phase.SYNTHESIS,
            Phase.EXPORT,
        ]
        assert phase_starts == expected

    def test_run_with_no_functions(self, tmp_path):
        client = _make_client(functions=[])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        results = sup.run()
        assert results == {}


class TestSupervisorTriage:
    def test_triage_enumerates_functions(self, tmp_path):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        enum_events = [
            e for e in events if e.type == EventType.TRIAGE_FUNCTIONS_ENUMERATED
        ]
        assert len(enum_events) == 1
        assert enum_events[0].data["total"] == 2

    def test_triage_builds_queue(self, tmp_path):
        funcs = [
            _func(0x1000, "a"),
            _func(0x2000, "b", cls=FunctionClassification.THUNK),
        ]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        queue_events = [e for e in events if e.type == EventType.TRIAGE_QUEUE_BUILT]
        assert len(queue_events) == 1
        assert queue_events[0].data["queue_size"] == 1


class TestSupervisorAnalysis:
    def test_skips_trivial_functions(self, tmp_path):
        funcs = [_func(0x1000, "tiny", size=10, cls=FunctionClassification.TRIVIAL)]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        results = sup.run()

        assert 0x1000 in results
        assert results[0x1000].skipped is True
        assert results[0x1000].skip_reason == "trivial"

    def test_produces_results_for_each_function(self, tmp_path):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b"), _func(0x3000, "c")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        results = sup.run()

        assert len(results) == 3
        for addr in [0x1000, 0x2000, 0x3000]:
            assert addr in results

    def test_chunk_analysis_emits_events(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="",
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        starts = [e for e in events if e.type == EventType.FUNCTION_START]
        assert len(starts) >= 1
        assert any(e.data["address"] == 0x1000 for e in starts)

        completes = [e for e in events if e.type == EventType.FUNCTION_COMPLETE]
        assert len(completes) >= 1

    def test_handles_chunk_error(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = RuntimeError("LLM timeout")
        mock_llm.analyze_function.return_value = LLMResponse(
            name="",
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        results = sup.run()

        assert results[0x1000].error
        error_events = [e for e in events if e.type == EventType.FUNCTION_ERROR]
        assert len(error_events) == 1


class TestAnalysisStats:
    def test_record_result_renamed(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000,
            original_name="FUN_1000",
            name="init_module",
            confidence=85,
            llm_calls=2,
        )
        stats.record_result(result)

        assert stats.analyzed == 1
        assert stats.renamed == 1
        assert stats.named == 1
        assert stats.high_confidence == 1
        assert stats.llm_calls == 2

    def test_record_result_skipped(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000,
            original_name="FUN_1000",
            skipped=True,
            skip_reason="trivial",
        )
        stats.record_result(result)

        assert stats.skipped == 1
        assert stats.analyzed == 0

    def test_record_result_error(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000,
            original_name="FUN_1000",
            error="timeout",
        )
        stats.record_result(result)

        assert stats.errors == 1
        assert stats.analyzed == 0

    def test_record_result_unchanged_name_is_confirmed(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000,
            original_name="FUN_1000",
            name="FUN_1000",
            confidence=30,
            llm_calls=1,
        )
        stats.record_result(result)

        assert stats.analyzed == 1
        assert stats.renamed == 0
        assert stats.confirmed == 1
        assert stats.named == 1
        assert stats.low_confidence == 1

    def test_confidence_buckets(self):
        stats = AnalysisStats(total_functions=10)

        stats.record_result(
            FunctionResult(
                address=1, original_name="a", name="x", confidence=90, llm_calls=1
            )
        )
        stats.record_result(
            FunctionResult(
                address=2, original_name="b", name="y", confidence=60, llm_calls=1
            )
        )
        stats.record_result(
            FunctionResult(
                address=3, original_name="c", name="z", confidence=30, llm_calls=1
            )
        )

        assert stats.high_confidence == 1
        assert stats.medium_confidence == 1
        assert stats.low_confidence == 1

    def test_name_rate(self):
        stats = AnalysisStats(total_functions=4)
        stats.renamed = 2
        stats.confirmed = 1
        assert stats.name_rate == 0.75


class TestSupervisorExport:
    def test_export_creates_source_file(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(
            output=OutputConfig(
                directory=tmp_path / "out",
                formats=["source"],
            )
        )
        sup = Supervisor(client, config)
        sup.run()

        assert (tmp_path / "out" / "decompiled.c").exists()

    def test_export_creates_json_file(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(
            output=OutputConfig(
                directory=tmp_path / "out",
                formats=["json"],
            )
        )
        sup = Supervisor(client, config)
        sup.run()

        assert (tmp_path / "out" / "analysis.json").exists()

    def test_export_emits_export_file_events(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(
            output=OutputConfig(
                directory=tmp_path / "out",
                formats=["source", "json"],
            )
        )
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        export_events = [e for e in events if e.type == EventType.EXPORT_FILE]
        assert len(export_events) == 2
        formats_emitted = {e.data["format"] for e in export_events}
        assert formats_emitted == {"source", "json"}

    def test_export_skips_ghidra_format(self, tmp_path):
        client = _make_client()
        config = KongConfig(
            output=OutputConfig(
                directory=tmp_path / "out",
                formats=["ghidra"],
            )
        )
        sup = Supervisor(client, config)
        sup.run()


class TestSupervisorSynthesis:
    def test_synthesis_phase_runs(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function.return_value = LLMResponse(
            name="init",
            confidence=80,
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        phase_starts = [e.phase for e in events if e.type == EventType.PHASE_START]
        assert Phase.SYNTHESIS in phase_starts

        phase_completes = [
            e.phase for e in events if e.type == EventType.PHASE_COMPLETE
        ]
        assert Phase.SYNTHESIS in phase_completes

    def test_synthesis_skipped_without_llm_client(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        synthesis_completes = [
            e
            for e in events
            if e.type == EventType.PHASE_COMPLETE and e.phase == Phase.SYNTHESIS
        ]
        assert len(synthesis_completes) == 1
        assert "skipped" in synthesis_completes[0].message.lower()

    def test_synthesis_skipped_with_no_results(self, tmp_path):
        client = _make_client(functions=[])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        synthesis_completes = [
            e
            for e in events
            if e.type == EventType.PHASE_COMPLETE and e.phase == Phase.SYNTHESIS
        ]
        assert len(synthesis_completes) == 1
        assert "skipped" in synthesis_completes[0].message.lower()


class TestDecompilationCache:
    def test_cached_decompilation_avoids_redundant_ghidra_calls(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        sup._decompilation_cache[0x1000] = "cached code"

        result = sup._get_decompilation(0x1000)
        assert result == "cached code"
        client.get_decompilation.assert_not_called()

    def test_uncached_decompilation_calls_ghidra_and_caches(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        client.get_decompilation.return_value = "void foo(void) {}"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        result = sup._get_decompilation(0x1000)
        assert result == "void foo(void) {}"
        assert 0x1000 in sup._decompilation_cache
        client.get_decompilation.assert_called_once_with(0x1000)


class TestCleanup:
    def test_cleanup_retries_signatures(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        sup.binary_info = client.get_binary_info()
        sup.results[0x1000] = FunctionResult(
            address=0x1000,
            original_name="FUN_1000",
            name="init",
            confidence=80,
            signature="void init(void)",
            signature_applied=False,
        )

        sup._run_cleanup()

        client.set_function_signature.assert_called_once_with(0x1000, "void init(void)")
        assert sup.results[0x1000].signature_applied is True


class TestChunkedPipelineIntegration:
    def test_full_pipeline_uses_chunk_calls(self, tmp_path):
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(5)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        client.get_xrefs_from.return_value = []
        client.get_function_info.return_value = funcs[0]
        client.list_custom_types.return_value = []
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="",
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_count > 0
        named = [r for r in sup.results.values() if r.name and not r.skipped]
        assert len(named) == 5

    def test_chunk_prompt_includes_decompilations(self, tmp_path):
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void special_func(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="",
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        prompt = mock_llm.analyze_function_batch.call_args[0][0]
        assert "0x00001000" in prompt
        assert "special_func" in prompt
        assert "x86-64" in prompt

    def test_all_functions_in_single_chunk(self, tmp_path):
        """With < CHUNK_SIZE functions, everything goes in one LLM call."""
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(10)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="",
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_count == 1
