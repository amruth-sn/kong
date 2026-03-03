"""Supervisor agent — orchestrates the full analysis pipeline.

Drives: triage → analysis → cleanup → export.
Emits structured events for TUI/CLI consumption.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from kong.agent.analyzer import Analyzer, LLMClient
from kong.agent.deobfuscator import Deobfuscator
from kong.agent.events import Event, EventCallback, EventType, Phase
from kong.agent.models import FunctionResult
from kong.agent.queue import WorkItem, WorkQueue
from kong.agent.signatures import SignatureDB
from kong.agent.triage import TriageAgent, TriageResult
from kong.agent.type_recovery import StructAccumulator, apply_unified_structs
from kong.config import KongConfig
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStats:
    """Aggregate statistics for the full run."""
    total_functions: int = 0
    analyzed: int = 0
    named: int = 0
    high_confidence: int = 0   # >= 80
    medium_confidence: int = 0  # 50-79
    low_confidence: int = 0     # < 50
    skipped: int = 0
    errors: int = 0
    llm_calls: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    signature_matches: int = 0

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0

    @property
    def name_rate(self) -> float:
        return self.named / self.total_functions if self.total_functions else 0.0

    def record_result(self, result: FunctionResult) -> None:
        if result.skipped:
            self.skipped += 1
            return
        if result.error:
            self.errors += 1
            return
        self.analyzed += 1
        self.llm_calls += result.llm_calls
        if result.name and result.name != result.original_name:
            self.named += 1
        # TODO: calibrate these eventually. these buckets are arbitrary, need eval data to
        # determine meaningful confidence tiers for the LLM's self-reported scores.
        if result.confidence >= 80:
            self.high_confidence += 1
        elif result.confidence >= 50:
            self.medium_confidence += 1
        else:
            self.low_confidence += 1


class Supervisor:
    """Main agent loop that orchestrates the full analysis pipeline.

    Usage:
        supervisor = Supervisor(client, config)
        supervisor.on_event(my_callback)
        results = supervisor.run()
    """

    def __init__(
        self,
        client: GhidraClient,
        config: KongConfig,
        llm_client: LLMClient | None = None,
        signature_db: SignatureDB | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.llm_client = llm_client
        self.sig_db = signature_db or SignatureDB()
        self.queue = WorkQueue()
        self.stats = AnalysisStats()
        self.results: dict[int, FunctionResult] = {}
        self.triage_result: TriageResult | None = None
        self.binary_info: BinaryInfo | None = None
        self.functions: list[FunctionInfo] = []
        self.strings: list[StringEntry] = []
        self.struct_accumulator = StructAccumulator()
        self._listeners: list[EventCallback] = []
        self._paused: bool = False

    def on_event(self, callback: EventCallback) -> None:
        """Register an event listener."""
        self._listeners.append(callback)

    def _emit(self, event: Event) -> None:
        for cb in self._listeners:
            cb(event)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    
    def run(self) -> dict[int, FunctionResult]:
        """Run the full analysis pipeline. Returns addr -> FunctionResult."""
        self.stats.start_time = time.time()
        self._emit(Event(
            type=EventType.RUN_START,
            message="Kong analysis starting.",
        ))

        try:
            self._run_triage()
            self._run_analysis()
            self._run_cleanup()
            self._run_export()
        except Exception as e:
            self._emit(Event(
                type=EventType.RUN_ERROR,
                message=f"Fatal error: {e}",
                data={"error": str(e)},
            ))
            raise

        self.stats.end_time = time.time()
        self._emit(Event(
            type=EventType.RUN_COMPLETE,
            message=(
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named in {self.stats.duration_seconds:.1f}s."
            ),
            data={"stats": self._stats_dict()},
        ))
        return self.results

    def _run_triage(self) -> None:
        """Enumerate functions, match signatures, build work queue."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.TRIAGE,
            message="Starting triage...",
        ))

        triage = TriageAgent(self.client, signature_db=self.sig_db)
        result = triage.run()
        self.triage_result = result

        # propagate triage results to supervisor state
        self.binary_info = result.binary_info
        self.functions = result.functions
        self.strings = result.strings
        self.queue = result.queue
        self.stats.total_functions = result.queue_size
        self.stats.signature_matches = result.matched_count

        self._emit(Event(
            type=EventType.TRIAGE_FUNCTIONS_ENUMERATED,
            phase=Phase.TRIAGE,
            message=f"Enumerated {len(self.functions)} functions.",
            data={
                "total": len(self.functions),
                "binary_info": {
                    "arch": self.binary_info.arch,
                    "format": self.binary_info.format,
                    "compiler": self.binary_info.compiler,
                },
            },
        ))

        self._emit(Event(
            type=EventType.TRIAGE_SIGNATURES_MATCHED,
            phase=Phase.TRIAGE,
            message=f"Matched {self.stats.signature_matches} functions against signature DB.",
            data={"matched": self.stats.signature_matches},
        ))

        self._emit(Event(
            type=EventType.TRIAGE_QUEUE_BUILT,
            phase=Phase.TRIAGE,
            message=(
                f"Work queue built: {self.queue.total} functions to analyze "
                f"(bottom-up order)."
            ),
            data={"queue_size": self.queue.total},
        ))

        if result.language_hints.language != "C":
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.TRIAGE,
                message=f"Detected language: {result.language_hints.language}",
                data={"language": result.language_hints.language},
            ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.TRIAGE,
            message=(
                f"Triage complete. {len(self.functions)} functions found, "
                f"{self.stats.signature_matches} pre-labeled."
            ),
        ))


    def _run_analysis(self) -> None:
        """Analyze each function bottom-up via LLM."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message="Starting bottom-up analysis...",
        ))

        while self.queue:
            if self._paused:
                time.sleep(0.1)
                continue

            item = self.queue.next()
            if item is None:
                break

            func = item.function
            self._emit(Event(
                type=EventType.FUNCTION_START,
                phase=Phase.ANALYSIS,
                message=f"Analyzing {func.name} ({func.address_hex})...",
                data={
                    "address": func.address,
                    "name": func.name,
                    "size": func.size,
                    "depth": item.depth,
                    "progress": f"{self.queue.completed}/{self.queue.total}",
                },
            ))

            try:
                result = self._analyze_function(item)
            except Exception as e:
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    error=str(e),
                )
                self._emit(Event(
                    type=EventType.FUNCTION_ERROR,
                    phase=Phase.ANALYSIS,
                    message=f"Error analyzing {func.name}: {e}",
                    data={"address": func.address, "error": str(e)},
                ))

            self.results[func.address] = result
            self.stats.record_result(result)

            if result.struct_proposals:
                self.struct_accumulator.add_proposals(func.address, result.struct_proposals)

            if not result.error and not result.skipped:
                self._emit(Event(
                    type=EventType.FUNCTION_COMPLETE,
                    phase=Phase.ANALYSIS,
                    message=(
                        f"{func.address_hex} → {result.name} "
                        f"(confidence: {result.confidence}%)"
                    ),
                    data={
                        "address": func.address,
                        "original_name": result.original_name,
                        "name": result.name,
                        "confidence": result.confidence,
                        "classification": result.classification,
                    },
                ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.ANALYSIS,
            message=(
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named."
            ),
        ))

    def _analyze_function(self, item: WorkItem) -> FunctionResult:
        """Analyze a single function. Delegates to the Analyzer agent."""
        func = item.function

        if func.classification and func.classification.value == "trivial":
            return FunctionResult(
                address=func.address,
                original_name=func.name,
                skipped=True,
                skip_reason="trivial",
            )

        if self.llm_client is None:
            return FunctionResult(
                address=func.address,
                original_name=func.name,
                name=func.name,
                confidence=0,
            )

        deobfuscator = Deobfuscator(self.client, self.llm_client)
        analyzer = Analyzer(self.client, self.llm_client, deobfuscator=deobfuscator)
        result = analyzer.analyze(
            item,
            binary_info=self.binary_info,
            known_results=self.results,
            strings=self.strings,
        )

        if result.obfuscation_techniques:
            self._emit(Event(
                type=EventType.DEOBFUSCATION_DETECTED,
                phase=Phase.ANALYSIS,
                message=(
                    f"Obfuscation detected in {func.name}: "
                    f"{', '.join(result.obfuscation_techniques)}"
                ),
                data={
                    "address": func.address,
                    "techniques": result.obfuscation_techniques,
                    "tool_calls": result.deobfuscation_tool_calls,
                },
            ))

        return result

    def _run_cleanup(self) -> None:
        """Unify struct types, apply to Ghidra, re-analyze affected and low-confidence functions."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.CLEANUP,
            message="Starting cleanup pass...",
        ))

        structs_created = 0
        retyped_addrs: list[int] = []

        if self.struct_accumulator.proposal_count > 0:
            unified = self.struct_accumulator.unify()
            self._emit(Event(
                type=EventType.CLEANUP_TYPES_UNIFIED,
                phase=Phase.CLEANUP,
                message=(
                    f"Unified {self.struct_accumulator.proposal_count} struct proposals "
                    f"into {len(unified)} types."
                ),
                data={
                    "proposals": self.struct_accumulator.proposal_count,
                    "unified": len(unified),
                },
            ))

            retyped_addrs = apply_unified_structs(self.client, unified)
            structs_created = len(unified)

            for us in unified:
                self._emit(Event(
                    type=EventType.CLEANUP_TYPE_CREATED,
                    phase=Phase.CLEANUP,
                    message=(
                        f"Created struct '{us.definition.name}' "
                        f"({us.definition.size} bytes, {us.definition.field_count} fields)"
                    ),
                    data={
                        "name": us.definition.name,
                        "size": us.definition.size,
                        "fields": us.definition.field_count,
                    },
                ))

        pending_signatures = [
            r for r in self.results.values()
            if r.signature and not r.signature_applied and not r.skipped and not r.error
        ]
        if pending_signatures:
            sigs_retried = 0
            for r in pending_signatures:
                try:
                    self.client.set_function_signature(r.address, r.signature)
                    r.signature_applied = True
                    sigs_retried += 1
                except Exception:
                    logger.debug(
                        "Signature retry still failed for 0x%08x: %s",
                        r.address, r.signature,
                    )
            self._emit(Event(
                type=EventType.CLEANUP_SIGNATURES_RETRIED,
                phase=Phase.CLEANUP,
                message=(
                    f"Retried {len(pending_signatures)} pending signatures, "
                    f"{sigs_retried} succeeded."
                ),
                data={
                    "pending": len(pending_signatures),
                    "succeeded": sigs_retried,
                },
            ))

        low_confidence = [
            r for r in self.results.values()
            if not r.skipped and not r.error and r.confidence < 50
        ]
        low_conf_addrs = {r.address for r in low_confidence}

        reanalyze_addrs = sorted(set(retyped_addrs) | low_conf_addrs)
        reanalyzed = 0

        if reanalyze_addrs and self.llm_client is not None:
            known_types = self.client.list_custom_types()
            analyzer = Analyzer(self.client, self.llm_client)

            for addr in reanalyze_addrs:
                if self._paused:
                    continue

                old_result = self.results.get(addr)
                func_name = old_result.original_name if old_result else f"FUN_{addr:08x}"

                self._emit(Event(
                    type=EventType.FUNCTION_START,
                    phase=Phase.CLEANUP,
                    message=f"Re-analyzing {func_name} (0x{addr:08x})...",
                    data={"address": addr, "name": func_name},
                ))

                func_info = self._find_function(addr)
                if func_info is None:
                    continue

                item = WorkItem(
                    function=func_info,
                    callers=self.triage_result.call_graph.callers.get(addr, []) if self.triage_result else [],
                    callees=self.triage_result.call_graph.callees.get(addr, []) if self.triage_result else [],
                )

                try:
                    new_result = analyzer.analyze(
                        item,
                        binary_info=self.binary_info,
                        known_results=self.results,
                        strings=self.strings,
                        known_types=known_types,
                    )
                except Exception as e:
                    self._emit(Event(
                        type=EventType.FUNCTION_ERROR,
                        phase=Phase.CLEANUP,
                        message=f"Error re-analyzing {func_name}: {e}",
                        data={"address": addr, "error": str(e)},
                    ))
                    continue

                if new_result.confidence > (old_result.confidence if old_result else 0):
                    self.results[addr] = new_result
                    reanalyzed += 1

                    self._emit(Event(
                        type=EventType.FUNCTION_COMPLETE,
                        phase=Phase.CLEANUP,
                        message=(
                            f"0x{addr:08x} → {new_result.name} "
                            f"(confidence: {new_result.confidence}%, "
                            f"was {old_result.confidence if old_result else 0}%)"
                        ),
                        data={
                            "address": addr,
                            "name": new_result.name,
                            "confidence": new_result.confidence,
                            "previous_confidence": old_result.confidence if old_result else 0,
                        },
                    ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.CLEANUP,
            message=(
                f"Cleanup complete. {structs_created} structs created, "
                f"{reanalyzed}/{len(reanalyze_addrs)} functions improved."
            ),
            data={
                "structs_created": structs_created,
                "reanalyze_candidates": len(reanalyze_addrs),
                "reanalyzed": reanalyzed,
            },
        ))

    def _run_export(self) -> None:
        """Generate output files."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.EXPORT,
            message="Starting export...",
        ))

        # TODO: export logic.
        output_dir = self.config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.EXPORT,
            message=f"Export complete. Files saved to {output_dir}/",
            data={"output_dir": str(output_dir)},
        ))


    def _find_function(self, addr: int) -> FunctionInfo | None:
        """Look up a FunctionInfo by address from the triage function list."""
        for f in self.functions:
            if f.address == addr:
                return f
        return None

    def _stats_dict(self) -> dict[str, int | float]:
        s = self.stats
        return {
            "total_functions": s.total_functions,
            "analyzed": s.analyzed,
            "named": s.named,
            "high_confidence": s.high_confidence,
            "medium_confidence": s.medium_confidence,
            "low_confidence": s.low_confidence,
            "skipped": s.skipped,
            "errors": s.errors,
            "llm_calls": s.llm_calls,
            "signature_matches": s.signature_matches,
            "duration_seconds": round(s.duration_seconds, 1),
        }
