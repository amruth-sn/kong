"""Supervisor agent — orchestrates the full analysis pipeline.

Drives: triage → analysis → cleanup → export.
Emits structured events for TUI/CLI consumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from kong.agent.events import Event, EventCallback, EventType, Phase
from kong.agent.queue import WorkQueue
from kong.config import KongConfig
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo


@dataclass
class FunctionResult:
    """Result of analyzing a single function."""
    address: int
    original_name: str
    name: str = ""
    signature: str = ""
    confidence: int = 0
    classification: str = ""
    comments: str = ""
    reasoning: str = ""
    error: str = ""
    llm_calls: int = 0
    skipped: bool = False
    skip_reason: str = ""


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
    ) -> None:
        self.client = client
        self.config = config
        self.queue = WorkQueue()
        self.stats = AnalysisStats()
        self.results: dict[int, FunctionResult] = {}  # Mapping of address to FunctionResult
        self.binary_info: BinaryInfo | None = None # initially None until set in _run_triage    
        self.functions: list[FunctionInfo] = [] # empty until set in _run_triage
        self.strings: list = [] # empty until set in _run_triage
        self._listeners: list[EventCallback] = [] # empty until on_event is called
        self._paused: bool = False # False until pause or resume is called

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

        self.binary_info = self.client.get_binary_info()
        self.functions = self.client.list_functions()
        self.strings = self.client.get_strings()

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

        callers_map: dict[int, list[int]] = {}
        callees_map: dict[int, list[int]] = {}
        for func in self.functions:
            addr = func.address
            callers_map[addr] = self.client.get_callers(addr)
            callees_map[addr] = self.client.get_callees(addr)

        # TODO: Match against signature DB.
        self._emit(Event(
            type=EventType.TRIAGE_SIGNATURES_MATCHED,
            phase=Phase.TRIAGE,
            message=f"Matched {self.stats.signature_matches} functions against signature DB.",
            data={"matched": self.stats.signature_matches},
        ))

        self.queue.build(self.functions, callers_map, callees_map)
        self.stats.total_functions = self.queue.total

        self._emit(Event(
            type=EventType.TRIAGE_QUEUE_BUILT,
            phase=Phase.TRIAGE,
            message=(
                f"Work queue built: {self.queue.total} functions to analyze "
                f"(bottom-up order)."
            ),
            data={"queue_size": self.queue.total},
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

    def _analyze_function(self, item) -> FunctionResult:
        """Analyze a single function. Delegates to the analyzer agent.

        TODO: Implement analyzer agent here. 
        This is the integration point where the analyzer agent will plug in. 
        """
        func = item.function

        if func.classification and func.classification.value == "trivial":
            return FunctionResult(
                address=func.address,
                original_name=func.name,
                skipped=True,
                skip_reason="trivial",
            )

        # TODO: analyzer agent something like
        # analyzer = Analyzer(self.client, self.llm_client, self.config)
        # return analyzer.analyze(item, context=self._build_context(item))
        return FunctionResult(
            address=func.address,
            original_name=func.name,
            name=func.name,  # unchanged for now
            confidence=0,
        )

    def _run_cleanup(self) -> None:
        """Re-analyze low-confidence functions with accumulated context."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.CLEANUP,
            message="Starting cleanup pass...",
        ))

        low_confidence = [
            r for r in self.results.values()
            if not r.skipped and not r.error and r.confidence < 50
        ]

        # TODO: Re-analyze with full context
        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.CLEANUP,
            message=f"Cleanup complete. {len(low_confidence)} functions re-examined.",
            data={"re_examined": len(low_confidence)},
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


    def _stats_dict(self) -> dict:
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
