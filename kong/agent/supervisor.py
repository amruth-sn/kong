"""Supervisor agent — orchestrates the full analysis pipeline.

Drives: triage → analysis → cleanup → synthesis → export.
Emits structured events for TUI/CLI consumption.
"""

from __future__ import annotations

import logging
import threading
import time

from kong.agent.analyzer import Analyzer, LLMClient
from kong.agent.deobfuscator import Deobfuscator, classify_obfuscation
from kong.agent.events import Event, EventCallback, EventType, Phase
from kong.agent.models import AnalysisStats, FunctionResult
from kong.agent.queue import WorkItem, WorkQueue
from kong.agent.signatures import SignatureDB
from kong.agent.triage import TriageAgent, TriageResult
from kong.agent.type_recovery import StructAccumulator, apply_unified_structs
from kong.config import KongConfig, LLMProvider
from kong.export.source import ExportData, export_source
from kong.export.structured import export_json
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry
from kong.llm.limits import ModelLimits, get_model_limits
from kong.llm.usage import TokenUsage
from kong.normalizer.syntactic import normalize
from kong.synthesis.semantic import SemanticSynthesizer, SynthesisResult

logger = logging.getLogger(__name__)


def _clean_api_error(exc: Exception) -> str:
    """Extract a concise error message from LLM API exceptions.

    The OpenAI SDK embeds raw HTTP response bodies (often HTML from proxies)
    in exception messages. This extracts just the useful parts.
    """
    import openai

    if isinstance(exc, openai.APIStatusError):
        return f"HTTP {exc.status_code}: {exc.response.reason_phrase}"
    return str(exc)[:200]


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
        self._synthesis_result: SynthesisResult | None = None
        self._listeners: list[EventCallback] = []
        self._paused: bool = False
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._decompilation_cache: dict[int, str] = {}
        self._functions_by_addr: dict[int, FunctionInfo] = {}

    def _get_decompilation(self, addr: int) -> str:
        if addr not in self._decompilation_cache:
            self._decompilation_cache[addr] = self.client.get_decompilation(addr)
        return self._decompilation_cache[addr]

    def on_event(self, callback: EventCallback) -> None:
        """Register an event listener."""
        self._listeners.append(callback)

    def _emit(self, event: Event) -> None:
        for cb in self._listeners:
            cb(event)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        self._resume_event.clear()

    def resume(self) -> None:
        self._paused = False
        self._resume_event.set()

    def _wait_if_paused(self) -> None:
        """Block until resumed. No-op if not paused."""
        self._resume_event.wait()

    def export(self) -> None:
        """Trigger export manually (e.g. from TUI keybind)."""
        self._run_export()

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
            self._run_synthesis()
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

        self.binary_info = result.binary_info
        self.functions = result.functions
        self._functions_by_addr = {f.address: f for f in self.functions}
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
        """Analyze functions in large chunks via sequential LLM calls."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message="Starting analysis...",
        ))

        all_items = self.queue.all_items()
        chunk_items: list[tuple[WorkItem, str]] = []
        sequential_items: list[WorkItem] = []
        completed_count = 0

        for item in all_items:
            func = item.function

            if func.classification and func.classification.value == "trivial":
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    skipped=True,
                    skip_reason="trivial",
                )
                self.results[func.address] = result
                self.stats.record_result(result)
                completed_count += 1
                continue

            if self.llm_client is None:
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    name=func.name,
                    confidence=0,
                )
                self.results[func.address] = result
                self.stats.record_result(result)
                completed_count += 1
                continue

            decompilation = self._get_decompilation(func.address)
            techniques = classify_obfuscation(decompilation)

            if techniques:
                sequential_items.append(item)
                continue

            chunk_items.append((item, normalize(decompilation)))

        if chunk_items:
            self._analyze_chunks(chunk_items, completed_count)
            completed_count += len(chunk_items)

        for item in sequential_items:
            self._wait_if_paused()
            completed_count += 1
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
                    "progress": f"{completed_count}/{self.queue.total}",
                },
            ))
            try:
                result = self._analyze_function_sequential(item)
            except Exception as e:
                err_msg = _clean_api_error(e)
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    error=err_msg,
                )
                self._emit(Event(
                    type=EventType.FUNCTION_ERROR,
                    phase=Phase.ANALYSIS,
                    message=f"Error analyzing {func.name}: {err_msg}",
                    data={"address": func.address, "error": err_msg},
                ))
            self._record_analysis_result(func, result)

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.ANALYSIS,
            message=(
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named."
            ),
        ))

    def _get_effective_limits(self) -> ModelLimits:
        cfg = self.config.llm
        base = get_model_limits(getattr(self.llm_client, "model", "") or "")
        if cfg.provider is not LLMProvider.CUSTOM:
            return base
        return ModelLimits(
            max_prompt_chars=cfg.max_prompt_chars if cfg.max_prompt_chars is not None else base.max_prompt_chars,
            max_chunk_functions=cfg.max_chunk_functions if cfg.max_chunk_functions is not None else base.max_chunk_functions,
            max_output_tokens=cfg.max_output_tokens if cfg.max_output_tokens is not None else base.max_output_tokens,
        )

    def _split_into_chunks(
        self,
        items: list[tuple[WorkItem, str]],
    ) -> list[list[tuple[WorkItem, str]]]:
        """Split items into chunks by building the actual prompt and measuring size."""
        assert self.binary_info is not None

        limits = self._get_effective_limits()

        overhead = self._build_chunk_prompt([])
        overhead_len = len(overhead)
        available = limits.max_prompt_chars - overhead_len

        chunks: list[list[tuple[WorkItem, str]]] = []
        current_chunk: list[tuple[WorkItem, str]] = []
        current_chars = 0

        for item, decomp in items:
            func = item.function
            entry = (
                f"### 0x{func.address:08x}: {func.name} ({func.size} bytes)\n"
                f"```c\n{decomp}\n```\n\n"
            )
            entry_len = len(entry)

            chunk_full = (
                current_chars + entry_len > available
                or len(current_chunk) >= limits.max_chunk_functions
            )
            if current_chunk and chunk_full:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0
            current_chunk.append((item, decomp))
            current_chars += entry_len

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _analyze_chunks(
        self,
        items: list[tuple[WorkItem, str]],
        completed_base: int,
    ) -> None:
        """Analyze functions in large chunks, streaming results per chunk."""
        assert self.llm_client is not None
        assert self.binary_info is not None

        items.sort(key=lambda x: len(x[1]))

        analyzer = Analyzer(self.client, self.llm_client)
        chunks = self._split_into_chunks(items)
        total_chunks = len(chunks)
        processed = 0

        for chunk_num, chunk in enumerate(chunks, start=1):

            for idx, (item, _) in enumerate(chunk):
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
                        "progress": f"{completed_base + processed + idx + 1}/{self.queue.total}",
                    },
                ))

            self._wait_if_paused()

            logger.info(
                "Sending chunk %d/%d (%d functions) to LLM...",
                chunk_num, total_chunks, len(chunk),
            )

            prompt = self._build_chunk_prompt(chunk)
            logger.info(
                "Chunk %d/%d prompt: %d chars (%d functions).",
                chunk_num, total_chunks, len(prompt), len(chunk),
            )

            try:
                responses = self.llm_client.analyze_function_batch(
                    prompt, model=self.llm_client.model,
                )
            except Exception as e:
                err_msg = _clean_api_error(e)
                logger.warning("Chunk %d/%d failed: %s", chunk_num, total_chunks, err_msg)
                for item, _ in chunk:
                    func = item.function
                    result = FunctionResult(
                        address=func.address,
                        original_name=func.name,
                        error=f"Chunk call failed: {err_msg}",
                    )
                    self._emit(Event(
                        type=EventType.FUNCTION_ERROR,
                        phase=Phase.ANALYSIS,
                        message=f"Error analyzing {func.name}: {err_msg}",
                        data={"address": func.address, "error": err_msg},
                    ))
                    self._record_analysis_result(func, result)
                continue

            by_addr = {r.address: r for r in responses if r.address}

            logger.info(
                "Chunk %d/%d: LLM returned %d responses, %d with valid addresses "
                "(chunk has %d functions).",
                chunk_num, total_chunks, len(responses), len(by_addr), len(chunk),
            )

            matched = 0
            for item, _ in chunk:
                func = item.function
                response = by_addr.get(func.address)

                if response and response.name:
                    sig_applied = analyzer._write_back(func.address, response)
                    result = FunctionResult(
                        address=func.address,
                        original_name=func.name,
                        name=response.name,
                        signature=response.signature,
                        confidence=response.confidence,
                        classification=response.classification,
                        comments=response.comments,
                        reasoning=response.reasoning,
                        llm_calls=1,
                        signature_applied=sig_applied,
                        struct_proposals=response.struct_proposals,
                    )
                    matched += 1
                else:
                    reason = "No matching response from LLM"
                    if response:
                        reason = response.reasoning or "Empty name in response"
                    result = FunctionResult(
                        address=func.address,
                        original_name=func.name,
                        error=reason,
                    )
                    self._emit(Event(
                        type=EventType.FUNCTION_ERROR,
                        phase=Phase.ANALYSIS,
                        message=f"No analysis for {func.name}",
                        data={"address": func.address, "error": reason},
                    ))

                self._record_analysis_result(func, result)

            logger.info(
                "Chunk %d/%d complete: %d/%d functions matched.",
                chunk_num, total_chunks, matched, len(chunk),
            )
            processed += len(chunk)

    def _build_chunk_prompt(self, items: list[tuple[WorkItem, str]]) -> str:
        """Build a prompt with all decompilations for this chunk."""
        assert self.binary_info is not None

        parts = [
            f"Binary: {self.binary_info.arch} {self.binary_info.format} "
            f"({self.binary_info.compiler})",
            "",
            f"Analyze the following {len(items)} functions.",
            "",
        ]

        known = {
            addr: r.name
            for addr, r in self.results.items()
            if r.name and not r.skipped and not r.error
        }
        if known:
            parts.append("### Already Identified Functions")
            for addr, name in sorted(known.items()):
                parts.append(f"- 0x{addr:08x}: {name}")
            parts.append("")

        for item, decompilation in items:
            func = item.function
            parts.append(f"### 0x{func.address:08x}: {func.name} ({func.size} bytes)")
            parts.append("```c")
            parts.append(decompilation)
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    def _analyze_function_sequential(self, item: WorkItem) -> FunctionResult:
        """Analyze a single function sequentially (used for obfuscated functions)."""
        assert self.llm_client is not None
        func = item.function
        deobfuscator = Deobfuscator(self.client, self.llm_client)
        analyzer = Analyzer(self.client, self.llm_client, deobfuscator=deobfuscator)
        result = analyzer.analyze(
            item,
            binary_info=self.binary_info,
            known_results=self.results,
            strings=self.strings,
            model=self.llm_client.model,
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

    def _record_analysis_result(self, func: FunctionInfo, result: FunctionResult) -> None:
        """Record a function result into the results dict and stats."""
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

    def _run_cleanup(self) -> None:
        """Unify struct types and retry failed signatures."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.CLEANUP,
            message="Starting cleanup pass...",
        ))

        structs_created = 0

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

            apply_unified_structs(self.client, unified)
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

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.CLEANUP,
            message=f"Cleanup complete. {structs_created} structs created.",
            data={"structs_created": structs_created},
        ))

    def _run_synthesis(self) -> None:
        """Cross-function synthesis: unify globals, synthesize structs, refine names."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.SYNTHESIS,
            message="Starting synthesis...",
        ))

        if self.llm_client is None:
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message="Synthesis skipped (no LLM client).",
            ))
            return

        decompilations: dict[int, str] = {}
        for addr, result in self.results.items():
            if result.skipped or result.error:
                continue
            decomp = self._get_decompilation(addr)
            if decomp:
                decompilations[addr] = normalize(decomp)

        if not decompilations:
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message="Synthesis skipped (no decompilations).",
            ))
            return

        synthesizer = SemanticSynthesizer(self.llm_client)
        try:
            synthesis_result = synthesizer.synthesize(
                list(self.results.values()), decompilations, model=self.llm_client.model,
            )
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message=f"Synthesis failed: {e}",
                data={"error": str(e)},
            ))
            return

        if synthesis_result.globals:
            self._emit(Event(
                type=EventType.SYNTHESIS_GLOBALS_UNIFIED,
                phase=Phase.SYNTHESIS,
                message=f"Unified {len(synthesis_result.globals)} global variables.",
                data={"count": len(synthesis_result.globals), "globals": synthesis_result.globals},
            ))

        if synthesis_result.structs:
            self._emit(Event(
                type=EventType.SYNTHESIS_STRUCTS_SYNTHESIZED,
                phase=Phase.SYNTHESIS,
                message=f"Synthesized {len(synthesis_result.structs)} structs.",
                data={"count": len(synthesis_result.structs)},
            ))

        if synthesis_result.name_refinements:
            for addr_str, new_name in synthesis_result.name_refinements.items():
                addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
                if addr in self.results:
                    self.results[addr].name = new_name
                    try:
                        self.client.rename_function(addr, new_name)
                    except Exception as e:
                        logger.warning(
                            "Failed to rename 0x%08x to %s during synthesis: %s",
                            addr, new_name, e,
                        )
            self._emit(Event(
                type=EventType.SYNTHESIS_NAMES_REFINED,
                phase=Phase.SYNTHESIS,
                message=f"Refined {len(synthesis_result.name_refinements)} function names.",
                data={"count": len(synthesis_result.name_refinements)},
            ))

        self._synthesis_result = synthesis_result

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.SYNTHESIS,
            message="Synthesis complete.",
        ))

    def _run_export(self) -> None:
        """Generate output files."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.EXPORT,
            message="Starting export...",
        ))

        output_dir = self.config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        exportable_addrs = [
            addr for addr, r in self.results.items()
            if not r.skipped and not r.error
        ]
        decompilations: dict[int, str] = {}
        for addr in exportable_addrs:
            decomp = self._get_decompilation(addr)
            if decomp:
                decompilations[addr] = normalize(decomp)

        if self._synthesis_result:
            synthesizer = SemanticSynthesizer(self.llm_client)
            decompilations = synthesizer.apply_to_decompilations(
                self._synthesis_result, decompilations,
            )

        raw_usage = getattr(self.llm_client, "usage", None) if self.llm_client else None
        token_usage = raw_usage if isinstance(raw_usage, TokenUsage) else TokenUsage()

        export_data = ExportData(
            binary_info=self.binary_info or BinaryInfo(
                arch="unknown", format="unknown", endianness="unknown",
                word_size=0, compiler="unknown", name="unknown",
            ),
            stats=self.stats,
            results=self.results,
            decompilations=decompilations,
            token_usage=token_usage,
            duration_seconds=self.stats.duration_seconds,
            provider=self.config.llm.provider,
        )

        formats = self.config.output.formats

        if "source" in formats:
            path = export_source(export_data, output_dir / "decompiled.c")
            self._emit(Event(
                type=EventType.EXPORT_FILE,
                phase=Phase.EXPORT,
                message=f"Exported {path}",
                data={"path": str(path), "format": "source"},
            ))

        if "json" in formats:
            path = export_json(export_data, output_dir / "analysis.json")
            self._emit(Event(
                type=EventType.EXPORT_FILE,
                phase=Phase.EXPORT,
                message=f"Exported {path}",
                data={"path": str(path), "format": "json"},
            ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.EXPORT,
            message=f"Export complete. Files saved to {output_dir}/",
            data={"output_dir": str(output_dir)},
        ))

    def _find_function(self, addr: int) -> FunctionInfo | None:
        return self._functions_by_addr.get(addr)

    def _stats_dict(self) -> dict[str, int | float]:
        s = self.stats
        return {
            "total_functions": s.total_functions,
            "analyzed": s.analyzed,
            "named": s.named,
            "renamed": s.renamed,
            "confirmed": s.confirmed,
            "high_confidence": s.high_confidence,
            "medium_confidence": s.medium_confidence,
            "low_confidence": s.low_confidence,
            "skipped": s.skipped,
            "errors": s.errors,
            "llm_calls": s.llm_calls,
            "signature_matches": s.signature_matches,
            "duration_seconds": round(s.duration_seconds, 1),
        }
