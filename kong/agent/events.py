"""Structured events emitted by the agent for TUI/CLI consumption."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(Enum):
    TRIAGE = "triage"
    ANALYSIS = "analysis"
    CLEANUP = "cleanup"
    SYNTHESIS = "synthesis"
    EXPORT = "export"


class EventType(Enum):
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"

    TRIAGE_FUNCTIONS_ENUMERATED = "triage_functions_enumerated"
    TRIAGE_SIGNATURES_MATCHED = "triage_signatures_matched"
    TRIAGE_QUEUE_BUILT = "triage_queue_built"

    FUNCTION_START = "function_start"
    FUNCTION_COMPLETE = "function_complete"
    FUNCTION_SKIPPED = "function_skipped"
    FUNCTION_ERROR = "function_error"

    LLM_CALL_START = "llm_call_start"
    LLM_CALL_COMPLETE = "llm_call_complete"

    CLEANUP_TYPES_UNIFIED = "cleanup_types_unified"
    CLEANUP_TYPE_CREATED = "cleanup_type_created"
    CLEANUP_SIGNATURES_RETRIED = "cleanup_signatures_retried"

    SYNTHESIS_GLOBALS_UNIFIED = "synthesis_globals_unified"
    SYNTHESIS_STRUCTS_SYNTHESIZED = "synthesis_structs_synthesized"
    SYNTHESIS_NAMES_REFINED = "synthesis_names_refined"

    DEOBFUSCATION_DETECTED = "deobfuscation_detected"
    DEOBFUSCATION_TOOL_CALL = "deobfuscation_tool_call"
    DEOBFUSCATION_COMPLETE = "deobfuscation_complete"

    EXPORT_FILE = "export_file"

    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"
    RUN_ERROR = "run_error"


@dataclass
class Event:
    type: EventType
    phase: Phase | None = None
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


EventCallback = Callable[[Event], None]
