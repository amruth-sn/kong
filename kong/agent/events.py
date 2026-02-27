"""Structured events emitted by the agent for TUI/CLI consumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

# We opt for an event-based state machine approach here.
class Phase(Enum):
    TRIAGE = "triage"
    ANALYSIS = "analysis"
    CLEANUP = "cleanup"
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
