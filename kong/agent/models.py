"""Shared data models for the agent pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kong.agent.analyzer import StructProposal


@dataclass
class AnalysisStats:
    """Aggregate statistics for the full run."""
    total_functions: int = 0
    analyzed: int = 0
    renamed: int = 0
    confirmed: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    skipped: int = 0
    errors: int = 0
    llm_calls: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    signature_matches: int = 0

    @property
    def named(self) -> int:
        return self.renamed + self.confirmed

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
        if result.name:
            if result.name != result.original_name:
                self.renamed += 1
            else:
                self.confirmed += 1
        # TODO: calibrate these eventually. these buckets are arbitrary, need eval data to
        # determine meaningful confidence tiers for the LLM's self-reported scores.
        if result.confidence >= 80:
            self.high_confidence += 1
        elif result.confidence >= 50:
            self.medium_confidence += 1
        else:
            self.low_confidence += 1


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
    signature_applied: bool = False
    struct_proposals: list[StructProposal] = field(default_factory=list)
    obfuscation_techniques: list[str] = field(default_factory=list)
    deobfuscation_tool_calls: int = 0
