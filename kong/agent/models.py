"""Shared data models for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kong.agent.analyzer import StructProposal


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
