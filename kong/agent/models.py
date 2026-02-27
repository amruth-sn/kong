"""Shared data models for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass


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
