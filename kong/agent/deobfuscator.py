"""Deobfuscator agent — classifies obfuscation techniques and orchestrates
LLM-driven deobfuscation with symbolic tool access.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from kong.agent.models import FunctionResult
from kong.agent.prompts import DEOBFUSCATION_SYSTEM_PROMPT, OUTPUT_SCHEMA
from kong.ghidra.types import BinaryInfo, StringEntry
from kong.llm.tools import DEOBFUSCATION_TOOLS, ToolExecutor

if TYPE_CHECKING:
    from kong.agent.analyzer import AnalysisContext, LLMClient, LLMResponse
    from kong.agent.queue import WorkItem
    from kong.ghidra.client import GhidraClient

logger = logging.getLogger(__name__)

_PATTERNS_DIR = Path(__file__).resolve().parent.parent / "patterns"


class ObfuscationType(Enum):
    CONTROL_FLOW_FLATTENING = "cff"
    BOGUS_CONTROL_FLOW = "bogus_cf"
    INSTRUCTION_SUBSTITUTION = "instruction_sub"
    STRING_ENCRYPTION = "string_encrypt"
    VM_PROTECTION = "vmprotect"


_PATTERN_FILES: dict[ObfuscationType, str] = {
    ObfuscationType.CONTROL_FLOW_FLATTENING: "cff.md",
    ObfuscationType.BOGUS_CONTROL_FLOW: "bogus_cf.md",
    ObfuscationType.INSTRUCTION_SUBSTITUTION: "instruction_sub.md",
    ObfuscationType.STRING_ENCRYPTION: "string_encrypt.md",
    ObfuscationType.VM_PROTECTION: "vmprotect.md",
}


_CFF_LOOP_RE = re.compile(
    r"while\s*\(\s*(?:1|true)\s*\)\s*\{",
    re.IGNORECASE,
)
_SWITCH_RE = re.compile(r"switch\s*\(")
_CASE_RE = re.compile(r"case\s+(?:0x[0-9a-fA-F]+|\d+)\s*:")

_OPAQUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(\s*\w+\s*\*\s*\(\s*\w+\s*[+\-]\s*\d+\s*\)\s*\)\s*%\s*2"),
    re.compile(r"\w+\s*\^\s*\w+\s*\)\s*!=\s*0"),
    re.compile(r"\w+\s*&\s*~\s*\w+"),
    re.compile(r"\w+\s*\|\s*~\s*\w+"),
]

_INSTRUCTION_SUB_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(\s*~\s*\w+\s*&\s*\w+\s*\)\s*\|\s*\(\s*\w+\s*&\s*~\s*\w+\s*\)"),
    re.compile(r"\(\s*\w+\s*\^\s*\w+\s*\)\s*\+\s*2\s*\*\s*\(\s*\w+\s*&\s*\w+\s*\)"),
    re.compile(r"~\s*\w+\s*\+\s*1"),
]

_STRING_ENCRYPT_XOR_RE = re.compile(
    r"for\s*\([^)]*\)\s*\{[^}]*\^\s*=\s*(?:0x[0-9a-fA-F]+|\d+)\s*;",
    re.DOTALL,
)
_STACK_STRING_RE = re.compile(
    r"(?:\w+\[\d+\]\s*=\s*(?:0x[0-9a-fA-F]{2}|'\\.?')\s*;\s*){4,}"
)


def classify_obfuscation(decompiled: str) -> list[ObfuscationType]:
    """Heuristic classification of obfuscation techniques in decompiled code."""
    techniques: list[ObfuscationType] = []

    if _detect_cff(decompiled):
        techniques.append(ObfuscationType.CONTROL_FLOW_FLATTENING)

    if _detect_bogus_cf(decompiled):
        techniques.append(ObfuscationType.BOGUS_CONTROL_FLOW)

    if _detect_instruction_sub(decompiled):
        techniques.append(ObfuscationType.INSTRUCTION_SUBSTITUTION)

    if _detect_string_encrypt(decompiled):
        techniques.append(ObfuscationType.STRING_ENCRYPTION)

    if _detect_vmprotect(decompiled):
        techniques.append(ObfuscationType.VM_PROTECTION)

    return techniques


def _detect_cff(code: str) -> bool:
    """while(1)/switch with many hex-constant cases and a state variable."""
    if not _CFF_LOOP_RE.search(code):
        return False
    if not _SWITCH_RE.search(code):
        return False
    cases = _CASE_RE.findall(code)
    return len(cases) >= 4


def _detect_bogus_cf(code: str) -> bool:
    match_count = sum(1 for p in _OPAQUE_PATTERNS if p.search(code))
    return match_count >= 2


def _detect_instruction_sub(code: str) -> bool:
    match_count = sum(1 for p in _INSTRUCTION_SUB_PATTERNS if p.search(code))
    return match_count >= 2


def _detect_string_encrypt(code: str) -> bool:
    if _STRING_ENCRYPT_XOR_RE.search(code):
        return True
    return bool(_STACK_STRING_RE.search(code))


def _detect_vmprotect(code: str) -> bool:
    if not _CFF_LOOP_RE.search(code):
        return False
    if not _SWITCH_RE.search(code):
        return False
    cases = _CASE_RE.findall(code)
    if len(cases) < 15:
        return False
    lines = code.splitlines()
    return len(lines) > 200


def load_patterns(techniques: list[ObfuscationType]) -> str:
    """Load and concatenate pattern library files for the given techniques."""
    parts: list[str] = []
    for tech in techniques:
        filename = _PATTERN_FILES.get(tech)
        if filename is None:
            continue
        path = _PATTERNS_DIR / filename
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
        else:
            logger.warning("Pattern file not found: %s", path)
    return "\n\n---\n\n".join(parts)


class Deobfuscator:
    """Orchestrates LLM-driven deobfuscation with symbolic tool access.

    Usage::

        deobfuscator = Deobfuscator(ghidra_client, llm_client)
        techniques = classify_obfuscation(decompilation)
        if techniques:
            result = deobfuscator.deobfuscate(item, binary_info, ...)
    """

    def __init__(
        self,
        client: GhidraClient,
        llm_client: LLMClient,
    ) -> None:
        self.client = client
        self.llm = llm_client

    def deobfuscate(
        self,
        context: AnalysisContext,
        techniques: list[ObfuscationType],
    ) -> tuple[LLMResponse, int]:
        """Run LLM-driven deobfuscation with tool access.

        Returns (LLMResponse, tool_call_count).
        """
        pattern_context = load_patterns(techniques)
        prompt = self._build_prompt(context, techniques, pattern_context)
        executor = ToolExecutor(self.client)

        response = self.llm.analyze_with_tools(
            prompt=prompt,
            system=DEOBFUSCATION_SYSTEM_PROMPT,
            tools=DEOBFUSCATION_TOOLS,
            tool_executor=executor,
        )
        return response, executor.call_count

    def _build_prompt(
        self,
        context: AnalysisContext,
        techniques: list[ObfuscationType],
        pattern_context: str,
    ) -> str:
        parts: list[str] = []

        parts.append(
            f"Binary: {context.binary_info.arch} {context.binary_info.format} "
            f"({context.binary_info.compiler})"
        )
        parts.append("")

        tech_names = ", ".join(t.value for t in techniques)
        parts.append(f"## Obfuscation Detected: {tech_names}")
        parts.append("")

        parts.append(
            f"## Target Function: {context.function.name} "
            f"(0x{context.function.address:08x})"
        )
        parts.append(f"Size: {context.function.size} bytes")
        parts.append("")
        parts.append("### Decompilation")
        parts.append("```c")
        parts.append(context.decompilation)
        parts.append("```")

        if context.referenced_strings:
            parts.append("")
            parts.append("### Referenced Strings")
            for s in context.referenced_strings:
                parts.append(f'- "{s}"')

        if context.callee_snippets:
            parts.append("")
            parts.append("### Called Functions")
            for cs in context.callee_snippets:
                parts.append(f"#### {cs.name} (0x{cs.address:08x})")
                parts.append("```c")
                parts.append(cs.snippet)
                parts.append("```")

        if context.caller_snippets:
            parts.append("")
            parts.append("### Calling Functions")
            for cs in context.caller_snippets:
                parts.append(f"#### {cs.name} (0x{cs.address:08x})")
                parts.append("```c")
                parts.append(cs.snippet)
                parts.append("```")

        if context.known_functions:
            parts.append("")
            parts.append("### Already Identified Functions")
            for addr, name in sorted(context.known_functions.items()):
                parts.append(f"- 0x{addr:08x}: {name}")

        if pattern_context:
            parts.append("")
            parts.append("### Obfuscation Pattern Reference")
            parts.append(pattern_context)

        return "\n".join(parts)
