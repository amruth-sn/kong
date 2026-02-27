"""Analyzer agent.

Takes a work item, gathers context, sends to the LLM, parses the
structured response, writes changes back to Ghidra, and returns
a FunctionResult.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from kong.agent.models import FunctionResult
from kong.agent.queue import WorkItem
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM interaction.

    TODO: add concrete implementation that uses Anthropic SDK.
    """

    def analyze_function(self, prompt: str) -> LLMResponse: ...


@dataclass
class LLMResponse:
    """Structured response from the LLM for a single function analysis."""
    name: str
    signature: str = ""
    confidence: int = 0
    classification: str = ""
    comments: str = ""
    reasoning: str = ""
    variables: list[VariableRename] = field(default_factory=list)
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class VariableRename:
    old_name: str
    new_name: str


@dataclass
class AnalysisContext:
    """All context assembled for analyzing a single function."""
    function: FunctionInfo
    decompilation: str
    binary_info: BinaryInfo
    caller_snippets: list[CallerSnippet] = field(default_factory=list)
    callee_snippets: list[CalleeSnippet] = field(default_factory=list)
    referenced_strings: list[str] = field(default_factory=list)
    known_functions: dict[int, str] = field(default_factory=dict)


@dataclass
class CallerSnippet:
    address: int
    name: str
    snippet: str


@dataclass
class CalleeSnippet:
    address: int
    name: str
    snippet: str


class Analyzer:
    """Analyzes a single function: gather context → LLM → parse → write back.

    Usage:
        analyzer = Analyzer(ghidra_client, llm_client)
        result = analyzer.analyze(work_item, binary_info, known_results, strings)
    """

    def __init__(
        self,
        client: GhidraClient,
        llm_client: LLMClient,
    ) -> None:
        self.client = client
        self.llm = llm_client

    def analyze(
        self,
        item: WorkItem,
        binary_info: BinaryInfo,
        known_results: dict[int, FunctionResult],
        strings: list[StringEntry],
    ) -> FunctionResult:
        """Full analysis pipeline for one function."""
        func = item.function

        context = self._build_context(item, binary_info, known_results, strings)
        prompt = self._build_prompt(context)

        response = self.llm.analyze_function(prompt)
        # TODO: validation?
        # TODO: error handling?
        # TODO: track token usage?

        self._write_back(func.address, response)

        return FunctionResult(
            address=func.address,
            original_name=func.name,
            name=response.name,
            signature=response.signature,
            confidence=response.confidence,
            classification=response.classification,
            comments=response.comments,
            reasoning=response.reasoning,
            llm_calls=1,
        )

    def _build_context(
        self,
        item: WorkItem,
        binary_info: BinaryInfo,
        known_results: dict[int, FunctionResult],
        strings: list[StringEntry],
    ) -> AnalysisContext:
        """Assemble all context the LLM needs to analyze this function."""
        func = item.function

        decompilation = self.client.get_decompilation(func.address)

        caller_snippets = self._get_caller_snippets(item.callers, known_results)
        callee_snippets = self._get_callee_snippets(item.callees, known_results)

        func_strings = self._get_referenced_strings(func.address, strings)

        known_map = {
            addr: r.name
            for addr, r in known_results.items()
            if r.name and not r.skipped and not r.error
        }

        return AnalysisContext(
            function=func,
            decompilation=decompilation,
            binary_info=binary_info,
            caller_snippets=caller_snippets,
            callee_snippets=callee_snippets,
            referenced_strings=func_strings,
            known_functions=known_map,
        )

    def _get_caller_snippets(
        self,
        caller_addrs: list[int],
        known_results: dict[int, FunctionResult],
    ) -> list[CallerSnippet]:
        """Get decompilation snippets for callers (first N lines each)."""
        snippets = []
        for addr in caller_addrs[:3]:
            name = self._resolve_name(addr, known_results)
            try:
                full = self.client.get_decompilation(addr)
                lines = full.split("\n")[:10]
                snippet = "\n".join(lines)
            except Exception:
                snippet = ""
            if snippet:
                snippets.append(CallerSnippet(address=addr, name=name, snippet=snippet))
        return snippets

    def _get_callee_snippets(
        self,
        callee_addrs: list[int],
        known_results: dict[int, FunctionResult],
    ) -> list[CalleeSnippet]:
        """Get decompilation snippets for callees (first N lines each)."""
        snippets = []
        for addr in callee_addrs[:5]:
            name = self._resolve_name(addr, known_results)
            try:
                full = self.client.get_decompilation(addr)
                lines = full.split("\n")[:10]
                snippet = "\n".join(lines)
            except Exception:
                snippet = ""
            if snippet:
                snippets.append(CalleeSnippet(address=addr, name=name, snippet=snippet))
        return snippets

    def _resolve_name(self, addr: int, known_results: dict[int, FunctionResult]) -> str:
        if addr in known_results and known_results[addr].name:
            return known_results[addr].name
        try:
            info = self.client.get_function_info(addr)
            return info.name
        except Exception:
            return f"FUN_{addr:08x}"

    def _get_referenced_strings(
        self,
        func_addr: int,
        strings: list[StringEntry],
    ) -> list[str]:
        """Find strings referenced by this function."""
        try:
            xrefs = self.client.get_xrefs_from(func_addr)
            ref_addrs = {x.to_addr for x in xrefs}
        except Exception:
            return []

        result = []
        for s in strings:
            if s.address in ref_addrs:
                result.append(s.value)
        return result

    def _build_prompt(self, context: AnalysisContext) -> str:
        """Build the LLM prompt from analysis context.

        The prompt structure is intentionally simple: present the
        decompilation and available context, ask for structured JSON output.

        TODO: refine prompt engineering from eval results.
        """
        parts = []

        parts.append(
            f"Binary: {context.binary_info.arch} {context.binary_info.format} "
            f"({context.binary_info.compiler})"
        )
        parts.append("")

        parts.append(f"## Target Function: {context.function.name} "
                     f"(0x{context.function.address:08x})")
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

        parts.append("")
        parts.append("### Instructions")
        parts.append(
            "Analyze the target function. Respond with a JSON object:\n"
            "```json\n"
            "{\n"
            '  "name": "descriptive_function_name",\n'
            '  "signature": "return_type name(param_type param_name, ...)",\n'
            '  "confidence": <0-100>,\n'
            '  "classification": "one of: crypto, networking, io, memory, '
            'string, math, init, cleanup, handler, parser, utility, unknown",\n'
            '  "comments": "Brief description of what the function does",\n'
            '  "reasoning": "Why you chose this name and classification",\n'
            '  "variables": [{"old_name": "var1", "new_name": "descriptive_name"}]\n'
            "}\n"
            "```"
        )

        return "\n".join(parts)

    def _write_back(self, addr: int, response: LLMResponse) -> None:
        """Write analysis results back to Ghidra."""
        if response.name:
            try:
                self.client.rename_function(addr, response.name)
            except Exception as e:
                logger.warning("Failed to rename 0x%08x to %s: %s", addr, response.name, e)

        if response.signature:
            try:
                self.client.set_function_signature(addr, response.signature)
            except Exception as e:
                logger.warning("Failed to set signature at 0x%08x: %s", addr, e)

        if response.comments:
            try:
                self.client.add_comment(addr, response.comments)
            except Exception as e:
                logger.warning("Failed to add comment at 0x%08x: %s", addr, e)

    @staticmethod
    def parse_llm_json(raw: str) -> LLMResponse:
        """Parse the LLM's JSON response into an LLMResponse.

        Handles common issues: markdown fences, trailing commas, etc.
        """
        text = raw.strip()

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return LLMResponse(
                name="",
                reasoning=f"Failed to parse LLM response as JSON: {raw[:200]}",
                raw=raw,
            )

        variables = [
            VariableRename(old_name=v["old_name"], new_name=v["new_name"])
            for v in data.get("variables", [])
            if "old_name" in v and "new_name" in v
        ]

        return LLMResponse(
            name=data.get("name", ""),
            signature=data.get("signature", ""),
            confidence=data.get("confidence", 0),
            classification=data.get("classification", ""),
            comments=data.get("comments", ""),
            reasoning=data.get("reasoning", ""),
            variables=variables,
            raw=raw,
        )
