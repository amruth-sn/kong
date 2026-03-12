"""Analyzer agent.

Takes a work item, gathers context, sends to the LLM, parses the
structured response, writes changes back to Ghidra, and returns
a FunctionResult.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from kong.agent.models import FunctionResult
from kong.agent.queue import WorkItem
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry, StructDefinition
from kong.normalizer.syntactic import normalize

if TYPE_CHECKING:
    from kong.agent.deobfuscator import Deobfuscator
    from kong.llm.tools import ToolExecutor, ToolSchema

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM interaction."""

    def analyze_function(self, prompt: str, *, model: str | None = None) -> LLMResponse: ...

    def analyze_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[ToolSchema],
        tool_executor: ToolExecutor,
        max_rounds: int = 10,
    ) -> LLMResponse: ...


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
    struct_proposals: list[StructProposal] = field(default_factory=list)
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class VariableRename:
    old_name: str
    new_name: str


@dataclass
class StructFieldProposal:
    name: str
    data_type: str
    offset: int
    size: int


@dataclass
class StructProposal:
    """A struct layout proposed by the LLM based on offset-based memory accesses."""
    name: str
    total_size: int
    fields: list[StructFieldProposal] = field(default_factory=list)
    used_by_param: str = ""
    source_function: int = 0


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
    known_types: list[StructDefinition] = field(default_factory=list)


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

    Usage::

        analyzer = Analyzer(ghidra_client, llm_client)
        result = analyzer.analyze(work_item, binary_info, known_results, strings)
    """

    def __init__(
        self,
        client: GhidraClient,
        llm_client: LLMClient,
        deobfuscator: Deobfuscator | None = None,
    ) -> None:
        self.client = client
        self.llm = llm_client
        self._deobfuscator = deobfuscator

    def analyze(
        self,
        item: WorkItem,
        binary_info: BinaryInfo,
        known_results: dict[int, FunctionResult],
        strings: list[StringEntry],
        known_types: list[StructDefinition] | None = None,
        model: str | None = None,
    ) -> FunctionResult:
        """Full analysis pipeline for one function."""
        # Inline import: deobfuscator imports from analyzer at module level (circular).
        from kong.agent.deobfuscator import classify_obfuscation

        func = item.function
        context = self._build_context(item, binary_info, known_results, strings, known_types)

        techniques = classify_obfuscation(context.decompilation) if self._deobfuscator else []

        if techniques and self._deobfuscator:
            response, tool_calls = self._deobfuscator.deobfuscate(context, techniques)
            sig_applied = self._write_back(func.address, response)
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
                signature_applied=sig_applied,
                struct_proposals=response.struct_proposals,
                obfuscation_techniques=[t.value for t in techniques],
                deobfuscation_tool_calls=tool_calls,
            )

        prompt = self._build_prompt(context)
        response = self.llm.analyze_function(prompt, model=model)
        sig_applied = self._write_back(func.address, response)

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
            signature_applied=sig_applied,
            struct_proposals=response.struct_proposals,
        )

    def _build_context(
        self,
        item: WorkItem,
        binary_info: BinaryInfo,
        known_results: dict[int, FunctionResult],
        strings: list[StringEntry],
        known_types: list[StructDefinition] | None = None,
    ) -> AnalysisContext:
        """Assemble all context the LLM needs to analyze this function."""
        func = item.function

        decompilation = normalize(self.client.get_decompilation(func.address))

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
            known_types=known_types or [],
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
                full = normalize(self.client.get_decompilation(addr))
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
                full = normalize(self.client.get_decompilation(addr))
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

        if context.known_types:
            parts.append("")
            parts.append("### Known Struct Types")
            parts.append(
                "These structs have been recovered from the binary. "
                "Use them in your signature if the function's parameters match."
            )
            for sd in context.known_types:
                parts.append(f"\n```c\nstruct {sd.name} {{ // {sd.size} bytes")
                for f in sd.fields:
                    parts.append(f"    {f.data_type} {f.name}; // offset 0x{f.offset:x}, {f.size} bytes")
                parts.append("};")
                parts.append("```")

        return "\n".join(parts)

    def build_batch_prompt(self, contexts: list[AnalysisContext]) -> str:
        """Build a single LLM prompt combining multiple function contexts for batch analysis."""
        if not contexts:
            return ""

        first = contexts[0]
        parts = []

        parts.append(
            f"Binary: {first.binary_info.arch} {first.binary_info.format} "
            f"({first.binary_info.compiler})"
        )
        parts.append("")
        parts.append(f"Analyze the following {len(contexts)} functions.")

        for i, context in enumerate(contexts, start=1):
            parts.append("")
            parts.append(
                f"=== Function {i}: {context.function.name} "
                f"(0x{context.function.address:08x}) ==="
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
                parts.append("### Known Callees")
                for cs in context.callee_snippets:
                    parts.append(f"- 0x{cs.address:08x}: {cs.name}")

        if first.known_functions:
            parts.append("")
            parts.append("### Already Identified Functions")
            for addr, name in sorted(first.known_functions.items()):
                parts.append(f"- 0x{addr:08x}: {name}")

        return "\n".join(parts)

    def _write_back(self, addr: int, response: LLMResponse) -> bool:
        """Write analysis results back to Ghidra.

        Returns True if the signature was successfully applied (or no
        signature was provided), False if it failed (typically because the
        signature references a type that doesn't exist yet).
        """
        if response.name:
            try:
                self.client.rename_function(addr, response.name)
            except Exception as e:
                logger.warning("Failed to rename 0x%08x to %s: %s", addr, response.name, e)

        signature_ok = True
        if response.signature:
            try:
                self.client.set_function_signature(addr, response.signature)
            except Exception as e:
                logger.debug("Signature deferred for 0x%08x (will retry in cleanup): %s", addr, e)
                signature_ok = False

        if response.comments:
            try:
                self.client.add_comment(addr, response.comments)
            except Exception as e:
                logger.warning("Failed to add comment at 0x%08x: %s", addr, e)

        return signature_ok

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

        struct_proposals = []
        for sp in data.get("struct_proposals", []):
            if not sp.get("name") or not sp.get("total_size"):
                continue
            try:
                total_size = int(sp["total_size"])
            except (ValueError, TypeError):
                continue
            fields = [
                StructFieldProposal(
                    name=f.get("name", f"field_{i}"),
                    data_type=f.get("data_type", "undefined"),
                    offset=int(f.get("offset", 0)),
                    size=int(f.get("size", 4)),
                )
                for i, f in enumerate(sp.get("fields", []))
            ]
            struct_proposals.append(StructProposal(
                name=sp["name"],
                total_size=total_size,
                fields=fields,
                used_by_param=sp.get("used_by_param", ""),
            ))

        return LLMResponse(
            name=data.get("name", ""),
            signature=data.get("signature", ""),
            confidence=data.get("confidence", 0),
            classification=data.get("classification", ""),
            comments=data.get("comments", ""),
            reasoning=data.get("reasoning", ""),
            variables=variables,
            struct_proposals=struct_proposals,
            raw=raw,
        )

    @staticmethod
    def parse_llm_json_batch(raw: str) -> list[LLMResponse]:
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
            logger.warning("Failed to parse batch LLM response: %s", raw[:200])
            return []

        if not isinstance(data, list):
            single = Analyzer.parse_llm_json(raw)
            return [single] if single.name else []

        results: list[LLMResponse] = []
        for entry in data:
            struct_proposals = []
            for sp in entry.get("struct_proposals", []):
                if not sp.get("name") or not sp.get("total_size"):
                    continue
                try:
                    total_size = int(sp["total_size"])
                except (ValueError, TypeError):
                    continue
                fields = [
                    StructFieldProposal(
                        name=f.get("name", f"field_{i}"),
                        data_type=f.get("data_type", "undefined"),
                        offset=int(f.get("offset", 0)),
                        size=int(f.get("size", 4)),
                    )
                    for i, f in enumerate(sp.get("fields", []))
                ]
                struct_proposals.append(StructProposal(
                    name=sp["name"],
                    total_size=total_size,
                    fields=fields,
                    used_by_param=sp.get("used_by_param", ""),
                ))

            results.append(LLMResponse(
                name=entry.get("name", ""),
                signature=entry.get("signature", ""),
                confidence=entry.get("confidence", 0),
                classification=entry.get("classification", ""),
                comments=entry.get("comments", ""),
                reasoning=entry.get("reasoning", ""),
                struct_proposals=struct_proposals,
                raw=raw,
            ))

        return results
