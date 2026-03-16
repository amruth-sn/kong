"""Semantic synthesis: single LLM call over complete analysis to unify globals, structs, and names."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kong.agent.analyzer import strip_markdown_fences
from kong.agent.models import FunctionResult

if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

logger = logging.getLogger(__name__)

SYNTHESIS_FUNCTION_CAP = 50

_DAT_PATTERN = re.compile(r"DAT_[0-9a-fA-F]+")


@dataclass
class SynthesisResult:
    globals: dict[str, str] = field(default_factory=dict)
    structs: list[dict[str, object]] = field(default_factory=list)
    name_refinements: dict[str, str] = field(default_factory=dict)


class SemanticSynthesizer:
    """Makes one LLM call over complete post-naming analysis to unify globals, synthesize structs,
    and refine names."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def synthesize(
        self,
        results: list[FunctionResult],
        decompilations: dict[int, str],
        model: str | None = None,
    ) -> SynthesisResult:
        prompt = self._build_synthesis_prompt(results, decompilations)
        response = self.llm.analyze_function(prompt, model=model)
        return self._parse_response(response.raw)

    def _build_synthesis_prompt(
        self,
        results: list[FunctionResult],
        decompilations: dict[int, str],
    ) -> str:
        parts: list[str] = []

        parts.append(
            "You are performing a cross-function synthesis pass over an analyzed binary. "
            "Your three tasks are:"
        )
        parts.append("")
        parts.append(
            "1. **Rename global variables**: For each DAT_XXXXXXXX address that appears "
            "in multiple functions, propose a meaningful name based on how it is used."
        )
        parts.append(
            "2. **Synthesize structs**: If multiple functions access fields at consistent "
            "offsets from the same base pointer or global, propose a struct definition."
        )
        parts.append(
            "3. **Refine function names**: If seeing all functions together reveals better "
            "names than the per-function pass produced, propose refinements."
        )
        parts.append("")
        parts.append("Respond with a single JSON object (no other text):")
        parts.append("```json")
        parts.append("{")
        parts.append('  "globals": {"DAT_XXXXXXXX": "meaningful_name", ...},')
        parts.append(
            '  "structs": [{"name": "StructName", "fields": [{"name": "field", "type": "int", "offset": 0}]}, ...],'
        )
        parts.append('  "name_refinements": {"0xADDRESS": "better_name", ...}')
        parts.append("}")
        parts.append("```")

        globals_map, xref_counts = self._extract_globals(decompilations)
        results_by_addr = {r.address: r for r in results}

        multi_use_globals = {
            name: addrs for name, addrs in globals_map.items() if len(addrs) >= 2
        }

        if multi_use_globals:
            parts.append("")
            parts.append("## Global Variables")
            parts.append("")
            for dat_name, addrs in sorted(multi_use_globals.items()):
                func_names = []
                for addr in sorted(addrs):
                    r = results_by_addr.get(addr)
                    label = r.name if r else f"FUN_{addr:08x}"
                    func_names.append(label)
                parts.append(f"- `{dat_name}` referenced by: {', '.join(func_names)}")

        parts.append("")
        parts.append("## Functions and Decompilations")
        parts.append("")

        eligible = [r for r in results if r.address in decompilations]
        if len(eligible) > SYNTHESIS_FUNCTION_CAP:
            eligible.sort(key=lambda r: xref_counts.get(r.address, 0), reverse=True)
            eligible = eligible[:SYNTHESIS_FUNCTION_CAP]

        for result in eligible:
            parts.append(f"### {result.name} (0x{result.address:08x})")
            parts.append(
                f"Classification: {result.classification}, Confidence: {result.confidence}"
            )
            parts.append("```c")
            parts.append(decompilations[result.address])
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _extract_globals(
        decompilations: dict[int, str],
    ) -> tuple[dict[str, set[int]], dict[int, int]]:
        """Extract globals and per-function xref counts in a single pass."""
        globals_map: dict[str, set[int]] = defaultdict(set)
        xref_counts: dict[int, int] = defaultdict(int)
        for addr, code in decompilations.items():
            for match in _DAT_PATTERN.finditer(code):
                globals_map[match.group(0)].add(addr)
                xref_counts[addr] += 1
        return dict(globals_map), dict(xref_counts)

    @staticmethod
    def _parse_response(raw: str) -> SynthesisResult:
        text = strip_markdown_fences(raw)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse synthesis response as JSON")
            return SynthesisResult()

        return SynthesisResult(
            globals=data.get("globals", {}),
            structs=data.get("structs", []),
            name_refinements=data.get("name_refinements", {}),
        )

    @staticmethod
    def _apply_globals(code: str, globals_map: dict[str, str]) -> str:
        for dat_name, meaningful_name in globals_map.items():
            code = code.replace(dat_name, meaningful_name)
        return code

    def apply_to_decompilations(
        self,
        result: SynthesisResult,
        decompilations: dict[int, str],
    ) -> dict[int, str]:
        return {
            addr: self._apply_globals(code, result.globals)
            for addr, code in decompilations.items()
        }
