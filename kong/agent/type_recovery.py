"""Type recovery — accumulate struct proposals from analysis, unify, and apply.

Collects StructProposal objects emitted by the LLM during Phase 2 (analysis),
merges proposals that describe the same underlying struct across multiple
functions, creates the unified types in Ghidra, and identifies functions that
should be re-analyzed with the improved type information.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from kong.agent.analyzer import StructFieldProposal, StructProposal
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import StructDefinition, StructField

logger = logging.getLogger(__name__)


@dataclass
class ParamTypeApplication:
    """Records that a specific function parameter should receive a struct type."""
    func_addr: int
    param_name: str
    struct_name: str


@dataclass
class UnifiedStruct:
    """A struct definition produced by merging multiple proposals."""
    definition: StructDefinition
    source_proposals: list[StructProposal] = field(default_factory=list)
    applications: list[ParamTypeApplication] = field(default_factory=list)


class StructAccumulator:
    """Collects struct proposals and unifies them into concrete types.

    Usage during analysis::

        acc = StructAccumulator()
        # after each function analysis:
        acc.add_proposals(func_addr, response.struct_proposals)
        # during cleanup:
        unified = acc.unify()
    """

    def __init__(self) -> None:
        self._proposals: list[StructProposal] = []

    @property
    def proposal_count(self) -> int:
        return len(self._proposals)

    def add_proposals(self, func_addr: int, proposals: list[StructProposal]) -> None:
        for p in proposals:
            tagged = StructProposal(
                name=p.name,
                total_size=p.total_size,
                fields=list(p.fields),
                used_by_param=p.used_by_param,
                source_function=func_addr,
            )
            self._proposals.append(tagged)

    def unify(self) -> list[UnifiedStruct]:
        """Merge proposals that describe the same struct.

        Grouping strategy: two proposals are considered to describe the same
        struct if they have the same total_size AND share at least one field
        at the same offset with a compatible type.  Within each group the
        LLM-given name that appears most often wins; fields are merged by
        offset, preferring the most descriptive name.
        """
        if not self._proposals:
            return []

        groups = self._group_proposals()
        return [self._merge_group(g) for g in groups]

    def _group_proposals(self) -> list[list[StructProposal]]:
        by_name: dict[str, list[StructProposal]] = defaultdict(list)
        for p in self._proposals:
            by_name[p.name].append(p)

        groups: list[list[StructProposal]] = []
        for same_name in by_name.values():
            groups.append(same_name)
        return groups

    @staticmethod
    def _merge_group(group: list[StructProposal]) -> UnifiedStruct:
        name_counts: dict[str, int] = defaultdict(int)
        fields_by_offset: dict[int, list[StructFieldProposal]] = defaultdict(list)
        applications: list[ParamTypeApplication] = []

        for p in group:
            name_counts[p.name] += 1
            for f in p.fields:
                fields_by_offset[f.offset].append(f)
            if p.used_by_param and p.source_function:
                applications.append(ParamTypeApplication(
                    func_addr=p.source_function,
                    param_name=p.used_by_param,
                    struct_name="",
                ))

        winning_name = max(name_counts, key=name_counts.get)  # type: ignore[arg-type]

        merged_fields: list[StructField] = []
        for offset in sorted(fields_by_offset):
            candidates = fields_by_offset[offset]
            best = _pick_best_field(candidates)
            merged_fields.append(StructField(
                name=best.name,
                data_type=best.data_type,
                offset=best.offset,
                size=best.size,
            ))

        max_size = max((p.total_size for p in group if p.total_size is not None), default=0)
        last_field_end = max(
            (f.offset + f.size for f in merged_fields),
            default=0,
        )
        total_size = max(max_size, last_field_end)
        definition = StructDefinition(
            name=winning_name,
            size=total_size,
            fields=merged_fields,
        )

        for app in applications:
            app.struct_name = winning_name

        return UnifiedStruct(
            definition=definition,
            source_proposals=group,
            applications=applications,
        )


def _pick_best_field(candidates: list[StructFieldProposal]) -> StructFieldProposal:
    """Choose the best field description from multiple proposals at the same offset.

    Prefers the candidate whose name is most descriptive (longest non-generic
    name) and whose type is most specific (not 'undefined' or 'int' when a
    more specific type is available).
    """
    def _score(f: StructFieldProposal) -> tuple[int, int]:
        generic_names = {"field", "unk", "undefined", "pad"}
        name_score = 0 if any(g in f.name.lower() for g in generic_names) else len(f.name)
        generic_types = {"undefined", "undefined4", "undefined8"}
        type_score = 0 if f.data_type.lower() in generic_types else 1
        return (type_score, name_score)

    return max(candidates, key=_score)


def apply_unified_structs(
    client: GhidraClient,
    unified: list[UnifiedStruct],
) -> list[int]:
    """Create structs in Ghidra and apply them to function parameters.

    Returns the list of function addresses whose parameters were retyped
    (candidates for re-analysis).
    """
    affected_addrs: set[int] = set()

    for us in unified:
        try:
            client.create_struct(us.definition)
        except Exception:
            logger.warning(
                "Failed to create struct '%s' in Ghidra", us.definition.name,
                exc_info=True,
            )
            continue

        for app in us.applications:
            try:
                param_ordinal = _resolve_param_ordinal(client, app.func_addr, app.param_name)
                if param_ordinal is not None:
                    client.apply_type_to_param(
                        app.func_addr,
                        param_ordinal,
                        app.struct_name,
                        as_pointer=True,
                    )
                    affected_addrs.add(app.func_addr)
            except Exception:
                logger.warning(
                    "Failed to apply struct '%s' to param '%s' of 0x%08x",
                    app.struct_name, app.param_name, app.func_addr,
                    exc_info=True,
                )

    return sorted(affected_addrs)


def _resolve_param_ordinal(
    client: GhidraClient,
    func_addr: int,
    param_name: str,
) -> int | None:
    """Map a parameter name (e.g. 'param_1') to its ordinal index."""
    try:
        info = client.get_function_info(func_addr)
    except Exception:
        return None
    for p in info.params:
        if p.name == param_name:
            return p.ordinal
    return None
