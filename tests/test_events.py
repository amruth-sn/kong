from __future__ import annotations

from kong.agent.events import EventType, Phase


def test_synthesis_phase_exists() -> None:
    assert Phase.SYNTHESIS.value == "synthesis"


def test_synthesis_event_types_exist() -> None:
    assert EventType.SYNTHESIS_GLOBALS_UNIFIED.value == "synthesis_globals_unified"
    assert EventType.SYNTHESIS_STRUCTS_SYNTHESIZED.value == "synthesis_structs_synthesized"
    assert EventType.SYNTHESIS_NAMES_REFINED.value == "synthesis_names_refined"
