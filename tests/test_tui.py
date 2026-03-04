"""Tests for TUI widgets and app."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

from textual.app import App
from textual.widget import Widget
from textual.widgets import RichLog, Static

from kong.agent.supervisor import Supervisor
from kong.tui.app import EVENT_STYLES, AgentEvent, KongApp
from kong.tui.widgets import AgentLog, BinaryHeader, ProgressWidget, StatusBar


class TestBinaryHeader:
    def test_is_static_subclass(self) -> None:
        widget = BinaryHeader()
        assert isinstance(widget, Static)

    def test_set_info_formats_correctly(self) -> None:
        widget = BinaryHeader()
        widget.set_info(
            name="test.bin",
            arch="x86_64",
            fmt="ELF",
            compiler="GCC",
        )
        expected = "Binary: test.bin | x86_64 ELF | GCC"
        assert widget._content == expected


class TestProgressWidget:
    def test_is_widget_subclass(self) -> None:
        widget = ProgressWidget()
        assert isinstance(widget, Widget)

    def test_default_reactive_values(self) -> None:
        widget = ProgressWidget()
        assert widget.phase_name == "Initializing"
        assert widget.completed == 0
        assert widget.total == 0
        assert widget.high == 0
        assert widget.medium == 0
        assert widget.low == 0

    def test_update_phase(self) -> None:
        widget = ProgressWidget()
        widget.update_phase("Analysis")
        assert widget.phase_name == "Analysis"

    def test_update_progress(self) -> None:
        widget = ProgressWidget()
        widget.update_progress(5, 10)
        assert widget.completed == 5
        assert widget.total == 10

    def test_update_confidence(self) -> None:
        widget = ProgressWidget()
        widget.update_confidence(3, 2, 1)
        assert widget.high == 3
        assert widget.medium == 2
        assert widget.low == 1


class TestAgentLog:
    def test_is_richlog_subclass(self) -> None:
        widget = AgentLog()
        assert isinstance(widget, RichLog)


class TestStatusBar:
    def test_is_static_subclass(self) -> None:
        widget = StatusBar()
        assert isinstance(widget, Static)

    def test_update_status_formats_correctly(self) -> None:
        widget = StatusBar()
        widget.update_status(elapsed=125.0, llm_calls=10, cost=0.0523, paused=False)
        content = widget._content
        assert "LLM: 10" in content
        assert "$0.0523" in content
        assert "2m 05s" in content
        assert "PAUSED" not in content

    def test_update_status_paused(self) -> None:
        widget = StatusBar()
        widget.update_status(elapsed=60.0, llm_calls=5, cost=0.01, paused=True)
        content = widget._content
        assert "PAUSED" in content


class TestKongApp:
    def test_is_app_subclass(self) -> None:
        supervisor = create_autospec(Supervisor, instance=True)
        supervisor._paused = False
        supervisor.stats = MagicMock()
        supervisor.stats.llm_calls = 0
        app = KongApp(supervisor)
        assert isinstance(app, App)

    def test_stores_supervisor(self) -> None:
        supervisor = create_autospec(Supervisor, instance=True)
        supervisor._paused = False
        supervisor.stats = MagicMock()
        supervisor.stats.llm_calls = 0
        app = KongApp(supervisor)
        assert app.supervisor is supervisor


class TestEventStyles:
    def test_event_styles_is_dict(self) -> None:
        assert isinstance(EVENT_STYLES, dict)

    def test_event_styles_has_expected_keys(self) -> None:
        from kong.agent.events import EventType
        expected_keys = {
            EventType.PHASE_START,
            EventType.PHASE_COMPLETE,
            EventType.FUNCTION_COMPLETE,
            EventType.FUNCTION_SKIPPED,
            EventType.FUNCTION_ERROR,
            EventType.RUN_ERROR,
            EventType.RUN_COMPLETE,
            EventType.DEOBFUSCATION_DETECTED,
            EventType.DEOBFUSCATION_COMPLETE,
            EventType.EXPORT_FILE,
            EventType.CLEANUP_TYPE_CREATED,
        }
        assert expected_keys.issubset(EVENT_STYLES.keys())


class TestAgentEvent:
    def test_agent_event_wraps_event(self) -> None:
        from kong.agent.events import Event, EventType
        event = Event(type=EventType.RUN_START, message="test")
        msg = AgentEvent(event)
        assert msg.event is event
