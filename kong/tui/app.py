"""Main Textual application for Kong TUI."""

from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message

from kong.agent.events import Event, EventType, Phase
from kong.agent.supervisor import Supervisor
from kong.tui.widgets import AgentLog, BinaryHeader, ProgressWidget, StatusBar

EVENT_STYLES: dict[EventType, str] = {
    EventType.PHASE_START: "bold cyan",
    EventType.PHASE_COMPLETE: "bold green",
    EventType.FUNCTION_COMPLETE: "green",
    EventType.FUNCTION_SKIPPED: "dim",
    EventType.FUNCTION_ERROR: "red",
    EventType.RUN_ERROR: "bold red",
    EventType.RUN_COMPLETE: "bold green",
    EventType.DEOBFUSCATION_DETECTED: "magenta",
    EventType.DEOBFUSCATION_COMPLETE: "magenta",
    EventType.EXPORT_FILE: "cyan",
    EventType.CLEANUP_TYPE_CREATED: "blue",
}


class AgentEvent(Message):
    """Bridges supervisor events from worker thread to Textual message system."""

    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


class KongApp(App):
    """Kong analysis TUI."""

    ALLOW_SELECT = True

    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; padding: 0 1; background: $primary; color: $text; }
    """

    BINDINGS = [
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("q", "quit_app", "Quit"),
        Binding("e", "export_now", "Export Now"),
    ]

    def __init__(self, supervisor: Supervisor) -> None:
        super().__init__()
        self.supervisor = supervisor
        self._start_time: float = 0.0
        self._paused_total: float = 0.0
        self._pause_start: float | None = None
        self._completed: int = 0
        self._total: int = 0

    def compose(self) -> ComposeResult:
        yield BinaryHeader(id="header")
        yield ProgressWidget()
        yield AgentLog()
        yield StatusBar()

    def on_mount(self) -> None:
        self._start_time = time.time()
        self.supervisor.on_event(self._on_supervisor_event)
        self.run_worker(self._run_analysis, thread=True)
        self.set_interval(1, self._tick)

    def _on_supervisor_event(self, event: Event) -> None:
        self.call_from_thread(self.post_message, AgentEvent(event))

    def _run_analysis(self) -> None:
        self.supervisor.run()

    def on_agent_event(self, message: AgentEvent) -> None:
        event = message.event
        progress = self.query_one(ProgressWidget)
        log = self.query_one(AgentLog)

        if event.type == EventType.RUN_START:
            pass

        if event.type == EventType.TRIAGE_FUNCTIONS_ENUMERATED:
            binary_data = event.data.get("binary_info", {})
            header = self.query_one(BinaryHeader)
            header.set_info(
                name=self.supervisor.binary_info.name if self.supervisor.binary_info else "unknown",
                arch=binary_data.get("arch", ""),
                fmt=binary_data.get("format", ""),
                compiler=binary_data.get("compiler", ""),
            )

        if event.type == EventType.TRIAGE_QUEUE_BUILT:
            self._total = event.data.get("queue_size", 0)
            progress.update_progress(self._completed, self._total)

        if event.type == EventType.PHASE_START:
            phase_name = event.phase.value if event.phase else "unknown"
            progress.update_phase(phase_name.capitalize())

        if event.type in (EventType.FUNCTION_COMPLETE, EventType.FUNCTION_SKIPPED):
            self._completed += 1
            progress.update_progress(self._completed, self._total)
            stats = self.supervisor.stats
            progress.update_confidence(
                stats.high_confidence,
                stats.medium_confidence,
                stats.low_confidence,
            )

        style = EVENT_STYLES.get(event.type, "")
        log.log_event(event.message, style=style)

    def _tick(self) -> None:
        now = time.time()
        current_pause = (now - self._pause_start) if self._pause_start else 0.0
        elapsed = (now - self._start_time - self._paused_total - current_pause) if self._start_time else 0.0
        llm_calls = self.supervisor.stats.llm_calls
        cost = 0.0
        llm_client = getattr(self.supervisor, "llm_client", None)
        if llm_client is not None and hasattr(llm_client, "total_cost_usd"):
            cost = llm_client.total_cost_usd
        status_bar = self.query_one(StatusBar)
        status_bar.update_status(
            elapsed=elapsed,
            llm_calls=llm_calls,
            cost=cost,
            paused=self.supervisor.is_paused,
        )

    def action_toggle_pause(self) -> None:
        if self.supervisor.is_paused:
            if self._pause_start is not None:
                self._paused_total += time.time() - self._pause_start
                self._pause_start = None
            self.supervisor.resume()
        else:
            self._pause_start = time.time()
            self.supervisor.pause()

    def action_quit_app(self) -> None:
        self.supervisor.pause()
        self.exit()

    def action_export_now(self) -> None:
        self.run_worker(self._run_export, thread=True)

    def _run_export(self) -> None:
        self.supervisor.export()
