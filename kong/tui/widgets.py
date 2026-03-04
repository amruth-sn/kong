"""Custom Textual widgets for the Kong TUI."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import RichLog, Static


class BinaryHeader(Static):
    """One-line binary info display."""

    DEFAULT_CSS = """
    BinaryHeader {
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $text;
    }
    """

    _content: str = ""

    def set_info(self, name: str, arch: str, fmt: str, compiler: str) -> None:
        self._content = f"Binary: {name} | {arch} {fmt} | {compiler}"
        self.update(self._content)


class ProgressWidget(Widget):
    """Phase indicator with progress bar and confidence breakdown."""

    DEFAULT_CSS = """
    ProgressWidget {
        height: 3;
        padding: 0 1;
    }
    """

    phase_name: reactive[str] = reactive("Initializing")
    completed: reactive[int] = reactive(0)
    total: reactive[int] = reactive(0)
    high: reactive[int] = reactive(0)
    medium: reactive[int] = reactive(0)
    low: reactive[int] = reactive(0)

    def update_phase(self, name: str) -> None:
        self.phase_name = name

    def update_progress(self, completed: int, total: int) -> None:
        self.completed = completed
        self.total = total

    def update_confidence(self, high: int, medium: int, low: int) -> None:
        self.high = high
        self.medium = medium
        self.low = low

    def render(self) -> Text:
        bar_width = 30
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        line1 = f"Phase: {self.phase_name}  {bar}  {self.completed}/{self.total} ({pct:.0f}%)"

        line2 = Text()
        line2.append("Confidence: ")
        line2.append("■■■", style="green")
        line2.append(f" {self.high} high  ")
        line2.append("■■", style="yellow")
        line2.append(f" {self.medium} med  ")
        line2.append("■", style="red")
        line2.append(f" {self.low} low")

        result = Text(line1 + "\n")
        result.append(line2)
        return result


class AgentLog(RichLog):
    """Scrollable event log with timestamps."""

    DEFAULT_CSS = """
    AgentLog {
        height: 1fr;
        border: solid $primary;
    }
    """

    def log_event(self, message: str, style: str = "") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = Text(f"[{timestamp}] {message}")
        if style:
            text.stylize(style)
        self.write(text)


class StatusBar(Static):
    """Footer with keybinds and status info."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    """

    _content: str = ""

    def update_status(
        self,
        elapsed: float,
        llm_calls: int,
        cost: float,
        paused: bool,
    ) -> None:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        text = (
            f"[p]ause  [q]uit  [e]xport now    "
            f"LLM: {llm_calls} | ${cost:.4f} | {minutes}m {seconds:02d}s"
        )
        if paused:
            text += "  PAUSED"
        self._content = text
        self.update(self._content)
