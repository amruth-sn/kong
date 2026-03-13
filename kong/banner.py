"""ASCII banner and styled output for Kong CLI."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kong import __version__

KONG_ASCII = r"""
 ██╗  ██╗ ██████╗ ███╗   ██╗ ██████╗
 ██║ ██╔╝██╔═══██╗████╗  ██║██╔════╝
 █████╔╝ ██║   ██║██╔██╗ ██║██║  ███╗
 ██╔═██╗ ██║   ██║██║╚██╗██║██║   ██║
 ██║  ██╗╚██████╔╝██║ ╚████║╚██████╔╝
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝"""


def print_banner(console: Console) -> None:
    banner = Text(KONG_ASCII, style="bold red")
    subtitle = Text("Autonomous Binary Analysis Engine", style="dim white")
    version = Text(f"v{__version__}", style="bold white")

    content = Text()
    content.append_text(banner)
    content.append("\n\n")
    content.append_text(subtitle)
    content.append("  ")
    content.append_text(version)

    console.print(Panel(
        content,
        border_style="cyan",
        padding=(0, 2),
    ))


def _load_dotenv() -> None:
    """Load ANTHROPIC_API_KEY from .env if not already in the environment.

    Checks cwd first, then the package root (two levels up from this file).
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    package_root = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd() / ".env", package_root / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    os.environ["ANTHROPIC_API_KEY"] = value
                return


def check_api_key() -> bool:
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def print_setup_needed(console: Console) -> None:
    console.print()
    console.print("[bold red]ANTHROPIC_API_KEY is not set.[/bold red]")
    console.print()
    console.print("Kong requires an Anthropic API key to analyze binaries.")
    console.print("Run [bold cyan]kong setup[/bold cyan] for guided configuration.")
    console.print()
    console.print("Or set it directly:")
    console.print("  [bold]export ANTHROPIC_API_KEY=sk-ant-...[/bold]")


def print_analyze_header(
    console: Console,
    binary_path: str,
    output_dir: str,
    formats: list[str],
) -> None:
    print_banner(console)
    console.print()
    console.print(Panel(
        f"[bold]Target:[/bold]   {binary_path}\n"
        f"[bold]Output:[/bold]   {output_dir}\n"
        f"[bold]Formats:[/bold]  {', '.join(formats)}",
        title="[bold white]Analysis Configuration[/bold white]",
        border_style="dim cyan",
        padding=(0, 2),
    ))
    console.print()
