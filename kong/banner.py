"""ASCII banner and styled output for Kong CLI."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kong import __version__
from kong.config import LLMProvider

KONG_ASCII = r"""
 ██╗  ██╗ ██████╗ ███╗   ██╗ ██████╗
 ██║ ██╔╝██╔═══██╗████╗  ██║██╔════╝
 █████╔╝ ██║   ██║██╔██╗ ██║██║  ███╗
 ██╔═██╗ ██║   ██║██║╚██╗██║██║   ██║
 ██║  ██╗╚██████╔╝██║ ╚████║╚██████╔╝
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝"""


def print_banner(console: Console) -> None:
    banner = Text(KONG_ASCII, style="bold red")
    subtitle = Text("World's first AI reverse engineer", style="dim white")
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


_ENV_VARS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
}

_KEY_EXAMPLES: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "sk-ant-...",
    LLMProvider.OPENAI: "sk-...",
}

_KEY_URLS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "https://console.anthropic.com/settings/keys",
    LLMProvider.OPENAI: "https://platform.openai.com/api-keys",
}


def _env_var_for(provider: LLMProvider) -> str | None:
    return _ENV_VARS.get(provider)


def _load_dotenv(provider: LLMProvider) -> None:
    """Load the API key for *provider* from .env if not already set.

    Checks cwd first, then the package root (two levels up from this file).
    """
    env_var = _env_var_for(provider)
    if env_var is None:
        return
    if os.environ.get(env_var):
        return
    package_root = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd() / ".env", package_root / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{env_var}="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    os.environ[env_var] = value
                return


def check_api_key(provider: LLMProvider) -> bool:
    env_var = _env_var_for(provider)
    if env_var is None:
        return True
    _load_dotenv(provider)
    return bool(os.environ.get(env_var))


def print_setup_needed(console: Console, provider: LLMProvider = LLMProvider.ANTHROPIC) -> None:
    env_var = _env_var_for(provider)
    if env_var is None:
        console.print()
        console.print("[bold red]Custom provider not configured.[/bold red]")
        console.print("Run [bold cyan]kong setup[/bold cyan] to configure a custom endpoint.")
        return
    console.print()
    console.print(f"[bold red]{env_var} is not set.[/bold red]")
    console.print()
    console.print(f"Kong requires a {provider.display_name} API key to analyze binaries.")
    console.print("Run [bold cyan]kong setup[/bold cyan] for guided configuration.")
    console.print()
    key_example = _KEY_EXAMPLES.get(provider, "your-key-here")
    console.print("Or set it directly:")
    console.print(f"  [bold]export {env_var}={key_example}[/bold]")


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
