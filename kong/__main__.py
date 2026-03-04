"""Kong CLI — autonomous binary analysis engine."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from kong import __version__
from collections import Counter

from kong.agent.events import Event, EventType
from kong.agent.supervisor import Supervisor
from kong.config import GhidraConfig, KongConfig, OutputConfig
from kong.ghidra.client import GhidraClient, GhidraClientError
from kong.llm.client import AnthropicClient
console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="kong")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Kong — autonomous binary analysis engine.

    Point it at a stripped binary, get back clean decompiled source,
    annotated Ghidra project, and structured JSON.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _print_final_stats(supervisor: Supervisor, llm_client: AnthropicClient) -> None:
    stats = supervisor.stats
    console.print()
    console.print(
        f"[bold]Results:[/bold] {stats.named}/{stats.total_functions} functions named "
        f"({stats.renamed} renamed, {stats.confirmed} confirmed)"
    )
    console.print(
        f"[bold]Confidence:[/bold] {stats.high_confidence} high, "
        f"{stats.medium_confidence} med, {stats.low_confidence} low"
    )
    console.print(f"[bold]LLM calls:[/bold] {stats.llm_calls}")
    for model_name, mu in llm_client.usage.by_model.items():
        short_name = model_name.split("-")[1] if "-" in model_name else model_name
        console.print(
            f"  {short_name}: {mu.calls} calls, "
            f"${mu.cost_usd(model_name):.4f}"
        )
    console.print(f"[bold]Cost:[/bold] ${llm_client.total_cost_usd:.4f}")
    console.print(f"[bold]Duration:[/bold] {stats.duration_seconds:.1f}s")


@cli.command()
@click.argument("binary", type=click.Path(exists=True, dir_okay=False))
@click.option("--headless", is_flag=True, help="Run without TUI (for CI/Docker).")
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./kong_output",
    help="Output directory.",
)
@click.option(
    "--format", "-f",
    "formats",
    type=click.Choice(["source", "json", "ghidra"], case_sensitive=False),
    multiple=True,
    default=["source", "json"],
    help="Output formats.",
)
@click.option("--ghidra-dir", default=None, help="Ghidra installation directory.")
@click.pass_context
def analyze(
    ctx: click.Context,
    binary: str,
    headless: bool,
    output: str,
    formats: tuple[str, ...],
    ghidra_dir: str | None,
) -> None:
    """Analyze a binary with Kong's autonomous agent."""
    config = KongConfig(
        ghidra=GhidraConfig(install_dir=ghidra_dir),
        output=OutputConfig(directory=Path(output), formats=list(formats)),
        headless=headless,
        verbose=ctx.obj["verbose"],
    )

    binary_path = Path(binary).resolve()
    console.print(f"[bold]Kong v{__version__}[/bold]")
    console.print(f"Binary: {binary_path}")
    console.print(f"Output: {config.output.directory}")
    console.print(f"Formats: {', '.join(config.output.formats)}")
    console.print()

    if not config.ghidra.install_dir:
        console.print("[red]Ghidra is not installed or not found.[/red]")
        console.print(
            "\nInstall Ghidra and try again:\n"
            "  [bold]brew install ghidra[/bold]\n"
            "\nOr set [bold]GHIDRA_INSTALL_DIR[/bold] to your Ghidra installation path."
        )
        raise SystemExit(1)

    try:
        with console.status(
            "[bold green]Opening binary in Ghidra (this may take 30-60s on first run) ...",
        ):
            client = GhidraClient(
                binary_path=str(binary_path),
                install_dir=config.ghidra.install_dir,
            )
            client.open()
    except GhidraClientError as e:
        console.print(f"[red]Failed to open binary:[/red] {escape(str(e))}")
        raise SystemExit(1)

    info = client.get_binary_info()
    console.print(f"[green]Loaded.[/green] {info.arch} {info.format} ({info.compiler})")

    llm_client = AnthropicClient()

    def print_event(event: Event) -> None:
        style = {
            EventType.PHASE_START: "bold cyan",
            EventType.PHASE_COMPLETE: "bold green",
            EventType.FUNCTION_COMPLETE: "green",
            EventType.FUNCTION_SKIPPED: "dim",
            EventType.FUNCTION_ERROR: "red",
            EventType.RUN_ERROR: "bold red",
            EventType.RUN_COMPLETE: "bold green",
        }.get(event.type, "")
        if style:
            console.print(f"[{style}]{escape(event.message)}[/{style}]")
        else:
            console.print(escape(event.message))

    supervisor = Supervisor(client, config, llm_client=llm_client)

    if headless:
        supervisor.on_event(print_event)
        try:
            supervisor.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        finally:
            _print_final_stats(supervisor, llm_client)
            client.close()
    else:
        from kong.tui.app import KongApp
        app = KongApp(supervisor)
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            _print_final_stats(supervisor, llm_client)
            client.close()


@cli.command()
@click.argument("binary", type=click.Path(exists=True, dir_okay=False))
@click.option("--ghidra-dir", default=None, help="Ghidra installation directory.")
def info(binary: str, ghidra_dir: str | None) -> None:
    """Show info about a binary."""
    
    ghidra_config = GhidraConfig(install_dir=ghidra_dir)
    if not ghidra_config.install_dir:
        console.print("[red]Ghidra is not installed or not found.[/red]")
        raise SystemExit(1)

    try:
        client = GhidraClient(
            binary_path=str(Path(binary).resolve()),
            install_dir=ghidra_config.install_dir,
        )
        client.open()
    except GhidraClientError as e:
        console.print(f"[red]Failed to open binary:[/red] {e}")
        raise SystemExit(1)

    bi = client.get_binary_info()
    functions = client.list_functions()

    console.print(f"[bold]Binary:[/bold] {bi.name}")
    console.print(f"[bold]Path:[/bold] {bi.path}")
    console.print(f"[bold]Arch:[/bold] {bi.arch}")
    console.print(f"[bold]Format:[/bold] {bi.format}")
    console.print(f"[bold]Endianness:[/bold] {bi.endianness}")
    console.print(f"[bold]Word Size:[/bold] {bi.word_size * 8}-bit")
    console.print(f"[bold]Compiler:[/bold] {bi.compiler}")
    console.print(f"[bold]Functions:[/bold] {len(functions)}")

    # Classification breakdown
    counts = Counter(f.classification.value for f in functions if f.classification)
    for cls, count in sorted(counts.items()):
        console.print(f"  {cls}: {count}")

    client.close()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
