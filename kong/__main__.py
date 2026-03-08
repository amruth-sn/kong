"""Kong CLI — autonomous binary analysis engine."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from kong import __version__
from kong.agent.events import Event, EventType
from kong.agent.supervisor import Supervisor
from kong.banner import check_api_key, print_analyze_header, print_banner, print_setup_needed
from kong.config import GhidraConfig, KongConfig, OutputConfig
from kong.evals.harness import score as eval_score
from kong.ghidra.client import GhidraClient, GhidraClientError
from kong.llm.client import AnthropicClient

console = Console()


@click.group(invoke_without_command=True)
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

    if ctx.invoked_subcommand is None:
        print_banner(console)
        console.print()
        console.print("Usage: [bold]kong analyze <binary>[/bold]")
        console.print("       [bold]kong info <binary>[/bold]")
        console.print("       [bold]kong setup[/bold]")
        console.print()
        console.print("Run [bold]kong --help[/bold] for all options.")


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

    print_analyze_header(
        console,
        binary_path=str(binary_path),
        output_dir=str(config.output.directory),
        formats=config.output.formats,
    )

    if not check_api_key():
        print_setup_needed(console)
        raise SystemExit(1)

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
        # Deferred to avoid loading Textual when running --headless
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


@cli.command()
def setup() -> None:
    """Guided setup for Kong."""
    print_banner(console)
    console.print()

    console.print("[bold]Step 1: Anthropic API Key[/bold]")
    console.print()

    current_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if current_key:
        masked = current_key[:7] + "..." + current_key[-4:]
        console.print(f"  [green]Found:[/green] {masked}")
    else:
        console.print("  [yellow]Not set.[/yellow]")
        console.print()
        console.print("  Get your key at: [bold]https://console.anthropic.com/settings/keys[/bold]")
        console.print()
        key = click.prompt("  Enter your API key", hide_input=True)
        if not key.startswith("sk-ant-"):
            console.print("  [red]Invalid key format. Keys start with sk-ant-[/red]")
            raise SystemExit(1)
        console.print()
        console.print("  Add to your shell profile:")
        console.print(f'  [bold]export ANTHROPIC_API_KEY="{key}"[/bold]')
        os.environ["ANTHROPIC_API_KEY"] = key

    console.print()
    console.print("[bold]Step 2: Ghidra[/bold]")
    console.print()

    ghidra_config = GhidraConfig()
    if ghidra_config.install_dir:
        console.print(f"  [green]Found:[/green] {ghidra_config.install_dir}")
    else:
        console.print("  [yellow]Not found.[/yellow]")
        console.print()
        console.print("  Install Ghidra:")
        console.print("    [bold]brew install ghidra[/bold]  (macOS)")
        console.print("    Or download from [bold]https://ghidra-sre.org[/bold]")
        console.print()
        console.print("  Then set [bold]GHIDRA_INSTALL_DIR[/bold] to the install path.")

    console.print()
    all_good = check_api_key() and ghidra_config.install_dir
    if all_good:
        console.print("[bold green]All set! Run [bold]kong analyze <binary>[/bold] to get started.[/bold green]")
    else:
        console.print("[yellow]Some dependencies are missing. See above for instructions.[/yellow]")


@cli.command(name="eval")
@click.argument("analysis_json", type=click.Path(exists=True, dir_okay=False))
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False))
def eval_cmd(analysis_json: str, source_file: str) -> None:
    """Score a Kong analysis against ground truth source code."""
    scorecard = eval_score(
        analysis_path=Path(analysis_json),
        source_path=Path(source_file),
    )

    console.print(f"[bold]Binary:[/bold] {scorecard.binary}")
    console.print(f"[bold]Functions:[/bold] {scorecard.functions_analyzed} analyzed / {scorecard.total_functions} in source")
    console.print(f"[bold]Symbol Accuracy:[/bold] {scorecard.symbol_accuracy:.1%}")
    console.print(f"[bold]Type Accuracy:[/bold] {scorecard.type_accuracy:.1%}")
    console.print()

    console.print("[bold]Per-Function Scores:[/bold]")
    for pf in scorecard.per_function:
        pred = pf["predicted_name"]
        truth = pf["truth_name"]
        sym = pf["symbol_accuracy"]
        typ = pf["type_accuracy"]
        match_indicator = "[green]OK[/green]" if sym >= 0.8 else "[yellow]~~[/yellow]" if sym > 0 else "[red]NO[/red]"
        console.print(f"  {match_indicator} {pred:30s} -> {truth:20s}  sym={sym:.2f}  type={typ:.2f}")

    console.print()
    console.print(f"[bold]LLM Calls:[/bold] {scorecard.llm_calls}")
    console.print(f"[bold]Duration:[/bold] {scorecard.duration_seconds:.1f}s")
    console.print(f"[bold]Cost:[/bold] ${scorecard.cost_usd:.4f}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
