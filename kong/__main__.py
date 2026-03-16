"""Kong CLI — world's first AI reverse engineer."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape

from kong import __version__
from kong.agent.events import Event, EventType
from kong.agent.supervisor import Supervisor
from kong.banner import check_api_key, print_analyze_header, print_banner, print_setup_needed
from kong.config import GhidraConfig, KongConfig, LLMConfig, LLMProvider, OutputConfig
from kong.evals.harness import score as eval_score
from kong.ghidra.client import GhidraClient, GhidraClientError
from kong.llm.usage import TokenUsage
from kong.banner import _ENV_VARS, _KEY_EXAMPLES, _KEY_URLS
from kong.llm.openai_client import OpenAIClient
from kong.llm.client import AnthropicClient
from kong.tui.app import KongApp


if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

console = Console()


_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "claude-opus-4-6",
    LLMProvider.OPENAI: "gpt-4o",
}


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the appropriate LLM client based on provider config."""
    # will generalize further in the future
    model = config.model or _DEFAULT_MODELS[config.provider]
    if config.provider is LLMProvider.OPENAI:
        return OpenAIClient(model=model, api_key=config.api_key)
    return AnthropicClient(model=model, api_key=config.api_key)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="kong")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Kong — world's first AI reverse engineer.

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


def _print_final_stats(supervisor: Supervisor, llm_client: LLMClient) -> None:
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
    usage = getattr(llm_client, "usage", None)
    if isinstance(usage, TokenUsage):
        for model_name, mu in usage.by_model.items():
            short_name = model_name.split("-")[1] if "-" in model_name else model_name
            console.print(
                f"  {short_name}: {mu.calls} calls, "
                f"${mu.cost_usd(model_name):.4f}"
            )
    total_cost = getattr(llm_client, "total_cost_usd", 0.0)
    console.print(f"[bold]Cost:[/bold] ${total_cost:.4f}")
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
@click.option(
    "--provider", "-p",
    type=click.Choice([p.value for p in LLMProvider], case_sensitive=False),
    default=LLMProvider.ANTHROPIC.value,
    help="LLM provider (anthropic or openai).",
)
@click.option("--model", "-m", default=None, help="Override the LLM model name.")
@click.pass_context
def analyze(
    ctx: click.Context,
    binary: str,
    headless: bool,
    output: str,
    formats: tuple[str, ...],
    ghidra_dir: str | None,
    provider: str,
    model: str | None,
) -> None:
    """Analyze a binary with Kong's autonomous agent."""
    llm_provider = LLMProvider(provider)
    config = KongConfig(
        ghidra=GhidraConfig(install_dir=ghidra_dir),
        llm=LLMConfig(provider=llm_provider, model=model),
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

    if not check_api_key(llm_provider):
        print_setup_needed(console, llm_provider)
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

    llm_client = create_llm_client(config.llm)

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

    step = 1
    for provider in LLMProvider:
        env_var = _ENV_VARS[provider]
        console.print(f"[bold]Step {step}: {provider.display_name} API Key[/bold]")
        console.print()

        current_key = os.environ.get(env_var, "")
        if current_key:
            masked = current_key[:7] + "..." + current_key[-4:]
            console.print(f"  [green]Found:[/green] {masked}")
        else:
            console.print("  [yellow]Not set.[/yellow]")
            console.print()
            console.print(f"  Get your key at: [bold]{_KEY_URLS[provider]}[/bold]")
            console.print(f"  [bold]export {env_var}={_KEY_EXAMPLES[provider]}[/bold]")

        console.print()
        step += 1

    console.print(f"[bold]Step {step}: Ghidra[/bold]")
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
    has_any_key = any(check_api_key(p) for p in LLMProvider)
    all_good = has_any_key and ghidra_config.install_dir
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
