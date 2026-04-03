"""Kong CLI — world's first AI reverse engineer."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

from kong import __version__
from kong.agent.events import Event, EventType
from kong.agent.supervisor import Supervisor
from kong.banner import (
    _ENV_VARS,
    _KEY_EXAMPLES,
    _KEY_URLS,
    check_api_key,
    print_analyze_header,
    print_banner,
)
from kong.config import GhidraConfig, KongConfig, LLMConfig, LLMProvider, OutputConfig
from kong.db import get_custom_config, get_default_provider, get_enabled_providers, is_setup_complete, save_setup
from kong.evals.harness import score as eval_score
from kong.ghidra.client import GhidraClient, GhidraClientError
from kong.llm.codex_auth import resolve_codex_credential
from kong.llm.codex_client import CodexClient
from kong.llm.openai_auth import resolve_openai_credential
from kong.llm.usage import TokenUsage
from kong.llm.openai_client import OpenAIClient
from kong.llm.client import AnthropicClient
from kong.tui.app import KongApp


if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

console = Console()


_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "claude-opus-4-6",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.CODEX: "gpt-5-codex",
}

_PROVIDER_LABELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "Anthropic (Claude)",
    LLMProvider.OPENAI: "OpenAI (GPT-4o)",
    LLMProvider.CODEX: "Codex (ChatGPT OAuth)",
    LLMProvider.CUSTOM: "Custom (OpenAI-compatible)",
}

_NOT_NEEDED_STR = "not-needed"


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the appropriate LLM client based on provider config."""
    from kong.llm.usage import register_custom_model

    model = config.model or _DEFAULT_MODELS.get(config.provider, "gpt-4o")
    if config.provider is LLMProvider.CUSTOM:
        # Local servers don't need auth, but the OpenAI SDK rejects None/empty
        # api_key by falling back to OPENAI_API_KEY env var or raising an error.
        # A dummy value satisfies the SDK while local servers ignore it.
        api_key = config.api_key if config.api_key else _NOT_NEEDED_STR
        register_custom_model(model)
        return OpenAIClient(
            model=model,
            base_url=config.base_url,
            api_key=api_key,
        )
    if config.provider is LLMProvider.CODEX:
        return CodexClient(
            model=model,
            max_output_tokens=config.max_output_tokens,
        )
    if config.provider is LLMProvider.OPENAI:
        return OpenAIClient(model=model, api_key=config.api_key)
    return AnthropicClient(model=model, api_key=config.api_key)


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def validate_base_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise click.BadParameter(
            f"base-url must start with http:// or https:// (got '{url}')"
        )
    return url.rstrip("/")


def resolve_provider(cli_override: str | None = None, base_url: str | None = None) -> LLMProvider:
    """Pick the best available provider: CLI flag > DB default > any enabled key."""
    if base_url and not cli_override:
        return LLMProvider.CUSTOM

    if cli_override:
        provider = LLMProvider(cli_override)
        if provider is LLMProvider.CUSTOM:
            return provider
        if check_api_key(provider):
            return provider
        if provider is LLMProvider.CODEX:
            console.print(
                "[yellow]Warning:[/yellow] --provider codex specified but no local Codex OAuth login was found."
            )
        else:
            env_var = _ENV_VARS.get(provider, "unknown")
            console.print(
                f"[yellow]Warning:[/yellow] --provider {provider.value} specified "
                f"but {env_var} is not set."
            )
        raise SystemExit(1)

    default = get_default_provider()
    if default is LLMProvider.CUSTOM:
        return default
    if default and check_api_key(default):
        return default

    for provider in get_enabled_providers():
        if provider is LLMProvider.CUSTOM:
            continue
        if check_api_key(provider):
            return provider

    console.print("[red]No API keys found for any configured provider.[/red]")
    if LLMProvider.OPENAI in get_enabled_providers():
        console.print("OpenAI can also use your local Codex OAuth login.")
    if LLMProvider.CODEX in get_enabled_providers():
        console.print("Codex requires a local ChatGPT login. Run [bold]codex[/bold] first.")
    console.print("Run [bold cyan]kong setup[/bold cyan] to configure providers.")
    raise SystemExit(1)


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
    is_non_billable = supervisor.config.llm.provider in (LLMProvider.CUSTOM, LLMProvider.CODEX)
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
        console.print(
            f"[bold]Tokens:[/bold] {usage.input_tokens:,} in / "
            f"{usage.output_tokens:,} out / "
            f"{usage.total_tokens:,} total"
        )
        for model_name, mu in usage.by_model.items():
            if is_non_billable:
                console.print(
                    f"  {model_name}: {mu.calls} calls, "
                    f"{mu.input_tokens:,} in / {mu.output_tokens:,} out"
                )
            else:
                console.print(
                    f"  {model_name}: {mu.calls} calls, "
                    f"${mu.cost_usd(model_name):.4f}"
                )
    if is_non_billable:
        provider_name = "custom provider" if supervisor.config.llm.provider is LLMProvider.CUSTOM else "Codex"
        console.print(f"[dim]Cost tracking disabled for {provider_name}[/dim]")
    else:
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
    default=None,
    help="LLM provider (anthropic, openai, codex, or custom).",
)
@click.option("--model", "-m", default=None, help="Override the LLM model name.")
@click.option("--base-url", default=None, help="Custom OpenAI-compatible endpoint URL.")
@click.option("--max-prompt-chars", type=int, default=None, help="Override prompt size limit.")
@click.option("--max-chunk-functions", type=int, default=None, help="Override batch size limit.")
@click.option("--max-output-tokens", type=int, default=None, help="Override output token limit.")
@click.pass_context
def analyze(
    ctx: click.Context,
    binary: str,
    headless: bool,
    output: str,
    formats: tuple[str, ...],
    ghidra_dir: str | None,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    max_prompt_chars: int | None,
    max_chunk_functions: int | None,
    max_output_tokens: int | None,
) -> None:
    """Analyze a binary with Kong's autonomous agent."""
    if not is_setup_complete():
        console.print("[yellow]Kong hasn't been set up yet.[/yellow]")
        console.print("Run [bold cyan]kong setup[/bold cyan] first.")
        raise SystemExit(1)

    if base_url:
        base_url = validate_base_url(base_url)
        if provider and provider not in ("custom",):
            console.print("[red]--base-url can only be used with --provider custom[/red]")
            raise SystemExit(1)

    llm_provider = resolve_provider(provider, base_url=base_url)

    api_key: str | None = None
    if llm_provider is LLMProvider.CUSTOM:
        custom_db = get_custom_config()
        base_url = base_url or custom_db.get("custom_base_url")
        model = model or custom_db.get("custom_model")
        api_key = custom_db.get("custom_api_key") or None
        if not model:
            console.print("[red]--model is required for custom provider[/red]")
            raise SystemExit(1)
        if not base_url:
            console.print("[red]--base-url is required for custom provider[/red]")
            raise SystemExit(1)
        if max_prompt_chars is None:
            max_prompt_chars = _int_or_none(custom_db.get("custom_max_prompt_chars"))
        if max_chunk_functions is None:
            max_chunk_functions = _int_or_none(custom_db.get("custom_max_chunk_functions"))
        if max_output_tokens is None:
            max_output_tokens = _int_or_none(custom_db.get("custom_max_output_tokens"))

    config = KongConfig(
        ghidra=GhidraConfig(install_dir=ghidra_dir),
        llm=LLMConfig(
            provider=llm_provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_prompt_chars=max_prompt_chars,
            max_chunk_functions=max_chunk_functions,
            max_output_tokens=max_output_tokens,
        ),
        output=OutputConfig(directory=Path(output), formats=list(formats)),
        headless=headless,
        verbose=ctx.obj["verbose"],
    )

    from kong.llm.probe import probe_endpoint

    if not probe_endpoint(config.llm):
        console.print("[red]Could not connect to LLM endpoint.[/red]")
        if llm_provider is LLMProvider.CUSTOM:
            console.print(f"Ensure your server is running at {config.llm.base_url}")
        elif llm_provider is LLMProvider.CODEX:
            if resolve_codex_credential() is None:
                console.print("Run [bold]codex[/bold] to sign in with ChatGPT first.")
            else:
                console.print("Your local Codex login was found, but the ChatGPT Codex backend rejected the request.")
        elif llm_provider is LLMProvider.OPENAI:
            credential = resolve_openai_credential()
            if credential and credential.source == "codex_oauth":
                console.print(
                    "Codex OAuth was detected, but OpenAI API billing and scopes are managed separately."
                )
                console.print(
                    "If this persists, enable API billing for your OpenAI project or set OPENAI_API_KEY."
                )
        raise SystemExit(1)

    if llm_provider in (LLMProvider.CUSTOM, LLMProvider.CODEX):
        label = "custom provider" if llm_provider is LLMProvider.CUSTOM else "Codex"
        console.print(f"[dim]Cost tracking disabled for {label} (token counts still recorded)[/dim]")

    binary_path = Path(binary).resolve()

    print_analyze_header(
        console,
        binary_path=str(binary_path),
        output_dir=str(config.output.directory),
        formats=config.output.formats,
    )

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
    """Interactive setup wizard for Kong."""
    print_banner(console)
    console.print()
    console.print("[bold]Welcome to Kong setup![/bold]")
    console.print()

    from kong.llm.limits import _DEFAULT_LIMITS
    from kong.llm.probe import probe_endpoint

    current_default = get_default_provider()
    current_enabled = get_enabled_providers()
    saved_custom = get_custom_config()

    def _current_marker(provider: LLMProvider) -> str:
        if provider == current_default:
            return " [green](current default)[/green]"
        if provider in current_enabled:
            return " [cyan](enabled)[/cyan]"
        return ""

    console.print("[bold]Step 1:[/bold] Which LLM providers would you like to use?")
    console.print()
    console.print(f"  [bold]1[/bold]) Anthropic (Claude){_current_marker(LLMProvider.ANTHROPIC)}")
    console.print(f"  [bold]2[/bold]) OpenAI (GPT-4o){_current_marker(LLMProvider.OPENAI)}")
    console.print(f"  [bold]3[/bold]) Codex (ChatGPT OAuth){_current_marker(LLMProvider.CODEX)}")
    console.print(f"  [bold]4[/bold]) Custom endpoint (OpenAI-compatible){_current_marker(LLMProvider.CUSTOM)}")
    console.print("  [bold]5[/bold]) Anthropic + OpenAI")
    console.print("  [bold]6[/bold]) Anthropic + Codex")
    console.print()

    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5", "6"], console=console)
    choice_int = int(choice)

    custom_config: dict[str, str] | None = None
    if choice_int == 1:
        enabled: list[LLMProvider] = [LLMProvider.ANTHROPIC]
    elif choice_int == 2:
        enabled = [LLMProvider.OPENAI]
    elif choice_int == 3:
        enabled = [LLMProvider.CODEX]
    elif choice_int == 4:
        enabled = [LLMProvider.CUSTOM]
    elif choice_int == 5:
        enabled = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
    else:
        enabled = [LLMProvider.ANTHROPIC, LLMProvider.CODEX]

    if LLMProvider.CUSTOM in enabled:
        saved_url = saved_custom.get("custom_base_url", "")
        saved_model = saved_custom.get("custom_model", "")
        saved_key = saved_custom.get("custom_api_key", "")
        saved_max_pc = saved_custom.get("custom_max_prompt_chars", str(_DEFAULT_LIMITS.max_prompt_chars))
        saved_max_cf = saved_custom.get("custom_max_chunk_functions", str(_DEFAULT_LIMITS.max_chunk_functions))
        saved_max_ot = saved_custom.get("custom_max_output_tokens", str(_DEFAULT_LIMITS.max_output_tokens))

        console.print()
        console.print("[bold]Step 2:[/bold] Configure custom endpoint")
        console.print()
        if not saved_url:
            console.print("  Examples:", style="dim")
            console.print("    http://localhost:11434/v1      (Ollama)", style="dim")
            console.print("    http://localhost:8000/v1       (vLLM)", style="dim")
            console.print("    https://openrouter.ai/api/v1  (OpenRouter)", style="dim")
            console.print()
        custom_base_url = Prompt.ask("  Endpoint URL", default=saved_url or None, console=console)
        custom_base_url = validate_base_url(custom_base_url)
        custom_model = Prompt.ask("  Model name", default=saved_model or None, console=console)
        custom_api_key = Prompt.ask("  API key (leave blank for none)", default=saved_key, console=console)
        custom_max_pc = Prompt.ask(
            "  Max prompt size (chars)",
            default=saved_max_pc,
            console=console,
        )
        custom_max_cf = Prompt.ask(
            "  Max functions per batch",
            default=saved_max_cf,
            console=console,
        )
        custom_max_ot = Prompt.ask(
            "  Max output tokens",
            default=saved_max_ot,
            console=console,
        )
        custom_config = {
            "custom_base_url": custom_base_url,
            "custom_model": custom_model,
            "custom_api_key": custom_api_key,
            "custom_max_prompt_chars": custom_max_pc,
            "custom_max_chunk_functions": custom_max_cf,
            "custom_max_output_tokens": custom_max_ot,
        }

        console.print()
        console.print("  Probing endpoint...")
        probe_cfg = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url=custom_base_url,
            api_key=custom_api_key or None,
        )
        if probe_endpoint(probe_cfg):
            console.print("  [green]Connected successfully.[/green]")
        else:
            console.print("  [yellow]Could not connect (server may not be running). Config saved anyway.[/yellow]")

    console.print()
    console.print("[bold]Step 2:[/bold] Checking API keys..." if LLMProvider.CUSTOM not in enabled else "[bold]Step 3:[/bold] Checking API keys...")
    console.print()

    any_key_found = False
    for p in enabled:
        if p is LLMProvider.CUSTOM:
            any_key_found = True
            continue
        if p is LLMProvider.CODEX:
            credential = resolve_codex_credential()
            if credential is not None:
                console.print(
                    f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green] (via {credential.source_label})"
                )
                any_key_found = True
            else:
                console.print(f"  {_PROVIDER_LABELS[p]:25s} [yellow]Not set[/yellow]")
                console.print("    Sign in locally with [bold]codex[/bold]")
            console.print()
            continue
        env_var = _ENV_VARS[p]
        if check_api_key(p):
            if p is LLMProvider.OPENAI:
                credential = resolve_openai_credential()
                if credential is not None:
                    if credential.masked_value:
                        console.print(
                            f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green] "
                            f"({credential.masked_value} via {credential.source_label})"
                        )
                    else:
                        console.print(
                            f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green] "
                            f"(via {credential.source_label})"
                        )
                else:
                    console.print(f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green]")
            else:
                key = os.environ.get(env_var, "")
                masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
                console.print(f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green] ({masked})")
            any_key_found = True
        else:
            console.print(f"  {_PROVIDER_LABELS[p]:25s} [yellow]Not set[/yellow]")
            console.print(f"    Get your key at: [bold]{_KEY_URLS[p]}[/bold]")
            console.print(f"    [bold]export {env_var}={_KEY_EXAMPLES[p]}[/bold]")
            if p is LLMProvider.OPENAI:
                console.print("    Or sign in locally with [bold]codex[/bold]")
        console.print()

    non_custom = [p for p in enabled if p is not LLMProvider.CUSTOM]
    if len(non_custom) > 1:
        console.print("[bold]Step 3:[/bold] Which provider should be the default?")
        console.print()
        for i, p in enumerate(non_custom, 1):
            console.print(f"  [bold]{i}[/bold]) {_PROVIDER_LABELS[p]}")
        console.print()
        default_choice = Prompt.ask(
            "Default",
            choices=[str(i) for i in range(1, len(non_custom) + 1)],
            console=console,
        )
        default_provider = non_custom[int(default_choice) - 1]
    else:
        default_provider = enabled[0]

    save_setup(enabled=enabled, default=default_provider, custom_config=custom_config)

    console.print()
    ghidra_config = GhidraConfig()
    console.print("[bold]Ghidra[/bold]")
    console.print()
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
    if any_key_found and ghidra_config.install_dir:
        console.print(
            f"[bold green]All set![/bold green] Default provider: "
            f"[bold]{_PROVIDER_LABELS[default_provider]}[/bold]"
        )
        console.print("Run [bold cyan]kong analyze <binary>[/bold cyan] to get started.")
    else:
        console.print(
            "[yellow]Setup saved, but some dependencies are missing. "
            "See above for instructions.[/yellow]"
        )


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
