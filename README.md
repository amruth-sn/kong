<div align="left">


# Kong: The Agentic Reverse Engineer

![PyPI - Version](https://img.shields.io/pypi/v/kong-re)
![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F0xamruth)


<img src="./assets/kong-logo.png" alt="Kong: World's first AI reverse engineer" width="50%">

**LLM orchestration for reverse engineering binaries** <br />

</div>

## What is Kong?
Most tasks follow a linear relationship: the more difficult a task, the longer it usually takes. Reverse engineering (and binary analysis) is a task in which the actual difficulty is somewhat trivial, but the time-to-execute can be on the order of hours (and days!), even for a binary with a couple hundred functions.   

Kong automates the mechanical layer, using an NSA-grade reverse engineering framework. Kong can take a fully obfuscated, stripped binary and run a full analysis pipeline: triaging functions, building call-graph context, recovering types and symbols through LLM-guided decompilation, and writing the results back into Ghidra's program database. The output is a binary where some `FUN_00401a30` is now `parse_http_header`, with recovered structs, parameter names, and calling conventions.

**Why this exists**

Stripped binaries lose all the context that makes code readable: function names, type information, variable names, struct layouts. Recovering that context is the bulk of the work in most RE tasks, and it's largely pattern matching: recognizing standard library functions, inferring types from usage, propagating names through call graphs.

LLMs are good at exactly this kind of pattern matching. But pointing an LLM at raw decompiler output and asking "what does this do?" gives you mediocre results. The model lacks calling context, cross-reference information, and the broader picture of how the binary is structured. In addition, most obfuscated binaries introduce extreme techniques in order to prevent reverse engineering.

Kong solves this by building rich context windows from Ghidra's program analysis (call graphs, cross-references, string references, data flow) before ever touching the LLM, then orchestrating the analysis in dependency order so each function benefits from its callees already being named. Additionally, Kong introduces its own, first-of-its-kind, agentic deobfuscation pipeline.

## In Action

<img src="./assets/github-banner.png" alt="Kong: World's first AI reverse engineer" width="100%">

<img src="./assets/kong-demo.gif" alt="Kong: World's first AI reverse engineer" width="100%">

## Features

- **Fully Autonomous Pipeline**: A single command runs the complete analysis. Triage, function analysis, cleanup, semantic synthesis, and export. No manual intervention required.
- **In-Process Ghidra Integration**: Runs Ghidra's analysis engine in-process via PyGhidra and JPype. No server, no RPC, no subprocess overhead. Direct access to the program database.
- **Call-Graph-Ordered Analysis**: Functions are analyzed bottom-up from the call graph. Leaf functions are named first, so callers benefit from already-resolved context in their decompilation.
- **Rich Context Windows**: Each LLM prompt includes the target function's decompilation plus cross-references, string references, caller/callee signatures, and neighboring data; not just raw decompiler output in isolation.
- **Semantic Synthesis**: A post-analysis pass that unifies naming conventions across the binary, synthesizes struct definitions from field access patterns, and resolves inconsistencies between independently analyzed functions.
- **Signature Matching**: Known standard library and cryptographic functions are identified by pattern before LLM analysis, skipping expensive inference for functions with known identities.
- **Syntactic Normalization**: Decompiler output is cleaned up (modulo recovery, negative literal reconstruction, dead assignment removal) before reaching the LLM, reducing noise and token waste. 
- **Agentic Deobfuscation**: Kong uses an agentic deobfuscation pipeline which can identify and remove obfuscation techniques (Control flow flattening, bogus control flow, instruction substitution, string encryption, VM protection, etc.) from the decompiler output.
- **Eval Framework**: Built-in evaluation harness that scores analysis output against ground-truth source code, measuring symbol accuracy (word-based Jaccard) and type accuracy (signature component scoring).
- **Cost-Tracking**: Tracks token usage and costs for each LLM call, and the total cost of the analysis.

## Supported Architectures

Kong works with most Ghidra-decompilable binaries (for now, more to come).

#### Confidence

| | C | C++ | Go | Rust |
|---|---|---|---|---|
| x86 | High | High | Medium | Medium |
| x86-64 | High | High | Medium | Medium |
| ARM (32-bit) | High | High | Medium | Low |
| AArch64 | High | High | Medium | Low |
| MIPS | Medium | Medium | Low | Low |
| PowerPC | Medium | Medium | Low | Low |

**High** — Kong reliably decompiles, deobfuscates, and recovers names, types, and structure.
**Medium** — Decompilation is usable but noisier. Expect partial recovery and lower confidence scores.
**Low** — Decompilation has significant gaps and results will stay incomplete, noisy, or unreadable.

Binary size scales positively with function count, LLM cost, and time to completion. However, binary size also scales negatively with confidence, so keep this in mind when analyzing larger binaries.

## Architecture

Kong uses a five-phase pipeline orchestrated by a supervisor that coordinates triage, parallel analysis, and post-processing:

```
                    ┌──────────────────────┐
                    │       Triage         │
                    │  enumerate, classify,│
                    │  build call graph,   │
                    │  match signatures    │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Analyze    │ │   Analyze    │ │     ...      │
     │  (leaf fns)  │ │ (next tier)  │ │              │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            └────────┬───────┴────────────────┘
                     │
                     ▼
            ┌──────────────────────┐
            │      Cleanup         │
            │  normalize, dedupe   │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │     Synthesis        │
            │  unify names, build  │
            │  structs, deobfuscate│
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │       Export         │
            │  analysis.json +     │
            │  Ghidra writeback    │
            └──────────────────────┘
```

### How it works

**Triage** enumerates all functions in the binary, classifies them by size (trivial / small / medium / large), builds the call graph, detects the source language, and runs signature matching against known standard library and crypto functions. Functions matched by signature are marked as resolved and skip LLM analysis entirely.

**Analysis** processes functions in bottom-up order from the call graph using a work queue. For each function, Kong builds a context window from Ghidra's program database — decompilation, cross-references, string references, and the signatures of already-analyzed callees — normalizes the decompiler output, and sends it to the LLM for name, type, and parameter recovery. If obfuscation is detected in a function's decompilation, Kong runs an agentic deobfuscation pass with symbolic tool access before producing the analysis. Results are written back to Ghidra immediately so downstream callers see updated names.

**Cleanup** unifies struct types from proposals accumulated during analysis and retries any function signatures that failed to apply during the analysis pass.

**Synthesis** takes a global view across all analyzed functions. A single LLM call reviews the most-connected functions, unifies naming conventions, synthesizes struct definitions from field access patterns, and refines names that look inconsistent in the broader context.

**Export** writes the final `analysis.json` and applies all recovered names, types, and signatures back to the Ghidra program database.

## Stack

- **Runtime**: Python 3.11+, managed with [uv](https://github.com/astral-sh/uv)
- **Binary analysis**: [Ghidra](https://ghidra-sre.org/) via [PyGhidra](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/PyGhidra) (in-process, JPype)
- **LLM**: [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (Claude)
- **Symbolic analysis**: [z3-solver](https://github.com/Z3Prover/z3)
- **CLI**: [Click](https://click.palletsprojects.com/)
- **TUI**: [Textual](https://textual.textualize.io/)
- **Display**: [Rich](https://rich.readthedocs.io/)
- **Build**: [hatchling](https://hatch.pypa.io/)
- **Testing**: [pytest](https://pytest.org/)

## Setup

### Prerequisites

- **Python 3.11+** — ([python.org](https://www.python.org/downloads/) or your system package manager)
- **uv** — Python package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Ghidra** — The National Security Agency's reverse engineering framework ([Install Ghidra](https://ghidra-sre.org/InstallationGuide.html))
- **JDK 21+** — Required by Ghidra ([Adoptium](https://adoptium.net/))
- **Anthropic API key** — Get from [Anthropic Console](https://console.anthropic.com)

### Quick Start

```bash
# 1. Install Kong
pip install kong-re

# 2. Set your API key
export ANTHROPIC_API_KEY="your-api-key"

# 3. Analyze a binary
kong analyze ./path/to/stripped_binary
```

Kong will auto-detect your Ghidra and JDK installations, load the binary into an in-process Ghidra instance, and run the full pipeline.

#### From source

```bash
git clone https://github.com/amruth-sn/kong.git
cd kong
uv sync
uv run kong analyze ./path/to/stripped_binary
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for LLM analysis |
| `GHIDRA_INSTALL_DIR` | No | Path to Ghidra installation (auto-detected if not set) |
| `JAVA_HOME` | No | Path to JDK (auto-detected if not set) |

### Usage

```bash
# Analyze a stripped binary (full pipeline)
uv run kong analyze ./binary

# Show binary metadata without running analysis
uv run kong info ./binary

# Evaluate analysis output against ground-truth source
uv run kong eval ./analysis.json ./source.c
```

### Output

Results are written to the output directory (default: `./kong_output_{binary_name}/`):

```
kong_output_{binary_name}/
├── analysis.json         # All recovered function names, types, parameters
└── events.log            # Pipeline execution trace
```

## Project Layout

```
kong/
├── __main__.py           # CLI entry point (click)
├── config.py             # KongConfig, GhidraConfig, OutputConfig
├── agent/
│   ├── supervisor.py     # Pipeline orchestrator
│   ├── triage.py         # Function enumeration + classification
│   ├── analyzer.py       # LLM-guided function analysis
│   ├── queue.py          # BFS work queue from call graph
│   ├── signatures.py     # Known function signature matching
│   ├── prompts.py        # System prompt + output schema
│   ├── events.py         # Phase/event types for pipeline tracing
│   └── models.py         # FunctionResult dataclass
├── ghidra/
│   ├── client.py         # In-process GhidraClient (PyGhidra/JPype)
│   ├── types.py          # FunctionInfo, BinaryInfo, XRef, etc.
│   └── environment.py    # Ghidra/JDK auto-detection
├── llm/
│   └── client.py         # AnthropicClient, TokenUsage, cost tracking
├── normalizer/
│   └── syntactic.py      # Decompiler output normalization
├── synthesis/
│   └── semantic.py       # Global name unification + struct synthesis
├── evals/
│   ├── harness.py        # Ground-truth extraction + scoring
│   └── metrics.py        # symbol_accuracy, type_accuracy
├── export/
│   └── source.py         # analysis.json + Ghidra writeback
├── signatures/
│   ├── stdlib.json       # C standard library signatures
│   └── crypto.json       # Cryptographic function signatures
└── tui/
    └── app.py            # Textual TUI
```

## License

[APACHE](LICENSE)

Kong is licensed under the Apache License 2.0. Kong is a free and open source project.

This license is compatible with the Ghidra license, and allows for commercial use.

## Contributing

Issues and feature requests are welcome via [GitHub Issues](https://github.com/amruth-sn/kong/issues).

Also, don't hesitate to reach out to me on [X](https://x.com/0xamruth) or [LinkedIn](https://www.linkedin.com/in/amruthn/)!

## Acknowledgments

- [Ghidra](https://ghidra-sre.org/)
- [PyGhidra](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/PyGhidra)
- [JPype](https://github.com/jpype-project/jpype)
- [LLM](https://github.com/anthropics/anthropic-sdk-python)
- [Z3](https://github.com/Z3Prover/z3)
- [Textual](https://textual.textualize.io/)
- [Rich](https://rich.readthedocs.io/)

A big shoutout to [KeygraphHQ](https://keygraph.io/)'s [Shannon](https://github.com/KeygraphHQ/shannon) project, which provided the inspiration for this project. My motivation was driven by replicating the same kind of pipeline that Shannon uses for its web-based pentesting tool, and adapting it for binary analysis and decompilation.

---

Fear the monkey.

---

<p align="center">
  <img src="./assets/kong-logo.png" width="40" alt="Kong"> <br />
  <b>Kong</b>: The world's first AI reverse engineer 
</p>
