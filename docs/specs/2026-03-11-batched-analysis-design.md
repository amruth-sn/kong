# Batched Analysis Pipeline

## Problem

Kong's current architecture makes one LLM call per function. For medium-sized binaries (200+ functions), this means 200+ API calls, which:
- Takes 35-40 minutes on Anthropic Tier 1 due to rate limits (30K ITPM Sonnet, 50K ITPM Haiku)
- Costs $8-10+ (mostly Sonnet at $3/$15 per 1M tokens)
- Makes Kong impractical for anything beyond toy binaries

## Solution

Batch multiple functions into single LLM calls. 200 functions becomes ~20-25 calls instead of 200.

## Design

### Batching Strategy

Functions are grouped into batches at each depth level (bottom-up ordering preserved). Two tiers:

- **Haiku batches (10-12 functions):** functions <= 512 bytes, <= 60 lines decompilation, no obfuscation, <= 5 callees
- **Sonnet batches (3-5 functions):** everything else (large, complex, obfuscated)

### Prompt Format

Each batch prompt contains N function decompilations with per-function callee context. System prompt and output schema are cached (shared across all calls).

```
[System prompt - cached]
[Output schema for batch response - cached]

Analyze the following {N} functions from a stripped binary.
For each, provide: name, signature, confidence, classification, comments, reasoning.

=== Function at 0x1000 ===
[decompilation]
Known callees: callee_a (0x2000), callee_b (0x3000)

=== Function at 0x4000 ===
[decompilation]
Known callees: callee_c (0x5000)

...
```

Response: JSON array of N analysis results.

### Concurrency

- Parallel workers: 4 -> 8
- Remove 0.5s stagger between batches
- Add exponential backoff with jitter on 429 responses
- Haiku and Sonnet calls run in parallel (separate rate limit pools)

### Model Selection Thresholds

Current (too aggressive toward Sonnet):
- Haiku: <= 200 bytes, <= 2 callees, <= 30 lines

New:
- Haiku: <= 512 bytes, <= 60 lines, <= 5 callees, no obfuscation
- Sonnet: everything else

### Cleanup & Synthesis

- Cleanup re-analysis: batch low-confidence retries (currently sequential)
- Synthesis: only include functions with shared global references or cross-references, cap at 50 functions per synthesis prompt

### Decompilation Caching

Cache decompilation results in memory (currently re-fetched from Ghidra for synthesis). Simple dict keyed by function address.

## Projected Performance (200-function binary, Tier 1)

| Metric | Current | Batched |
|---|---|---|
| API calls | ~200 | ~20-25 |
| Rate limit wait | 20+ min | ~3 min |
| LLM processing | ~15 min | ~3 min |
| Total time | 35-40 min | 6-8 min |
| Cost | ~$8-10 | ~$2-4 |

## Scope

This spec covers streaming batched mode only. Batch API mode (`--batch` flag using Anthropic's Message Batches API) is deferred to a follow-up.

## Files to Modify

- `kong/agent/supervisor.py` — batch assembly, parallel execution, decompilation cache
- `kong/agent/analyzer.py` — new `analyze_batch()` method alongside existing `analyze()`
- `kong/agent/prompts.py` — batch prompt template, batch output schema
- `kong/llm/client.py` — batch-aware LLM call method
- `kong/agent/supervisor.py` — model selection threshold updates
- Tests for all of the above
