# Batched Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce API calls from N (one per function) to ~N/10 by batching multiple functions into single LLM prompts, cutting analysis time from 35-40 min to 6-8 min for 200-function binaries.

**Architecture:** Functions at the same call-graph depth are grouped into batches (10-12 for Haiku, 3-5 for Sonnet). Each batch prompt contains multiple function decompilations with per-function context. The LLM responds with a JSON array. Bottom-up ordering is preserved — all callees are named before their callers.

**Tech Stack:** Python 3.11+, Anthropic SDK, pytest, existing Kong pipeline

**Spec:** `docs/specs/2026-03-11-batched-analysis-design.md`

---

## Chunk 1: Batch Prompt Infrastructure

### Task 1: Batch output schema in prompts.py

**Files:**
- Modify: `kong/agent/prompts.py`
- Test: `tests/test_prompts.py` (create)

- [ ] **Step 1: Write test for batch output schema existence**

```python
# tests/test_prompts.py
"""Tests for prompt templates."""

from __future__ import annotations

from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT


class TestBatchPrompts:
    def test_batch_output_schema_is_json_array(self) -> None:
        assert '"address"' in BATCH_OUTPUT_SCHEMA
        assert "[" in BATCH_OUTPUT_SCHEMA

    def test_batch_system_prompt_mentions_multiple(self) -> None:
        assert "multiple" in BATCH_SYSTEM_PROMPT.lower() or "batch" in BATCH_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Add BATCH_SYSTEM_PROMPT and BATCH_OUTPUT_SCHEMA to prompts.py**

Add after the existing `OUTPUT_SCHEMA` in `kong/agent/prompts.py`:

```python
BATCH_SYSTEM_PROMPT = """\
You are an expert reverse engineer analyzing stripped binaries. You are given \
multiple decompiled C functions from Ghidra and must determine what each function does.

Your analysis must be precise:
- Name functions based on what they actually do, not what they might do
- Use standard naming conventions (snake_case for C, camelCase only if the binary is C++)
- If you cannot determine a function's purpose, say so — do not guess
- Confidence reflects how certain you are, not how important the function is

Type recovery:
- When a parameter is a pointer and the function accesses it at multiple fixed \
offsets (e.g. *(param + 0x10), param[3], param->field_0x8), propose a struct layout
- Only propose structs when there are at least 2 distinct field accesses at \
concrete offsets — do not propose structs for single-field or ambiguous accesses
- Name struct fields based on how they are used in the function

You respond only with a JSON array. No prose before or after."""


BATCH_OUTPUT_SCHEMA = """\
Respond with exactly one JSON array. Each element corresponds to one function, \
in the same order as presented:
```json
[
  {
    "address": "0x00001000",
    "name": "descriptive_function_name",
    "signature": "return_type name(param_type param_name, ...)",
    "confidence": <0-100>,
    "classification": "<category>",
    "comments": "Brief description of what the function does",
    "reasoning": "Why you chose this name and classification",
    "variables": [{"old_name": "local_10", "new_name": "buffer"}],
    "struct_proposals": [
      {
        "name": "struct_name",
        "total_size": 32,
        "fields": [
          {"name": "field_name", "data_type": "int", "offset": 0, "size": 4}
        ],
        "used_by_param": "param_1"
      }
    ]
  }
]
```

Classification must be one of: crypto, networking, io, memory, string, math, \
init, cleanup, handler, parser, utility, unknown.

Confidence guidelines:
- 90-100: You recognize this exact algorithm or pattern (e.g., RC4, quicksort)
- 70-89: Strong structural evidence for the name (clear string refs, API calls)
- 50-69: Reasonable inference from context but multiple interpretations possible
- 30-49: Educated guess based on limited evidence
- 0-29: Very uncertain, minimal evidence

struct_proposals: Only include when you see a pointer parameter accessed at \
multiple fixed offsets. Omit entirely if there are no struct patterns."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kong/agent/prompts.py tests/test_prompts.py
git commit -m "feat: add batch prompt template and output schema"
```

---

### Task 2: Batch prompt builder in analyzer.py

**Files:**
- Modify: `kong/agent/analyzer.py`
- Test: `tests/test_agent_analyzer.py`

- [ ] **Step 1: Write test for build_batch_prompt**

Add to `tests/test_agent_analyzer.py`:

```python
class TestBuildBatchPrompt:
    def test_batch_prompt_contains_all_functions(self) -> None:
        """Batch prompt should contain separator and decompilation for each context."""
        client = MagicMock()
        llm = MagicMock()
        analyzer = Analyzer(client, llm)

        binary_info = BinaryInfo(
            arch="x86", format="ELF", endianness="little",
            word_size=8, compiler="gcc", name="test",
        )

        contexts = []
        for i in range(3):
            ctx = AnalysisContext(
                function=FunctionInfo(
                    address=0x1000 + i * 0x100,
                    name=f"FUN_{0x1000 + i * 0x100:08x}",
                    size=64,
                ),
                decompilation=f"void func_{i}(void) {{ return; }}",
                binary_info=binary_info,
            )
            contexts.append(ctx)

        prompt = analyzer.build_batch_prompt(contexts)
        assert "=== Function 1" in prompt
        assert "=== Function 2" in prompt
        assert "=== Function 3" in prompt
        assert "func_0" in prompt
        assert "func_2" in prompt
        assert "0x00001000" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_analyzer.py::TestBuildBatchPrompt -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement build_batch_prompt in Analyzer**

Add method to the `Analyzer` class in `kong/agent/analyzer.py`:

```python
def build_batch_prompt(self, contexts: list[AnalysisContext]) -> str:
    """Build a single prompt containing multiple functions for batch analysis."""
    parts = [
        f"Binary: {contexts[0].binary_info.arch} {contexts[0].binary_info.format} "
        f"({contexts[0].binary_info.compiler})",
        "",
        f"Analyze the following {len(contexts)} functions.",
        "",
    ]

    for idx, ctx in enumerate(contexts, 1):
        parts.append(f"=== Function {idx}: {ctx.function.name} "
                     f"(0x{ctx.function.address:08x}) ===")
        parts.append(f"Size: {ctx.function.size} bytes")
        parts.append("")
        parts.append("### Decompilation")
        parts.append("```c")
        parts.append(ctx.decompilation)
        parts.append("```")

        if ctx.referenced_strings:
            parts.append("")
            parts.append("### Referenced Strings")
            for s in ctx.referenced_strings:
                parts.append(f'- "{s}"')

        if ctx.callee_snippets:
            parts.append("")
            parts.append("### Known Callees")
            for cs in ctx.callee_snippets:
                parts.append(f"- 0x{cs.address:08x}: {cs.name}")

        parts.append("")

    if contexts[0].known_functions:
        parts.append("### Already Identified Functions")
        for addr, name in sorted(contexts[0].known_functions.items()):
            parts.append(f"- 0x{addr:08x}: {name}")

    return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_analyzer.py::TestBuildBatchPrompt -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kong/agent/analyzer.py tests/test_agent_analyzer.py
git commit -m "feat: add batch prompt builder to Analyzer"
```

---

### Task 3: Batch JSON parser in analyzer.py

**Files:**
- Modify: `kong/agent/analyzer.py`
- Test: `tests/test_agent_analyzer.py`

- [ ] **Step 1: Write test for parse_llm_json_batch**

Add to `tests/test_agent_analyzer.py`:

```python
class TestParseBatchJson:
    def test_parses_json_array(self) -> None:
        raw = json.dumps([
            {"address": "0x1000", "name": "foo", "signature": "void foo(void)",
             "confidence": 85, "classification": "utility", "comments": "", "reasoning": ""},
            {"address": "0x2000", "name": "bar", "signature": "int bar(int x)",
             "confidence": 70, "classification": "math", "comments": "", "reasoning": ""},
        ])
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"
        assert results[0].confidence == 85

    def test_handles_markdown_fences(self) -> None:
        raw = '```json\n[{"address": "0x1000", "name": "test", "confidence": 50}]\n```'
        results = Analyzer.parse_llm_json_batch(raw)
        assert len(results) == 1
        assert results[0].name == "test"

    def test_returns_empty_on_parse_failure(self) -> None:
        results = Analyzer.parse_llm_json_batch("not json at all")
        assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_analyzer.py::TestParseBatchJson -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement parse_llm_json_batch**

Add static method to `Analyzer` class in `kong/agent/analyzer.py`:

```python
@staticmethod
def parse_llm_json_batch(raw: str) -> list[LLMResponse]:
    """Parse a JSON array response from a batch analysis call."""
    text = raw.strip()

    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse batch LLM response: %s", raw[:200])
        return []

    if not isinstance(data, list):
        single = Analyzer.parse_llm_json(raw)
        return [single] if single.name else []

    results: list[LLMResponse] = []
    for entry in data:
        struct_proposals = []
        for sp in entry.get("struct_proposals", []):
            if not sp.get("name") or not sp.get("total_size"):
                continue
            try:
                total_size = int(sp["total_size"])
            except (ValueError, TypeError):
                continue
            fields = [
                StructFieldProposal(
                    name=f.get("name", f"field_{i}"),
                    data_type=f.get("data_type", "undefined"),
                    offset=int(f.get("offset", 0)),
                    size=int(f.get("size", 4)),
                )
                for i, f in enumerate(sp.get("fields", []))
            ]
            struct_proposals.append(StructProposal(
                name=sp["name"],
                total_size=total_size,
                fields=fields,
                used_by_param=sp.get("used_by_param", ""),
            ))

        results.append(LLMResponse(
            name=entry.get("name", ""),
            signature=entry.get("signature", ""),
            confidence=entry.get("confidence", 0),
            classification=entry.get("classification", ""),
            comments=entry.get("comments", ""),
            reasoning=entry.get("reasoning", ""),
            struct_proposals=struct_proposals,
            raw=raw,
        ))

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_analyzer.py::TestParseBatchJson -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kong/agent/analyzer.py tests/test_agent_analyzer.py
git commit -m "feat: add batch JSON parser for multi-function LLM responses"
```

---

### Task 4: LLM client batch method

**Files:**
- Modify: `kong/llm/client.py`
- Modify: `kong/agent/analyzer.py` (LLMClient protocol)
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write test for analyze_function_batch**

Add to `tests/test_llm_client.py` (check if file exists first, create if not):

```python
class TestAnalyzeFunctionBatch:
    def test_batch_returns_list_of_responses(self) -> None:
        """Verify that analyze_function_batch returns parsed list."""
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text=json.dumps([
            {"address": "0x1000", "name": "foo", "confidence": 80},
            {"address": "0x2000", "name": "bar", "confidence": 70},
        ]))]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        results = client.analyze_function_batch("batch prompt", model="claude-haiku-4-5-20251001")

        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_client.py::TestAnalyzeFunctionBatch -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Add analyze_function_batch to LLMClient protocol**

In `kong/agent/analyzer.py`, add to the `LLMClient` protocol class:

```python
def analyze_function_batch(self, prompt: str, *, model: str | None = None) -> list[LLMResponse]: ...
```

- [ ] **Step 4: Implement analyze_function_batch in AnthropicClient**

Add method to `AnthropicClient` in `kong/llm/client.py`:

```python
def analyze_function_batch(self, prompt: str, *, model: str | None = None) -> list[LLMResponse]:
    """Send a batch analysis prompt and return parsed list of responses."""
    from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT

    effective_model = model or self.model
    message = self._client.messages.create(
        model=effective_model,
        max_tokens=self.max_tokens * 2,
        system=[{
            "type": "text",
            "text": f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = self._extract_text(message)
    self._record_usage(message, effective_model)

    responses = Analyzer.parse_llm_json_batch(raw_text)
    for resp in responses:
        resp.input_tokens = message.usage.input_tokens
        resp.output_tokens = message.usage.output_tokens
    return responses
```

Note: `max_tokens` is doubled for batch responses since we expect N function outputs.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_client.py::TestAnalyzeFunctionBatch -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kong/agent/analyzer.py kong/llm/client.py tests/test_llm_client.py
git commit -m "feat: add analyze_function_batch to LLM client"
```

---

## Chunk 2: Supervisor Batching Integration

### Task 5: Decompilation cache

**Files:**
- Modify: `kong/agent/supervisor.py`
- Test: `tests/test_agent_supervisor.py`

- [ ] **Step 1: Write test for decompilation caching**

Add to `tests/test_agent_supervisor.py`:

```python
class TestDecompilationCache:
    def test_cached_decompilation_avoids_redundant_ghidra_calls(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        sup._decompilation_cache[0x1000] = "cached code"

        result = sup._get_decompilation(0x1000)
        assert result == "cached code"
        client.get_decompilation.assert_not_called()

    def test_uncached_decompilation_calls_ghidra_and_caches(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        client.get_decompilation.return_value = "void foo(void) {}"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        result = sup._get_decompilation(0x1000)
        assert result == "void foo(void) {}"
        assert 0x1000 in sup._decompilation_cache
        client.get_decompilation.assert_called_once_with(0x1000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_supervisor.py::TestDecompilationCache -v`
Expected: FAIL

- [ ] **Step 3: Add decompilation cache to Supervisor**

In `kong/agent/supervisor.py`, add to `__init__`:
```python
self._decompilation_cache: dict[int, str] = {}
```

Add method:
```python
def _get_decompilation(self, addr: int) -> str:
    """Get decompilation with caching to avoid redundant Ghidra calls."""
    if addr not in self._decompilation_cache:
        self._decompilation_cache[addr] = self.client.get_decompilation(addr)
    return self._decompilation_cache[addr]
```

Then replace all `self.client.get_decompilation(...)` calls in supervisor.py with `self._get_decompilation(...)`. There are 3 locations:
- Line 242 in `_run_analysis`
- Line 638 in `_run_synthesis`
- Line 725 in `_run_export`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_supervisor.py::TestDecompilationCache -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kong/agent/supervisor.py tests/test_agent_supervisor.py
git commit -m "feat: add decompilation cache to avoid redundant Ghidra calls"
```

---

### Task 6: Batched analysis in supervisor

This is the core change. Replace the per-function parallel dispatch with batch assembly.

**Files:**
- Modify: `kong/agent/supervisor.py`
- Test: `tests/test_agent_supervisor.py`

- [ ] **Step 1: Write test for batch assembly**

Add to `tests/test_agent_supervisor.py`:

```python
class TestBatchedAnalysis:
    def test_functions_are_batched_by_model(self, tmp_path):
        """Small functions should be grouped into Haiku batches."""
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(12)]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        batch_responses = [
            LLMResponse(name=f"func_{i}", confidence=80, classification="utility")
            for i in range(12)
        ]
        mock_llm.analyze_function_batch.return_value = batch_responses[:10]

        sup = Supervisor(client, config, llm_client=mock_llm)
        batches = sup._assemble_batches([
            WorkItem(function=f, depth=0, callees=[]) for f in funcs
        ])

        # 12 small functions should produce 1-2 Haiku batches
        assert len(batches) >= 1
        for batch_items, model in batches:
            assert len(batch_items) <= HAIKU_BATCH_SIZE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_supervisor.py::TestBatchedAnalysis -v`
Expected: FAIL

- [ ] **Step 3: Update constants and add batch assembly**

In `kong/agent/supervisor.py`, update constants:

```python
MAX_PARALLEL_LLM = 8
HAIKU_BATCH_SIZE = 10
SONNET_BATCH_SIZE = 4
```

Remove `REQUEST_STAGGER_SECONDS`.

Add new `_pick_model` thresholds:

```python
def _pick_model(self, item: WorkItem, decompilation: str) -> str:
    """Choose Haiku for small/simple functions, Sonnet for complex ones."""
    line_count = decompilation.count("\n")
    if (
        item.function.size <= 512
        and len(item.callees) <= 5
        and line_count <= 60
    ):
        return HAIKU_MODEL
    return SONNET_MODEL
```

Add `_assemble_batches` method:

```python
def _assemble_batches(
    self,
    items: list[tuple[WorkItem, str, str]],
) -> list[tuple[list[tuple[WorkItem, str, str]], str]]:
    """Group items into batches by model, respecting batch size limits."""
    haiku_items = [(it, p, m) for it, p, m in items if m == HAIKU_MODEL]
    sonnet_items = [(it, p, m) for it, p, m in items if m == SONNET_MODEL]

    batches: list[tuple[list[tuple[WorkItem, str, str]], str]] = []

    for i in range(0, len(haiku_items), HAIKU_BATCH_SIZE):
        chunk = haiku_items[i:i + HAIKU_BATCH_SIZE]
        batches.append((chunk, HAIKU_MODEL))

    for i in range(0, len(sonnet_items), SONNET_BATCH_SIZE):
        chunk = sonnet_items[i:i + SONNET_BATCH_SIZE]
        batches.append((chunk, SONNET_MODEL))

    return batches
```

- [ ] **Step 4: Rewrite _analyze_parallel to use batches**

Replace the existing `_analyze_parallel` method with a new `_analyze_batched` method that:
1. Calls `_assemble_batches` to group items
2. For each batch, builds a batch prompt via `analyzer.build_batch_prompt`
3. Calls `llm_client.analyze_function_batch`
4. Maps responses back to WorkItems by index
5. Falls back to individual calls if batch parsing fails
6. Uses ThreadPoolExecutor with MAX_PARALLEL_LLM=8 to run batches in parallel

Update `_run_analysis` to call `_analyze_batched` instead of `_analyze_parallel`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_agent_supervisor.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add kong/agent/supervisor.py tests/test_agent_supervisor.py
git commit -m "feat: batched function analysis — 10x fewer API calls"
```

---

### Task 7: Batched cleanup re-analysis

**Files:**
- Modify: `kong/agent/supervisor.py`
- Test: `tests/test_agent_supervisor.py`

- [ ] **Step 1: Write test for batched cleanup**

```python
class TestBatchedCleanup:
    def test_cleanup_reanalysis_uses_batches(self, tmp_path):
        """Low-confidence functions should be re-analyzed in batches, not one at a time."""
        # Verify that analyze_function_batch is called during cleanup
        # rather than individual analyze calls
        pass  # Test structure depends on final implementation
```

- [ ] **Step 2: Modify _run_cleanup to batch re-analysis calls**

Replace the sequential loop at lines 543-602 with batch assembly:
- Group reanalyze candidates by model tier
- Build batch prompts
- Call `analyze_function_batch`
- Map results back and apply the confidence-upgrade check

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_agent_supervisor.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add kong/agent/supervisor.py tests/test_agent_supervisor.py
git commit -m "feat: batch cleanup re-analysis for low-confidence functions"
```

---

## Chunk 3: Synthesis Optimization & Backoff

### Task 8: Selective synthesis prompt

**Files:**
- Modify: `kong/synthesis/semantic.py`
- Test: `tests/test_synthesis.py`

- [ ] **Step 1: Write test for synthesis function cap**

```python
class TestSynthesisCap:
    def test_synthesis_caps_at_50_functions(self) -> None:
        """Synthesis should only include top-50 functions by cross-reference count."""
        results = [
            FunctionResult(address=i, original_name=f"FUN_{i}", name=f"func_{i}")
            for i in range(100)
        ]
        decomps = {i: f"void func_{i}(void) {{ DAT_1000 = 1; }}" for i in range(100)}

        synth = SemanticSynthesizer(MagicMock())
        prompt = synth._build_synthesis_prompt(results, decomps)

        # Should not include all 100 function decompilations
        func_count = prompt.count("### Function:")
        assert func_count <= 50
```

- [ ] **Step 2: Implement function cap in _build_synthesis_prompt**

In `kong/synthesis/semantic.py`, modify `_build_synthesis_prompt` to:
1. Count global references per function
2. Sort functions by cross-reference count (descending)
3. Include only the top 50 functions in the prompt
4. Always include functions that reference multi-use globals

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_synthesis.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add kong/synthesis/semantic.py tests/test_synthesis.py
git commit -m "feat: cap synthesis prompt at 50 most-connected functions"
```

---

### Task 9: Exponential backoff with jitter

**Files:**
- Modify: `kong/llm/client.py`

- [ ] **Step 1: Verify Anthropic SDK retry behavior**

The Anthropic SDK already has `max_retries=5` configured (client.py:102). Verify that this handles 429s with backoff by checking the SDK docs. If so, no additional retry logic is needed — just remove the `REQUEST_STAGGER_SECONDS` sleep (already removed in Task 6).

- [ ] **Step 2: If additional backoff is needed, add to client.py**

Only if the SDK's built-in retry is insufficient. The SDK's `max_retries=5` should handle most 429 cases.

- [ ] **Step 3: Commit if changes were made**

```bash
git add kong/llm/client.py
git commit -m "feat: configure retry backoff for rate limit handling"
```

---

### Task 10: Integration test — full pipeline with batching

**Files:**
- Test: `tests/test_agent_supervisor.py`

- [ ] **Step 1: Write integration test**

```python
class TestBatchedPipelineIntegration:
    def test_full_pipeline_uses_batch_calls(self, tmp_path):
        """End-to-end: supervisor should call analyze_function_batch, not individual analyze_function."""
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(15)]
        client = _make_client(functions=funcs)

        mock_llm = MagicMock()
        # Set up batch response
        batch_response = [
            LLMResponse(name=f"func_{i}", confidence=80, classification="utility")
            for i in range(10)
        ]
        mock_llm.analyze_function_batch.return_value = batch_response

        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        # Should have called batch method, not individual method
        assert mock_llm.analyze_function_batch.call_count > 0
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_agent_supervisor.py::TestBatchedPipelineIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_supervisor.py
git commit -m "test: add integration test for batched analysis pipeline"
```

---

### Task 11: Final validation

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass

- [ ] **Step 2: Run against minitest_stripped to verify correctness**

```bash
source .env && export ANTHROPIC_API_KEY
uv run kong analyze kong/internal-testing/minitest/minitest_stripped --headless --output ./kong_output
uv run kong eval kong_output/analysis.json kong/internal-testing/minitest/minitest.c
```

Expected: Symbol and type accuracy at least as good as baseline (75.4% / 81.1%).
Verify: fewer LLM calls than before (should be 2-3 instead of 14).

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: adjustments from integration testing"
```
