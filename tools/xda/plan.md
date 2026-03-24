Random
docs/superpowers/plans/2026-03-20-xda-training-and-candle-inference.md

# XDA Per-Architecture Model Training & Candle Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete pipeline to train per-architecture XDA function boundary detection models and load them for inference in Rust via candle.

**Architecture:** Two independent workstreams — (A) a Python training pipeline that produces `.safetensors` model artifacts per ISA, and (B) a Rust candle-based inference module that loads those artifacts and runs function boundary prediction. The training pipeline is a standalone tool in `tools/xda/`. The Rust inference lives in the `kong-rs` workspace as part of `kong-binary`.

**Tech Stack:**
- Training: Python, PyTorch, HuggingFace Transformers, pyelftools
- Inference: Rust, candle (candle-core, candle-nn, candle-transformers), serde
- Dataset: Debian arm64 packages, cross-compiled C/C++ corpus
- Model format: safetensors (shared between Python export and Rust loading)

---

## File Structure

### Python Training Pipeline (`tools/xda/`)

```
tools/xda/
├── pyproject.toml                    # uv project: torch, transformers, pyelftools, safetensors
├── README.md                         # How to train a new arch model
├── scripts/
│   ├── build_corpus.sh               # Cross-compile C projects for a target arch
│   └── fetch_debian_packages.sh      # Download Debian arm64 packages + dbgsym
├── src/
│   ├── __init__.py
│   ├── extract_labels.py             # DWARF → function boundaries → per-byte labels
│   ├── dataset.py                    # ByteDataset: chunking, train/val split, DataLoader
│   ├── tokenizer.py                  # Byte tokenizer (0-255 + 5 special tokens)
│   ├── model.py                      # XDA model definition (BERT + 2-layer MLP head)
│   ├── pretrain.py                   # Masked byte modeling pre-training
│   ├── finetune.py                   # Function boundary fine-tuning
│   └── export.py                     # PyTorch checkpoint → safetensors + config.json
├── configs/
│   ├── pretrain.yaml                 # Pre-training hyperparameters
│   └── finetune.yaml                 # Fine-tuning hyperparameters
└── tests/
    ├── test_extract_labels.py
    ├── test_dataset.py
    ├── test_tokenizer.py
    └── test_model.py
```

### Rust Inference (`kong-rs/crates/kong-ml/`)

```
kong-rs/
├── Cargo.toml                        # Workspace root
├── crates/
│   ├── kong-types/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── binary.rs             # BinaryInfo, Arch enum
│   │       └── function.rs           # FunctionBoundary, AmbiguousRegion
│   └── kong-ml/
│       ├── Cargo.toml                # candle-core, candle-nn, candle-transformers, safetensors, serde, dirs
│       └── src/
│           ├── lib.rs                # pub mod tokenizer, model, detector, registry
│           ├── tokenizer.rs          # ByteTokenizer: bytes → token IDs
│           ├── config.rs             # XdaConfig: loaded from config.json alongside weights
│           ├── model.rs              # XdaModel: BERT encoder + classification head
│           ├── detector.rs           # XdaDetector: sliding window, batching, thresholding
│           └── registry.rs           # ModelRegistry: per-arch download, cache, load
├── tests/
│   └── xda_integration.rs           # End-to-end: load model, run on test bytes, check predictions
└── models/
    └── .gitkeep                      # Model artifacts go here (gitignored, downloaded at runtime)
```

---

## Part A: Python Training Pipeline

### Task 1: Project Scaffolding & Byte Tokenizer

**Files:**
- Create: `tools/xda/pyproject.toml`
- Create: `tools/xda/src/__init__.py`
- Create: `tools/xda/src/tokenizer.py`
- Test: `tools/xda/tests/test_tokenizer.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "xda-training"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "pyelftools>=0.30",
    "safetensors>=0.4",
    "pyyaml>=6.0",
    "tqdm>=4.65",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

- [ ] **Step 2: Write the tokenizer test**

```python
# tools/xda/tests/test_tokenizer.py
from src.tokenizer import ByteTokenizer

def test_special_token_ids():
    tok = ByteTokenizer()
    assert tok.pad_id == 256
    assert tok.cls_id == 257
    assert tok.sep_id == 258
    assert tok.mask_id == 259
    assert tok.unk_id == 260
    assert tok.vocab_size == 261

def test_encode_bytes():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48, 0x89, 0xe5])
    ids = tok.encode(raw)
    # [CLS] + byte IDs + [SEP]
    assert ids == [257, 0x55, 0x48, 0x89, 0xe5, 258]

def test_encode_with_padding():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48])
    ids = tok.encode(raw, max_length=8)
    assert ids == [257, 0x55, 0x48, 258, 256, 256, 256, 256]
    assert len(ids) == 8

def test_attention_mask():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48])
    ids = tok.encode(raw, max_length=8)
    mask = tok.attention_mask(ids)
    assert mask == [1, 1, 1, 1, 0, 0, 0, 0]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd tools/xda && uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement ByteTokenizer**

```python
# tools/xda/src/tokenizer.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """Byte-level tokenizer for XDA. Maps raw bytes 0-255 to token IDs 0-255.
    Special tokens occupy IDs 256-260."""

    pad_id: int = 256
    cls_id: int = 257
    sep_id: int = 258
    mask_id: int = 259
    unk_id: int = 260
    vocab_size: int = 261

    def encode(self, raw_bytes: bytes, max_length: int | None = None) -> list[int]:
        ids = [self.cls_id] + list(raw_bytes) + [self.sep_id]
        if max_length is not None:
            ids = ids[:max_length]
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def attention_mask(self, token_ids: list[int]) -> list[int]:
        return [0 if t == self.pad_id else 1 for t in token_ids]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd tools/xda && uv run pytest tests/test_tokenizer.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Commit**

```bash
git add tools/xda/pyproject.toml tools/xda/src/__init__.py tools/xda/src/tokenizer.py tools/xda/tests/test_tokenizer.py
git commit -m "feat(xda): scaffold training project, implement byte tokenizer"
```

---

### Task 2: Ground Truth Extraction from DWARF

**Files:**
- Create: `tools/xda/src/extract_labels.py`
- Test: `tools/xda/tests/test_extract_labels.py`

- [ ] **Step 1: Write the extraction test**

This test compiles a tiny C program, extracts labels, verifies correctness.

```python
# tools/xda/tests/test_extract_labels.py
import subprocess
import tempfile
from pathlib import Path

from src.extract_labels import extract_function_boundaries, generate_byte_labels


def _compile_test_binary(src: str, out: Path, arch: str = "native") -> Path:
    """Compile a C string to a binary with debug info."""
    src_file = out / "test.c"
    bin_file = out / "test.elf"
    src_file.write_text(src)

    cc = "gcc" if arch == "native" else f"{arch}-linux-gnu-gcc"
    subprocess.run(
        [cc, "-g", "-O0", "-o", str(bin_file), str(src_file)],
        check=True,
    )
    return bin_file


def test_extract_boundaries_from_simple_binary():
    src = """
    int add(int a, int b) { return a + b; }
    int mul(int a, int b) { return a * b; }
    int main() { return add(1, 2) + mul(3, 4); }
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _compile_test_binary(src, Path(tmp))
        boundaries = extract_function_boundaries(str(binary))

        names = {b["name"] for b in boundaries}
        assert "add" in names
        assert "mul" in names
        assert "main" in names

        for b in boundaries:
            assert b["start"] < b["end"]
            assert b["end"] - b["start"] > 0


def test_generate_byte_labels():
    src = "int foo() { return 42; } int main() { return foo(); }"
    with tempfile.TemporaryDirectory() as tmp:
        binary = _compile_test_binary(src, Path(tmp))
        boundaries = extract_function_boundaries(str(binary))
        text_bytes, labels = generate_byte_labels(str(binary), boundaries)

        assert len(text_bytes) == len(labels)
        assert len(text_bytes) > 0

        # At least some bytes should be labeled as function_start (1)
        assert 1 in labels
        # At least some bytes should be labeled as function_body (2)
        assert 2 in labels
        # At least some bytes should be non-function (0) unless .text is fully covered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools/xda && uv run pytest tests/test_extract_labels.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement extract_labels.py**

```python
# tools/xda/src/extract_labels.py
"""Extract function boundaries from DWARF debug info and generate per-byte labels."""

from elftools.elf.elffile import ELFFile


def extract_function_boundaries(elf_path: str) -> list[dict]:
    """Extract function start/end addresses from DWARF debug info.

    Returns list of {"name": str | None, "start": int, "end": int}.
    """
    functions: list[dict] = []
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return []

        dwarf = elf.get_dwarf_info()
        for cu in dwarf.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_subprogram":
                    continue
                if "DW_AT_low_pc" not in die.attributes:
                    continue

                low_pc = die.attributes["DW_AT_low_pc"].value
                high_pc_attr = die.attributes.get("DW_AT_high_pc")
                if high_pc_attr is None:
                    continue

                if high_pc_attr.form in ("DW_FORM_addr",):
                    high_pc = high_pc_attr.value
                else:
                    high_pc = low_pc + high_pc_attr.value

                if high_pc <= low_pc:
                    continue

                name_attr = die.attributes.get("DW_AT_name")
                name = name_attr.value.decode() if name_attr else None
                functions.append({"name": name, "start": low_pc, "end": high_pc})

    return functions


def generate_byte_labels(
    elf_path: str,
    boundaries: list[dict],
) -> tuple[bytes, list[int]]:
    """Generate per-byte labels for the .text section.

    Labels: 0 = non-function, 1 = function_start, 2 = function_body.
    Returns (text_section_bytes, labels).
    """
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        if text is None:
            return b"", []

        text_start = text["sh_addr"]
        text_bytes = text.data()

    labels = [0] * len(text_bytes)

    for func in boundaries:
        start_off = func["start"] - text_start
        end_off = func["end"] - text_start

        if start_off < 0 or start_off >= len(text_bytes):
            continue
        end_off = min(end_off, len(text_bytes))

        labels[start_off] = 1  # function_start
        for i in range(start_off + 1, end_off):
            labels[i] = 2  # function_body

    return bytes(text_bytes), labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools/xda && uv run pytest tests/test_extract_labels.py -v`
Expected: PASS (both tests). Requires `gcc` available on system.

- [ ] **Step 5: Commit**

```bash
git add tools/xda/src/extract_labels.py tools/xda/tests/test_extract_labels.py
git commit -m "feat(xda): DWARF-based ground truth extraction for function boundaries"
```

---

### Task 3: Dataset Pipeline (Chunking + DataLoader)

**Files:**
- Create: `tools/xda/src/dataset.py`
- Test: `tools/xda/tests/test_dataset.py`

- [ ] **Step 1: Write the dataset test**

```python
# tools/xda/tests/test_dataset.py
import torch
from src.dataset import ByteDataset, chunk_labeled_bytes
from src.tokenizer import ByteTokenizer


def test_chunk_labeled_bytes():
    text_bytes = bytes(range(256)) * 4  # 1024 bytes
    labels = [0] * 1024
    labels[0] = 1    # function start at byte 0
    labels[512] = 1  # function start at byte 512

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=512, stride=256)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk["bytes"]) == 512
        assert len(chunk["labels"]) == 512


def test_byte_dataset_shapes():
    text_bytes = bytes(range(256)) * 4
    labels = [0] * 1024
    labels[0] = 1

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=512, stride=256)
    ds = ByteDataset(chunks, ByteTokenizer())

    item = ds[0]
    assert item["input_ids"].shape == (514,)    # 512 + [CLS] + [SEP]
    assert item["attention_mask"].shape == (514,)
    assert item["labels"].shape == (514,)
    assert item["input_ids"][0].item() == 257    # [CLS]
    assert item["input_ids"][-1].item() == 258   # [SEP]
    # Labels for [CLS] and [SEP] should be -100 (ignored in loss)
    assert item["labels"][0].item() == -100
    assert item["labels"][-1].item() == -100


def test_byte_dataset_label_values():
    text_bytes = bytes([0x55, 0x48, 0x89, 0xe5, 0xc3, 0x00, 0x00, 0x00])
    labels = [1, 2, 2, 2, 2, 0, 0, 0]

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=8, stride=8)
    ds = ByteDataset(chunks, ByteTokenizer())

    item = ds[0]
    # Labels offset by 1 for [CLS] prefix
    assert item["labels"][1].item() == 1   # function_start
    assert item["labels"][2].item() == 2   # function_body
    assert item["labels"][6].item() == 0   # non-function
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools/xda && uv run pytest tests/test_dataset.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement dataset.py**

```python
# tools/xda/src/dataset.py
"""Chunking and DataLoader for XDA training."""

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


def chunk_labeled_bytes(
    text_bytes: bytes,
    labels: list[int],
    window_size: int = 512,
    stride: int = 256,
) -> list[dict]:
    """Chunk labeled bytes into fixed-size windows with overlap."""
    chunks = []
    for i in range(0, max(1, len(text_bytes) - window_size + 1), stride):
        end = i + window_size
        if end > len(text_bytes):
            # Pad the last chunk
            b = text_bytes[i:] + bytes(end - len(text_bytes))
            l = labels[i:] + [0] * (end - len(text_bytes))
        else:
            b = text_bytes[i:end]
            l = labels[i:end]
        chunks.append({"bytes": b, "labels": l})
    return chunks


class ByteDataset(Dataset):
    """PyTorch dataset for XDA training. Each item is a fixed-size byte window."""

    def __init__(self, chunks: list[dict], tokenizer: ByteTokenizer) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        raw_bytes = chunk["bytes"]
        raw_labels = chunk["labels"]

        # Tokenize: [CLS] + bytes + [SEP]
        input_ids = self.tokenizer.encode(raw_bytes)
        attention_mask = self.tokenizer.attention_mask(input_ids)

        # Labels: -100 for special tokens (ignored by CrossEntropyLoss)
        labels = [-100] + list(raw_labels) + [-100]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools/xda && uv run pytest tests/test_dataset.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/xda/src/dataset.py tools/xda/tests/test_dataset.py
git commit -m "feat(xda): chunked byte dataset with sliding window for training"
```

---

### Task 4: XDA Model Definition

**Files:**
- Create: `tools/xda/src/model.py`
- Test: `tools/xda/tests/test_model.py`

- [ ] **Step 1: Write the model test**

```python
# tools/xda/tests/test_model.py
import torch
from src.model import XdaModel, XdaConfig


def test_model_config_defaults():
    config = XdaConfig()
    assert config.vocab_size == 261
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.intermediate_size == 3072
    assert config.num_labels == 3
    assert config.max_position_embeddings == 514  # 512 + [CLS] + [SEP]


def test_model_forward_shape():
    config = XdaConfig(
        num_hidden_layers=2,  # smaller for test speed
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
    )
    model = XdaModel(config)

    batch_size = 4
    seq_len = 514
    input_ids = torch.randint(0, 261, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, seq_len, 3)


def test_model_parameter_count():
    """Sanity check that full-size model has ~87M params."""
    config = XdaConfig()
    model = XdaModel(config)
    total = sum(p.numel() for p in model.parameters())
    # Allow some tolerance — should be in the 80-95M range
    assert 80_000_000 < total < 100_000_000, f"Param count {total} outside expected range"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools/xda && uv run pytest tests/test_model.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement XDA model**

```python
# tools/xda/src/model.py
"""XDA model: BERT encoder + 2-layer MLP classification head."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import BertConfig, BertModel


@dataclass
class XdaConfig:
    vocab_size: int = 261                # 256 bytes + 5 special tokens
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514   # 512 byte window + [CLS] + [SEP]
    num_labels: int = 3                  # non-function, function_start, function_body
    classifier_hidden: int = 256         # MLP hidden dimension
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def to_bert_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        )


class XdaModel(nn.Module):
    """XDA: BERT encoder with a 2-layer MLP token classification head."""

    def __init__(self, config: XdaConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BertModel(config.to_bert_config(), add_pooling_layer=False)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.classifier_hidden, config.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits of shape (batch, seq_len, num_labels)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools/xda && uv run pytest tests/test_model.py -v`
Expected: PASS (all 3 tests). The param count test confirms architecture is correct.

- [ ] **Step 5: Commit**

```bash
git add tools/xda/src/model.py tools/xda/tests/test_model.py
git commit -m "feat(xda): BERT-based XDA model with 2-layer MLP classification head"
```

---

### Task 5: Corpus Build Scripts

**Files:**
- Create: `tools/xda/scripts/fetch_debian_packages.sh`
- Create: `tools/xda/scripts/build_corpus.sh`
- Create: `tools/xda/scripts/Dockerfile.corpus`

- [ ] **Step 1: Create Dockerfile for Debian arm64 corpus**

```dockerfile
# tools/xda/scripts/Dockerfile.corpus
# Build with: docker build --platform linux/arm64 -t xda-corpus -f scripts/Dockerfile.corpus .
# Run with:   docker run --platform linux/arm64 -v $(pwd)/corpus:/out xda-corpus

FROM --platform=linux/arm64 debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Real-world C/C++ binaries
    coreutils binutils findutils diffutils grep gawk sed \
    tar gzip bzip2 xz-utils \
    curl wget \
    openssh-client openssh-server \
    openssl \
    sqlite3 \
    nginx-core \
    redis-server \
    lua5.4 \
    vim-tiny \
    tmux \
    git \
    # Debug symbols
    && echo "deb http://deb.debian.org/debian-debug bookworm-debug main" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    coreutils-dbgsym binutils-dbgsym findutils-dbgsym \
    tar-dbgsym grep-dbgsym gawk-dbgsym sed-dbgsym \
    curl-dbgsym openssh-client-dbgsym \
    libssl3-dbgsym libsqlite3-0-dbgsym \
    redis-server-dbgsym lua5.4-dbgsym \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/fetch_debian_packages.sh /extract.sh
RUN chmod +x /extract.sh

CMD ["/extract.sh"]
```

- [ ] **Step 2: Create extraction script**

```bash
#!/usr/bin/env bash
# tools/xda/scripts/fetch_debian_packages.sh
# Pairs stripped binaries with their debug info from /usr/lib/debug
# Output: /out/pairs/<binary_name>/binary and /out/pairs/<binary_name>/debug

set -euo pipefail

OUT_DIR="/out/pairs"
mkdir -p "$OUT_DIR"

count=0

# Find all ELF binaries in standard paths
for bin in /usr/bin/* /usr/sbin/* /usr/lib/*/lib*.so*; do
    [ -f "$bin" ] || continue

    # Check if it's actually ELF
    file_type=$(file -b "$bin" 2>/dev/null) || continue
    echo "$file_type" | grep -q "ELF" || continue

    # Look for corresponding debug info
    # Debian stores debug files in /usr/lib/debug/<original-path>.debug
    debug_path="/usr/lib/debug${bin}.debug"
    if [ ! -f "$debug_path" ]; then
        # Also check build-id based paths
        build_id=$(readelf -n "$bin" 2>/dev/null | grep "Build ID" | awk '{print $3}') || continue
        if [ -n "$build_id" ]; then
            prefix="${build_id:0:2}"
            suffix="${build_id:2}"
            debug_path="/usr/lib/debug/.build-id/${prefix}/${suffix}.debug"
        fi
    fi

    [ -f "$debug_path" ] || continue

    name=$(basename "$bin")
    pair_dir="${OUT_DIR}/${name}"
    mkdir -p "$pair_dir"
    cp "$bin" "${pair_dir}/binary"
    cp "$debug_path" "${pair_dir}/debug"

    count=$((count + 1))
done

echo "Extracted $count binary/debug pairs to $OUT_DIR"
```

- [ ] **Step 3: Create cross-compilation corpus builder**

```bash
#!/usr/bin/env bash
# tools/xda/scripts/build_corpus.sh
# Cross-compiles C projects at multiple optimization levels for a target architecture.
# Usage: ./build_corpus.sh <arch> <output_dir>
# Example: ./build_corpus.sh aarch64 ./corpus/aarch64

set -euo pipefail

ARCH="${1:?Usage: build_corpus.sh <arch> <output_dir>}"
OUT="${2:?Usage: build_corpus.sh <arch> <output_dir>}"

case "$ARCH" in
    x86_64)  CC="gcc"; STRIP="strip" ;;
    aarch64) CC="aarch64-linux-gnu-gcc"; STRIP="aarch64-linux-gnu-strip" ;;
    arm)     CC="arm-linux-gnueabihf-gcc"; STRIP="arm-linux-gnueabihf-strip" ;;
    mips)    CC="mips-linux-gnu-gcc"; STRIP="mips-linux-gnu-strip" ;;
    riscv64) CC="riscv64-linux-gnu-gcc"; STRIP="riscv64-linux-gnu-strip" ;;
    *)       echo "Unknown arch: $ARCH"; exit 1 ;;
esac

# Verify compiler exists
command -v "$CC" >/dev/null 2>&1 || { echo "$CC not found. Install cross-compiler."; exit 1; }

mkdir -p "$OUT"

# Source files to compile — create a set of diverse test programs
SOURCES_DIR=$(mktemp -d)
trap 'rm -rf "$SOURCES_DIR"' EXIT

# Generate diverse source files
cat > "$SOURCES_DIR/algorithms.c" << 'CSRC'
#include <stdlib.h>
#include <string.h>

void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) { int t = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = t; }
}
int binary_search(int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (arr[mid] == target) return mid; if (arr[mid] < target) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
void *mempool_alloc(size_t size) { static char pool[4096]; static size_t offset = 0; if (offset + size > 4096) return NULL; void *p = &pool[offset]; offset += size; return p; }
struct node { int val; struct node *next; };
struct node *list_insert(struct node *head, int val) { struct node *n = malloc(sizeof(*n)); n->val = val; n->next = head; return n; }
int list_sum(struct node *head) { int s = 0; while (head) { s += head->val; head = head->next; } return s; }
unsigned crc32_byte(unsigned crc, unsigned char b) { crc ^= b; for (int i = 0; i < 8; i++) crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1)); return crc; }
int main() { int arr[] = {5,3,1,4,2}; bubble_sort(arr, 5); return binary_search(arr, 5, 3); }
CSRC

cat > "$SOURCES_DIR/string_ops.c" << 'CSRC'
#include <string.h>
#include <ctype.h>
int str_count_char(const char *s, char c) { int n = 0; while (*s) { if (*s == c) n++; s++; } return n; }
void str_reverse(char *s) { int len = strlen(s); for (int i = 0; i < len / 2; i++) { char t = s[i]; s[i] = s[len - 1 - i]; s[len - 1 - i] = t; } }
void str_to_upper(char *s) { while (*s) { *s = toupper(*s); s++; } }
int str_is_palindrome(const char *s) { int l = 0, r = strlen(s) - 1; while (l < r) { if (s[l++] != s[r--]) return 0; } return 1; }
char *str_find_substr(const char *haystack, const char *needle) { int nlen = strlen(needle); if (!nlen) return (char*)haystack; for (; *haystack; haystack++) { if (!strncmp(haystack, needle, nlen)) return (char*)haystack; } return 0; }
int main() { char buf[] = "hello"; str_reverse(buf); return str_is_palindrome(buf); }
CSRC

cat > "$SOURCES_DIR/state_machine.c" << 'CSRC'
enum state { IDLE, RUNNING, PAUSED, ERROR, DONE };
enum event { START, PAUSE, RESUME, FAIL, FINISH };
enum state transition(enum state s, enum event e) {
    switch (s) {
        case IDLE: return e == START ? RUNNING : s;
        case RUNNING: switch (e) { case PAUSE: return PAUSED; case FAIL: return ERROR; case FINISH: return DONE; default: return s; }
        case PAUSED: return e == RESUME ? RUNNING : (e == FAIL ? ERROR : s);
        case ERROR: return s;
        case DONE: return s;
    }
    return s;
}
int run_machine(enum event *events, int n) { enum state s = IDLE; for (int i = 0; i < n; i++) s = transition(s, events[i]); return s; }
int main() { enum event seq[] = {START, PAUSE, RESUME, FINISH}; return run_machine(seq, 4); }
CSRC

# Compile each source at each optimization level
for src in "$SOURCES_DIR"/*.c; do
    name=$(basename "$src" .c)
    for opt in O0 O1 O2 O3 Os; do
        out_debug="${OUT}/${name}_${opt}_debug"
        out_stripped="${OUT}/${name}_${opt}_stripped"

        "$CC" -g "-${opt}" -o "$out_debug" "$src" 2>/dev/null || continue
        cp "$out_debug" "$out_stripped"
        "$STRIP" "$out_stripped"

        echo "Built: ${name}_${opt}"
    done
done

echo "Corpus built in $OUT"
```

- [ ] **Step 4: Test the corpus builder locally**

Run: `cd tools/xda && chmod +x scripts/build_corpus.sh && ./scripts/build_corpus.sh x86_64 ./corpus/x86_64`
Expected: `algorithms_O0`, `algorithms_O1`, ..., `state_machine_Os` (debug + stripped pairs) created in `corpus/x86_64/`.

- [ ] **Step 5: Commit**

```bash
git add tools/xda/scripts/
git commit -m "feat(xda): corpus build scripts for Debian packages and cross-compilation"
```

---

### Task 6: Full Dataset Builder (Corpus → Training Data)

**Files:**
- Create: `tools/xda/src/build_dataset.py`

This ties together extraction + chunking to produce a single training-ready dataset from a corpus directory.

- [ ] **Step 1: Implement the dataset builder**

```python
# tools/xda/src/build_dataset.py
"""Build a training dataset from a corpus of binary/debug pairs."""

import json
import random
from pathlib import Path

from tqdm import tqdm

from .extract_labels import extract_function_boundaries, generate_byte_labels
from .dataset import chunk_labeled_bytes


def build_from_pairs_dir(
    pairs_dir: Path,
    output_path: Path,
    window_size: int = 512,
    stride: int = 256,
    val_split: float = 0.1,
) -> dict:
    """Process all binary/debug pairs into chunked training data.

    Expects pairs_dir to contain subdirectories, each with 'binary' and 'debug' files.
    Saves train.json and val.json to output_path.

    Returns stats dict.
    """
    all_chunks: list[dict] = []
    stats = {"binaries": 0, "functions": 0, "chunks": 0, "errors": []}

    for pair_dir in tqdm(sorted(pairs_dir.iterdir()), desc="Processing binaries"):
        if not pair_dir.is_dir():
            continue

        debug_file = pair_dir / "debug"
        binary_file = pair_dir / "binary"

        # Support both layouts: paired (binary + debug) and single (debug binary with DWARF)
        if debug_file.exists():
            label_source = str(debug_file)
            byte_source = str(binary_file) if binary_file.exists() else str(debug_file)
        else:
            # Single file with embedded DWARF (e.g., from build_corpus.sh debug binaries)
            candidates = [f for f in pair_dir.iterdir() if f.is_file() and "debug" in f.name]
            if not candidates:
                continue
            label_source = str(candidates[0])
            byte_source = label_source

        try:
            boundaries = extract_function_boundaries(label_source)
            if not boundaries:
                continue

            text_bytes, labels = generate_byte_labels(byte_source, boundaries)
            if not text_bytes:
                continue

            chunks = chunk_labeled_bytes(text_bytes, labels, window_size, stride)
            all_chunks.extend(chunks)

            stats["binaries"] += 1
            stats["functions"] += len(boundaries)
            stats["chunks"] += len(chunks)
        except Exception as e:
            stats["errors"].append({"file": str(pair_dir), "error": str(e)})

    # Shuffle and split
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * (1 - val_split))
    train_chunks = all_chunks[:split_idx]
    val_chunks = all_chunks[split_idx:]

    # Serialize — store bytes as list[int] for JSON compatibility
    output_path.mkdir(parents=True, exist_ok=True)

    def serialize_chunks(chunks: list[dict], path: Path) -> None:
        serialized = []
        for c in chunks:
            serialized.append({
                "bytes": list(c["bytes"]),
                "labels": c["labels"],
            })
        path.write_text(json.dumps(serialized))

    serialize_chunks(train_chunks, output_path / "train.json")
    serialize_chunks(val_chunks, output_path / "val.json")

    stats["train_chunks"] = len(train_chunks)
    stats["val_chunks"] = len(val_chunks)
    (output_path / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats


def build_from_corpus_dir(
    corpus_dir: Path,
    output_path: Path,
    window_size: int = 512,
    stride: int = 256,
    val_split: float = 0.1,
) -> dict:
    """Process debug binaries from build_corpus.sh output.

    Expects corpus_dir to contain *_debug and *_stripped files.
    Uses debug files for labels, stripped files for byte content.
    """
    all_chunks: list[dict] = []
    stats = {"binaries": 0, "functions": 0, "chunks": 0, "errors": []}

    debug_files = sorted(corpus_dir.glob("*_debug"))

    for debug_file in tqdm(debug_files, desc="Processing corpus"):
        try:
            boundaries = extract_function_boundaries(str(debug_file))
            if not boundaries:
                continue

            # Use the debug binary itself for .text bytes
            # (stripped version has same .text content but no DWARF)
            text_bytes, labels = generate_byte_labels(str(debug_file), boundaries)
            if not text_bytes:
                continue

            chunks = chunk_labeled_bytes(text_bytes, labels, window_size, stride)
            all_chunks.extend(chunks)

            stats["binaries"] += 1
            stats["functions"] += len(boundaries)
            stats["chunks"] += len(chunks)
        except Exception as e:
            stats["errors"].append({"file": str(debug_file), "error": str(e)})

    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * (1 - val_split))
    train_chunks = all_chunks[:split_idx]
    val_chunks = all_chunks[split_idx:]

    output_path.mkdir(parents=True, exist_ok=True)

    def serialize_chunks(chunks: list[dict], path: Path) -> None:
        serialized = [{"bytes": list(c["bytes"]), "labels": c["labels"]} for c in chunks]
        path.write_text(json.dumps(serialized))

    serialize_chunks(train_chunks, output_path / "train.json")
    serialize_chunks(val_chunks, output_path / "val.json")

    stats["train_chunks"] = len(train_chunks)
    stats["val_chunks"] = len(val_chunks)
    (output_path / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats
```

- [ ] **Step 2: Test end-to-end on x86_64 corpus**

Run: `cd tools/xda && uv run python -c "from src.build_dataset import build_from_corpus_dir; from pathlib import Path; stats = build_from_corpus_dir(Path('./corpus/x86_64'), Path('./data/x86_64')); print(stats)"`
Expected: `train.json` and `val.json` created in `data/x86_64/`, stats printed showing chunk/function counts.

- [ ] **Step 3: Commit**

```bash
git add tools/xda/src/build_dataset.py
git commit -m "feat(xda): end-to-end dataset builder from corpus to training data"
```

---

### Task 7: Pre-training Script (Masked Byte Modeling)

**Files:**
- Create: `tools/xda/src/pretrain.py`
- Create: `tools/xda/configs/pretrain.yaml`

This is the optional step for new architectures. For x86_64 you'd use the CUMLSec checkpoint. For AArch64/ARM/MIPS, you pre-train on unlabeled bytes first.

- [ ] **Step 1: Create pre-training config**

```yaml
# tools/xda/configs/pretrain.yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072

training:
  epochs: 10
  batch_size: 64
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  mask_probability: 0.20    # 20% of bytes masked
  mask_token_ratio: 0.50    # 50% of masked → [MASK], 50% → random byte
  max_length: 514           # 512 + [CLS] + [SEP]
  fp16: true

data:
  window_size: 512
  stride: 256

output:
  save_every_n_epochs: 2
  checkpoint_dir: checkpoints/pretrain
```

- [ ] **Step 2: Implement pre-training script**

```python
# tools/xda/src/pretrain.py
"""Masked byte modeling pre-training for XDA on a target architecture."""

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import XdaConfig, XdaModel
from .tokenizer import ByteTokenizer


class MaskedByteDataset(Dataset):
    """Dataset that applies masking on-the-fly for pre-training."""

    def __init__(
        self,
        chunks: list[dict],
        tokenizer: ByteTokenizer,
        mask_prob: float = 0.2,
        mask_token_ratio: float = 0.5,
    ) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_ratio = mask_token_ratio

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw_bytes = bytes(self.chunks[idx]["bytes"])
        input_ids = self.tokenizer.encode(raw_bytes)
        labels = list(input_ids)  # original tokens are labels

        # Mask byte positions (skip [CLS] at 0 and [SEP] at -1)
        for i in range(1, len(input_ids) - 1):
            if random.random() < self.mask_prob:
                if random.random() < self.mask_token_ratio:
                    input_ids[i] = self.tokenizer.mask_id
                else:
                    input_ids[i] = random.randint(0, 255)
            else:
                labels[i] = -100  # don't compute loss on unmasked

        labels[0] = -100   # [CLS]
        labels[-1] = -100  # [SEP]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pretrain(config_path: str, data_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_chunks = json.loads(Path(data_path, "train.json").read_text())
    print(f"Loaded {len(train_chunks)} training chunks")

    tokenizer = ByteTokenizer()
    dataset = MaskedByteDataset(
        train_chunks,
        tokenizer,
        mask_prob=cfg["training"]["mask_probability"],
        mask_token_ratio=cfg["training"]["mask_token_ratio"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Build model — for pre-training, the "classification" head predicts masked bytes
    # So num_labels = vocab_size (261), not 3
    model_cfg = XdaConfig(
        hidden_size=cfg["model"]["hidden_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        intermediate_size=cfg["model"]["intermediate_size"],
        num_labels=tokenizer.vocab_size,  # predict byte identity
    )
    model = XdaModel(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["fp16"] and device.type == "cuda")

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device.type, enabled=cfg["training"]["fp16"]):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.view(-1, model_cfg.num_labels), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % cfg["output"]["save_every_n_epochs"] == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"pretrain_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), checkpoint_dir / "pretrain_final.pt")
    print(f"Pre-training complete. Final checkpoint: {checkpoint_dir / 'pretrain_final.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    pretrain(args.config, args.data)
```

- [ ] **Step 3: Commit**

```bash
git add tools/xda/src/pretrain.py tools/xda/configs/pretrain.yaml
git commit -m "feat(xda): masked byte modeling pre-training script"
```

---

### Task 8: Fine-tuning Script

**Files:**
- Create: `tools/xda/src/finetune.py`
- Create: `tools/xda/configs/finetune.yaml`

- [ ] **Step 1: Create fine-tuning config**

```yaml
# tools/xda/configs/finetune.yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072

training:
  epochs: 5
  batch_size: 64
  learning_rate: 2.0e-5     # lower LR for fine-tuning
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true

data:
  window_size: 512
  stride: 256

output:
  checkpoint_dir: checkpoints/finetune
  save_best: true

# Optional: path to pre-trained checkpoint to initialize from
pretrained_checkpoint: null   # e.g., "checkpoints/pretrain/pretrain_final.pt"
```

- [ ] **Step 2: Implement fine-tuning script**

```python
# tools/xda/src/finetune.py
"""Fine-tune XDA for function boundary detection (3-class token classification)."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ByteDataset, chunk_labeled_bytes
from .model import XdaConfig, XdaModel
from .tokenizer import ByteTokenizer


def evaluate(model: XdaModel, loader: DataLoader, device: torch.device) -> dict:
    """Compute per-class precision, recall, F1 on validation set."""
    model.eval()
    # Per-class counts: 0=non-func, 1=func_start, 2=func_body
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]

            for cls in range(3):
                tp[cls] += ((preds == cls) & (labels == cls)).sum().item()
                fp[cls] += ((preds == cls) & (labels != cls)).sum().item()
                fn[cls] += ((preds != cls) & (labels == cls)).sum().item()

    results = {}
    for cls, name in enumerate(["non_function", "function_start", "function_body"]):
        precision = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        results[name] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def finetune(config_path: str, data_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load chunked data
    train_chunks = json.loads(Path(data_path, "train.json").read_text())
    val_chunks = json.loads(Path(data_path, "val.json").read_text())
    print(f"Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks")

    tokenizer = ByteTokenizer()
    train_ds = ByteDataset(train_chunks, tokenizer)
    val_ds = ByteDataset(val_chunks, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], num_workers=4)

    # Build model (3-class: non-function, function_start, function_body)
    model_cfg = XdaConfig(
        hidden_size=cfg["model"]["hidden_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        intermediate_size=cfg["model"]["intermediate_size"],
        num_labels=3,
    )
    model = XdaModel(model_cfg)

    # Load pre-trained encoder weights if available
    pretrained_path = cfg.get("pretrained_checkpoint")
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pre-trained weights from {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # Load only encoder weights (skip classifier head — different num_labels)
        encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder_state, strict=False)
        print(f"Loaded {len(encoder_state)} encoder weight tensors")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["fp16"] and device.type == "cuda")

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        # Train
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device.type, enabled=cfg["training"]["fp16"]):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.view(-1, 3), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

        # Evaluate
        results = evaluate(model, val_loader, device)
        fs_f1 = results["function_start"]["f1"]

        print(f"Epoch {epoch + 1}: loss={total_loss / steps:.4f}")
        for name, metrics in results.items():
            print(f"  {name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

        # Save best by function_start F1 (the metric that matters most)
        if cfg["output"]["save_best"] and fs_f1 > best_f1:
            best_f1 = fs_f1
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  New best function_start F1: {fs_f1:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / f"epoch{epoch + 1}.pt")

    # Save final config alongside weights (needed for Rust loading)
    config_out = {
        "vocab_size": model_cfg.vocab_size,
        "hidden_size": model_cfg.hidden_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "intermediate_size": model_cfg.intermediate_size,
        "max_position_embeddings": model_cfg.max_position_embeddings,
        "num_labels": model_cfg.num_labels,
        "classifier_hidden": model_cfg.classifier_hidden,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config_out, indent=2))

    print(f"Fine-tuning complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    finetune(args.config, args.data)
```

- [ ] **Step 3: Commit**

```bash
git add tools/xda/src/finetune.py tools/xda/configs/finetune.yaml
git commit -m "feat(xda): function boundary fine-tuning with per-class eval"
```

---

### Task 9: Export to Safetensors

**Files:**
- Create: `tools/xda/src/export.py`

This is the bridge between Python training and Rust inference. Produces the exact artifacts that candle loads.

- [ ] **Step 1: Implement export script**

```python
# tools/xda/src/export.py
"""Export trained XDA checkpoint to safetensors format for Rust/candle loading."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from .model import XdaConfig, XdaModel


def export(checkpoint_path: str, config_path: str, output_dir: str, arch: str) -> None:
    """Export a PyTorch checkpoint + config to safetensors + config.json.

    Output structure:
        output_dir/
            xda_{arch}.safetensors    # model weights
            xda_{arch}_config.json    # model config (for candle)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)
    config = XdaConfig(**config_dict)

    # Load model and checkpoint
    model = XdaModel(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # Flatten state dict — candle needs simple key → tensor mapping
    # Rename keys to match what the Rust model expects
    flat_state = {}
    for key, tensor in model.state_dict().items():
        # Convert to float32 (candle safetensors loading expects f32)
        flat_state[key] = tensor.float()

    # Save as safetensors
    weights_path = out / f"xda_{arch}.safetensors"
    save_file(flat_state, str(weights_path))

    # Save config JSON (Rust side reads this to construct the model)
    config_out = out / f"xda_{arch}_config.json"
    config_out.write_text(json.dumps(config_dict, indent=2))

    # Print summary
    total_params = sum(t.numel() for t in flat_state.values())
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"Exported {arch} model:")
    print(f"  Weights: {weights_path} ({file_size_mb:.1f} MB)")
    print(f"  Config:  {config_out}")
    print(f"  Parameters: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to config.json from training")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--arch", required=True, help="Architecture name (x86_64, aarch64, etc.)")
    args = parser.parse_args()
    export(args.checkpoint, args.config, args.output, args.arch)
```

- [ ] **Step 2: Commit**

```bash
git add tools/xda/src/export.py
git commit -m "feat(xda): export trained models to safetensors for Rust/candle loading"
```

---

## Part B: Rust Candle Inference

### Task 10: Cargo Workspace & kong-types Crate

**Files:**
- Create: `kong-rs/Cargo.toml` (workspace root)
- Create: `kong-rs/crates/kong-types/Cargo.toml`
- Create: `kong-rs/crates/kong-types/src/lib.rs`
- Create: `kong-rs/crates/kong-types/src/binary.rs`
- Create: `kong-rs/crates/kong-types/src/function.rs`

- [ ] **Step 1: Create workspace Cargo.toml**

```toml
# kong-rs/Cargo.toml
[workspace]
members = [
    "crates/kong-types",
    "crates/kong-ml",
]
resolver = "2"

[workspace.package]
version = "1.0.0"
edition = "2021"
license = "MIT"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 2: Create kong-types crate**

```toml
# kong-rs/crates/kong-types/Cargo.toml
[package]
name = "kong-types"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
```

```rust
// kong-rs/crates/kong-types/src/lib.rs
pub mod binary;
pub mod function;
```

```rust
// kong-rs/crates/kong-types/src/binary.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arch {
    X86_64,
    X86,
    Aarch64,
    Arm,
    Mips,
    Mips64,
    Riscv64,
    PowerPc,
}

impl Arch {
    /// Model artifact name for this architecture.
    pub fn model_name(&self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::X86 => "x86",
            Arch::Aarch64 => "aarch64",
            Arch::Arm => "arm",
            Arch::Mips => "mips",
            Arch::Mips64 => "mips64",
            Arch::Riscv64 => "riscv64",
            Arch::PowerPc => "powerpc",
        }
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.model_name())
    }
}
```

```rust
// kong-rs/crates/kong-types/src/function.rs
use serde::{Deserialize, Serialize};

/// A confirmed function boundary from either heuristic or ML detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBoundary {
    pub start: u64,
    pub end: u64,
    pub confidence: f32,
    pub source: DetectionSource,
}

/// How a function boundary was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionSource {
    Symbol,         // from symbol table
    Heuristic,      // from prologue/call-target analysis
    Ml,             // from XDA model
    ExceptionInfo,  // from .eh_frame / .pdata
}

/// A region of .text that heuristics couldn't confidently classify.
#[derive(Debug, Clone)]
pub struct AmbiguousRegion {
    pub start: u64,
    pub bytes: Vec<u8>,
    pub reason: AmbiguityReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbiguityReason {
    NoPrologue,       // reachable but no recognizable prologue
    IndirectTarget,   // suspected indirect call/jump target
    GapRegion,        // unreached .text gap between known functions
    OptimizedCode,    // code region with non-standard patterns
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd kong-rs && cargo check`
Expected: Compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add kong-rs/
git commit -m "feat(kong-rs): workspace scaffolding with kong-types crate"
```

---

### Task 11: kong-ml Crate — Byte Tokenizer & Config

**Files:**
- Create: `kong-rs/crates/kong-ml/Cargo.toml`
- Create: `kong-rs/crates/kong-ml/src/lib.rs`
- Create: `kong-rs/crates/kong-ml/src/tokenizer.rs`
- Create: `kong-rs/crates/kong-ml/src/config.rs`

- [ ] **Step 1: Create kong-ml Cargo.toml**

```toml
# kong-rs/crates/kong-ml/Cargo.toml
[package]
name = "kong-ml"
version.workspace = true
edition.workspace = true

[dependencies]
kong-types = { path = "../kong-types" }
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
safetensors = "0.4"
serde.workspace = true
serde_json.workspace = true
dirs = "5"
```

- [ ] **Step 2: Implement byte tokenizer in Rust**

```rust
// kong-rs/crates/kong-ml/src/tokenizer.rs

/// Byte-level tokenizer for XDA. Maps raw bytes 0-255 to token IDs 0-255.
/// Special tokens occupy IDs 256-260.
pub struct ByteTokenizer {
    pub pad_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub unk_id: u32,
    pub vocab_size: u32,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self {
            pad_id: 256,
            cls_id: 257,
            sep_id: 258,
            mask_id: 259,
            unk_id: 260,
            vocab_size: 261,
        }
    }
}

impl ByteTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode raw bytes as token IDs: [CLS] + byte_values + [SEP], padded to max_length.
    pub fn encode(&self, bytes: &[u8], max_length: Option<usize>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len() + 2);
        ids.push(self.cls_id);
        ids.extend(bytes.iter().map(|&b| b as u32));
        ids.push(self.sep_id);

        if let Some(max_len) = max_length {
            ids.truncate(max_len);
            ids.resize(max_len, self.pad_id);
        }

        ids
    }

    /// Generate attention mask: 1 for real tokens, 0 for padding.
    pub fn attention_mask(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids
            .iter()
            .map(|&id| if id == self.pad_id { 0 } else { 1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_token_ids() {
        let tok = ByteTokenizer::new();
        assert_eq!(tok.pad_id, 256);
        assert_eq!(tok.cls_id, 257);
        assert_eq!(tok.sep_id, 258);
        assert_eq!(tok.vocab_size, 261);
    }

    #[test]
    fn test_encode_bytes() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48, 0x89, 0xe5];
        let ids = tok.encode(&raw, None);
        assert_eq!(ids, vec![257, 0x55, 0x48, 0x89, 0xe5, 258]);
    }

    #[test]
    fn test_encode_with_padding() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48];
        let ids = tok.encode(&raw, Some(8));
        assert_eq!(ids, vec![257, 0x55, 0x48, 258, 256, 256, 256, 256]);
    }

    #[test]
    fn test_attention_mask() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48];
        let ids = tok.encode(&raw, Some(8));
        let mask = tok.attention_mask(&ids);
        assert_eq!(mask, vec![1, 1, 1, 1, 0, 0, 0, 0]);
    }
}
```

- [ ] **Step 3: Implement config loading**

```rust
// kong-rs/crates/kong-ml/src/config.rs
use serde::{Deserialize, Serialize};

/// XDA model configuration. Loaded from config.json alongside safetensors weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XdaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_labels: usize,
    pub classifier_hidden: usize,
}

impl Default for XdaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 261,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 514,
            num_labels: 3,
            classifier_hidden: 256,
        }
    }
}

impl XdaConfig {
    pub fn from_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Convert to candle-transformers BertConfig for the encoder.
    pub fn to_bert_config(&self) -> candle_transformers::models::bert::Config {
        candle_transformers::models::bert::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            hidden_dropout_prob: 0.0,  // no dropout during inference
            ..Default::default()
        }
    }
}
```

- [ ] **Step 4: Wire up lib.rs**

```rust
// kong-rs/crates/kong-ml/src/lib.rs
pub mod config;
pub mod tokenizer;
pub mod model;
pub mod detector;
pub mod registry;
```

- [ ] **Step 5: Verify it compiles**

Run: `cd kong-rs && cargo check`
Expected: Compiles (model.rs, detector.rs, registry.rs will be empty stubs initially — create them with just `// TODO` if needed to satisfy mod declarations, or gate them behind a cfg flag). For now, comment out the three unimplemented mods in lib.rs:

```rust
// kong-rs/crates/kong-ml/src/lib.rs
pub mod config;
pub mod tokenizer;
// pub mod model;     // Task 12
// pub mod detector;  // Task 13
// pub mod registry;  // Task 14
```

Run: `cd kong-rs && cargo test -p kong-ml`
Expected: All 4 tokenizer tests pass.

- [ ] **Step 6: Commit**

```bash
git add kong-rs/crates/kong-ml/
git commit -m "feat(kong-ml): byte tokenizer and config loading for candle inference"
```

---

### Task 12: XDA Model in Candle

**Files:**
- Create: `kong-rs/crates/kong-ml/src/model.rs`

This is the core Rust implementation — BERT encoder (from candle-transformers) + 2-layer MLP classification head.

- [ ] **Step 1: Implement XdaModel**

```rust
// kong-rs/crates/kong-ml/src/model.rs
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

use crate::config::XdaConfig;

/// 2-layer MLP classification head.
struct ClassifierHead {
    fc1: Linear,
    fc2: Linear,
}

impl ClassifierHead {
    fn load(vb: VarBuilder, hidden_size: usize, classifier_hidden: usize, num_labels: usize) -> Result<Self> {
        let fc1 = linear(hidden_size, classifier_hidden, vb.pp("classifier.0"))?;
        let fc2 = linear(classifier_hidden, num_labels, vb.pp("classifier.3"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(hidden_states)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

/// XDA model: BERT encoder + classification head for function boundary detection.
pub struct XdaModel {
    encoder: BertModel,
    classifier: ClassifierHead,
}

impl XdaModel {
    /// Load model from safetensors weights.
    pub fn load(config: &XdaConfig, vb: VarBuilder) -> Result<Self> {
        let bert_config = config.to_bert_config();
        let encoder = BertModel::load(vb.pp("encoder"), &bert_config)?;
        let classifier = ClassifierHead::load(
            vb.clone(),
            config.hidden_size,
            config.classifier_hidden,
            config.num_labels,
        )?;
        Ok(Self { encoder, classifier })
    }

    /// Run inference. Returns logits of shape (batch, seq_len, num_labels).
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // BertModel::forward requires token_type_ids (mandatory) and attention_mask (optional).
        // For XDA we use single-segment input, so token_type_ids is all zeros.
        let token_type_ids = input_ids.zeros_like()?;
        let hidden = self.encoder.forward(input_ids, &token_type_ids, Some(attention_mask))?;
        self.classifier.forward(&hidden)
    }
}
```

**Important note on weight key mapping:** The Python model saves weights with keys like `encoder.embeddings.word_embeddings.weight`, `classifier.0.weight`, `classifier.3.weight`. The Rust side uses `VarBuilder::pp("encoder")` and `VarBuilder::pp("classifier.0")` etc. to match these keys. If the keys don't align, you'll get a "tensor not found" error at load time — debug by printing the keys in the safetensors file:

```rust
// Debug helper — use temporarily if weight loading fails
let tensors = safetensors::SafeTensors::deserialize(&std::fs::read(path)?)?;
for name in tensors.names() {
    println!("{name}");
}
```

- [ ] **Step 2: Uncomment model in lib.rs**

```rust
// kong-rs/crates/kong-ml/src/lib.rs
pub mod config;
pub mod tokenizer;
pub mod model;
// pub mod detector;  // Task 13
// pub mod registry;  // Task 14
```

- [ ] **Step 3: Verify it compiles**

Run: `cd kong-rs && cargo check -p kong-ml`
Expected: Compiles. No runtime test yet — we need actual weights for that.

- [ ] **Step 4: Commit**

```bash
git add kong-rs/crates/kong-ml/src/model.rs kong-rs/crates/kong-ml/src/lib.rs
git commit -m "feat(kong-ml): XDA model implementation in candle (BERT + MLP head)"
```

---

### Task 13: XDA Detector (Sliding Window + Batched Inference)

**Files:**
- Create: `kong-rs/crates/kong-ml/src/detector.rs`

- [ ] **Step 1: Implement XdaDetector**

```rust
// kong-rs/crates/kong-ml/src/detector.rs
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use kong_types::function::{AmbiguousRegion, DetectionSource, FunctionBoundary};

use crate::config::XdaConfig;
use crate::model::XdaModel;
use crate::tokenizer::ByteTokenizer;

pub struct XdaDetector {
    model: XdaModel,
    tokenizer: ByteTokenizer,
    device: Device,
    confidence_threshold: f32,
    window_size: usize,
    stride: usize,
}

/// Per-byte prediction from a single window.
#[derive(Debug, Clone)]
struct BytePrediction {
    address: u64,
    label: u8,       // 0=non-func, 1=func_start, 2=func_body
    confidence: f32,
}

impl XdaDetector {
    pub fn new(
        model: XdaModel,
        device: Device,
        confidence_threshold: f32,
    ) -> Self {
        Self {
            model,
            tokenizer: ByteTokenizer::new(),
            device,
            confidence_threshold,
            window_size: 512,
            stride: 256,
        }
    }

    /// Predict function boundaries in ambiguous regions.
    pub fn predict(&self, regions: &[AmbiguousRegion]) -> Result<Vec<FunctionBoundary>> {
        let mut boundaries = Vec::new();

        for region in regions {
            let region_predictions = self.predict_region(region)?;
            boundaries.extend(region_predictions);
        }

        Ok(boundaries)
    }

    fn predict_region(&self, region: &AmbiguousRegion) -> Result<Vec<FunctionBoundary>> {
        let bytes = &region.bytes;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        // Slide windows over the region
        let windows = self.create_windows(bytes);
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        // Batch inference
        let predictions = self.run_batch(&windows)?;

        // Merge overlapping window predictions — for overlapping bytes,
        // average the confidence scores
        let merged = self.merge_predictions(&predictions, bytes.len(), region.start);

        // Extract function boundaries from merged predictions
        Ok(self.extract_boundaries(&merged))
    }

    /// Returns (window_bytes, actual_byte_count) pairs. actual_byte_count tracks
    /// how many bytes are real vs padding, since binary code can legitimately contain 0x00.
    fn create_windows(&self, bytes: &[u8]) -> Vec<(Vec<u8>, usize)> {
        let mut windows = Vec::new();
        let mut offset = 0;

        while offset < bytes.len() {
            let end = (offset + self.window_size).min(bytes.len());
            let actual_len = end - offset;
            let mut window = bytes[offset..end].to_vec();

            // Pad if necessary
            if window.len() < self.window_size {
                window.resize(self.window_size, 0);
            }

            windows.push((window, actual_len));
            offset += self.stride;

            if end == bytes.len() {
                break;
            }
        }

        windows
    }

    fn run_batch(&self, windows: &[(Vec<u8>, usize)]) -> Result<Vec<Vec<(u8, f32)>>> {
        let seq_len = self.window_size + 2; // +2 for [CLS] and [SEP]
        let batch_size = windows.len();

        // Tokenize all windows
        let mut all_ids = Vec::with_capacity(batch_size * seq_len);
        let mut all_mask = Vec::with_capacity(batch_size * seq_len);

        for (window, _actual_len) in windows {
            let ids = self.tokenizer.encode(window, Some(seq_len));
            let mask = self.tokenizer.attention_mask(&ids);
            all_ids.extend(ids.iter().map(|&x| x as i64));
            all_mask.extend(mask.iter().map(|&x| x as i64));
        }

        let input_ids = Tensor::from_vec(all_ids, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(all_mask, (batch_size, seq_len), &self.device)?;

        // Forward pass
        let logits = self.model.forward(&input_ids, &attention_mask)?;

        // Softmax over last dimension to get probabilities
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        // Extract per-byte predictions (skip [CLS] and [SEP])
        let mut batch_predictions = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let window_probs = probs.i((i, 1..seq_len - 1))?; // skip [CLS] at 0, [SEP] at end
            let window_probs = window_probs.to_vec2::<f32>()?;

            // Use the tracked actual_len instead of inferring from byte content —
            // binary code can legitimately contain 0x00 bytes.
            let actual_len = windows[i].1;
            let mut preds = Vec::with_capacity(actual_len);

            for j in 0..actual_len {
                let p = &window_probs[j];
                let (label, confidence) = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, &conf)| (idx as u8, conf))
                    .unwrap_or((0, 0.0));
                preds.push((label, confidence));
            }

            batch_predictions.push(preds);
        }

        Ok(batch_predictions)
    }

    fn merge_predictions(
        &self,
        batch_predictions: &[Vec<(u8, f32)>],
        total_bytes: usize,
        base_address: u64,
    ) -> Vec<BytePrediction> {
        // For overlapping windows, accumulate (label_votes, confidence_sum, count)
        let mut votes: Vec<([f32; 3], u32)> = vec![([0.0; 3], 0); total_bytes];

        for (window_idx, preds) in batch_predictions.iter().enumerate() {
            let offset = window_idx * self.stride;
            for (byte_idx, &(label, confidence)) in preds.iter().enumerate() {
                let global_idx = offset + byte_idx;
                if global_idx < total_bytes {
                    votes[global_idx].0[label as usize] += confidence;
                    votes[global_idx].1 += 1;
                }
            }
        }

        votes
            .into_iter()
            .enumerate()
            .map(|(idx, (scores, count))| {
                let count = count.max(1) as f32;
                let avg_scores = [
                    scores[0] / count,
                    scores[1] / count,
                    scores[2] / count,
                ];
                let (label, confidence) = avg_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(l, &c)| (l as u8, c))
                    .unwrap_or((0, 0.0));

                BytePrediction {
                    address: base_address + idx as u64,
                    label,
                    confidence,
                }
            })
            .collect()
    }

    fn extract_boundaries(&self, predictions: &[BytePrediction]) -> Vec<FunctionBoundary> {
        let mut boundaries = Vec::new();
        let mut func_start: Option<&BytePrediction> = None;

        for pred in predictions {
            match pred.label {
                1 if pred.confidence >= self.confidence_threshold => {
                    // Close previous function if open
                    if let Some(start) = func_start.take() {
                        boundaries.push(FunctionBoundary {
                            start: start.address,
                            end: pred.address,
                            confidence: start.confidence,
                            source: DetectionSource::Ml,
                        });
                    }
                    func_start = Some(pred);
                }
                0 if func_start.is_some() => {
                    // Non-function byte after function body — close the function
                    let start = func_start.take().unwrap();
                    boundaries.push(FunctionBoundary {
                        start: start.address,
                        end: pred.address,
                        confidence: start.confidence,
                        source: DetectionSource::Ml,
                    });
                }
                _ => {}
            }
        }

        // Close trailing function
        if let Some(start) = func_start {
            if let Some(last) = predictions.last() {
                boundaries.push(FunctionBoundary {
                    start: start.address,
                    end: last.address + 1,
                    confidence: start.confidence,
                    source: DetectionSource::Ml,
                });
            }
        }

        boundaries
    }
}
```

- [ ] **Step 2: Uncomment detector in lib.rs**

- [ ] **Step 3: Verify it compiles**

Run: `cd kong-rs && cargo check -p kong-ml`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add kong-rs/crates/kong-ml/src/detector.rs kong-rs/crates/kong-ml/src/lib.rs
git commit -m "feat(kong-ml): XDA detector with sliding window and batch inference"
```

---

### Task 14: Model Registry (Download, Cache, Load Per-Arch)

**Files:**
- Create: `kong-rs/crates/kong-ml/src/registry.rs`

- [ ] **Step 1: Implement ModelRegistry**

```rust
// kong-rs/crates/kong-ml/src/registry.rs
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use kong_types::binary::Arch;

use crate::config::XdaConfig;
use crate::detector::XdaDetector;
use crate::model::XdaModel;

pub struct ModelRegistry {
    cache_dir: PathBuf,
    confidence_threshold: f32,
}

impl ModelRegistry {
    pub fn new(confidence_threshold: f32) -> Self {
        let cache_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("kong")
            .join("models");

        Self {
            cache_dir,
            confidence_threshold,
        }
    }

    pub fn with_cache_dir(cache_dir: PathBuf, confidence_threshold: f32) -> Self {
        Self {
            cache_dir,
            confidence_threshold,
        }
    }

    /// Check if a model is available locally for the given architecture.
    pub fn has_model(&self, arch: Arch) -> bool {
        self.weights_path(arch).exists() && self.config_path(arch).exists()
    }

    /// List all locally available architectures.
    pub fn available_architectures(&self) -> Vec<Arch> {
        let all = [
            Arch::X86_64,
            Arch::X86,
            Arch::Aarch64,
            Arch::Arm,
            Arch::Mips,
            Arch::Mips64,
            Arch::Riscv64,
            Arch::PowerPc,
        ];
        all.into_iter().filter(|a| self.has_model(*a)).collect()
    }

    /// Load a detector for the given architecture.
    /// Returns None if no model is available (graceful degradation).
    pub fn load_detector(&self, arch: Arch, device: &Device) -> Result<Option<XdaDetector>, Box<dyn std::error::Error>> {
        if !self.has_model(arch) {
            return Ok(None);
        }

        let config = XdaConfig::from_json(&self.config_path(arch))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.weights_path(arch)],
                DType::F32,
                device,
            )?
        };
        let model = XdaModel::load(&config, vb)?;
        let detector = XdaDetector::new(model, device.clone(), self.confidence_threshold);

        Ok(Some(detector))
    }

    /// Install a model from a local safetensors + config.json pair.
    pub fn install(&self, arch: Arch, weights_src: &Path, config_src: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.cache_dir)?;
        std::fs::copy(weights_src, self.weights_path(arch))?;
        std::fs::copy(config_src, self.config_path(arch))?;
        Ok(())
    }

    fn weights_path(&self, arch: Arch) -> PathBuf {
        self.cache_dir.join(format!("xda_{}.safetensors", arch.model_name()))
    }

    fn config_path(&self, arch: Arch) -> PathBuf {
        self.cache_dir.join(format!("xda_{}_config.json", arch.model_name()))
    }
}
```

- [ ] **Step 2: Uncomment registry in lib.rs, wire up all modules**

```rust
// kong-rs/crates/kong-ml/src/lib.rs
pub mod config;
pub mod tokenizer;
pub mod model;
pub mod detector;
pub mod registry;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd kong-rs && cargo check -p kong-ml`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add kong-rs/crates/kong-ml/src/registry.rs kong-rs/crates/kong-ml/src/lib.rs
git commit -m "feat(kong-ml): model registry with per-arch cache and lazy loading"
```

---

### Task 15: End-to-End Integration Test

**Files:**
- Create: `kong-rs/tests/xda_integration.rs`

This test validates the full pipeline: tokenize → model → predict. It requires a trained model to exist, so it's gated behind an env var.

- [ ] **Step 1: Write the integration test**

```rust
// kong-rs/tests/xda_integration.rs

/// End-to-end integration test. Requires a trained model.
/// Run with: XDA_MODEL_DIR=/path/to/models cargo test --test xda_integration
#[test]
fn test_full_pipeline() {
    let model_dir = match std::env::var("XDA_MODEL_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => {
            eprintln!("Skipping integration test: XDA_MODEL_DIR not set");
            return;
        }
    };

    let device = candle_core::Device::Cpu;
    let registry = kong_ml::registry::ModelRegistry::with_cache_dir(model_dir, 0.85);

    // Check x86_64 model exists
    let arch = kong_types::binary::Arch::X86_64;
    assert!(registry.has_model(arch), "x86_64 model not found in model dir");

    // Load detector
    let detector = registry
        .load_detector(arch, &device)
        .expect("Failed to load model")
        .expect("Model should exist");

    // Create a fake ambiguous region with x86_64 function prologue bytes
    let prologue = vec![
        0x55,                               // push rbp
        0x48, 0x89, 0xe5,                   // mov rbp, rsp
        0x48, 0x83, 0xec, 0x10,             // sub rsp, 0x10
        0x89, 0x7d, 0xfc,                   // mov [rbp-4], edi
        0x8b, 0x45, 0xfc,                   // mov eax, [rbp-4]
        0x83, 0xc0, 0x01,                   // add eax, 1
        0xc9,                               // leave
        0xc3,                               // ret
        // padding / gap
        0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
        // second function
        0x55,                               // push rbp
        0x48, 0x89, 0xe5,                   // mov rbp, rsp
        0x89, 0x7d, 0xfc,                   // mov [rbp-4], edi
        0x8b, 0x45, 0xfc,                   // mov eax, [rbp-4]
        0x0f, 0xaf, 0xc0,                   // imul eax, eax
        0x5d,                               // pop rbp
        0xc3,                               // ret
    ];

    let region = kong_types::function::AmbiguousRegion {
        start: 0x401000,
        bytes: prologue,
        reason: kong_types::function::AmbiguityReason::GapRegion,
    };

    let boundaries = detector.predict(&[region]).expect("Inference failed");

    // With a well-trained model, we'd expect it to find function starts
    // at offsets corresponding to the two push rbp instructions.
    // For now, just verify inference runs without error and produces output.
    println!("Found {} function boundaries:", boundaries.len());
    for b in &boundaries {
        println!(
            "  0x{:x} - 0x{:x} (confidence: {:.2}, source: {:?})",
            b.start, b.end, b.confidence, b.source
        );
    }
}

/// Smoke test that runs without a model — verifies graceful degradation.
#[test]
fn test_no_model_graceful() {
    let tmp = std::env::temp_dir().join("kong_test_empty_models");
    let _ = std::fs::create_dir_all(&tmp);

    let registry = kong_ml::registry::ModelRegistry::with_cache_dir(tmp, 0.85);
    let device = candle_core::Device::Cpu;

    let detector = registry
        .load_detector(kong_types::binary::Arch::X86_64, &device)
        .expect("Should not error");

    assert!(detector.is_none(), "Should return None when no model exists");
}
```

- [ ] **Step 2: Verify the no-model test passes**

Run: `cd kong-rs && cargo test --test xda_integration test_no_model_graceful`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add kong-rs/tests/xda_integration.rs
git commit -m "feat(kong-ml): end-to-end integration test for XDA inference pipeline"
```

---

## Part C: Training Your First Model (x86_64)

### Task 16: Train x86_64 Model End-to-End

This task uses the scripts from Part A to actually train and export a model.

- [ ] **Step 1: Build x86_64 corpus**

```bash
cd tools/xda
chmod +x scripts/build_corpus.sh
./scripts/build_corpus.sh x86_64 ./corpus/x86_64
```

Expected: ~15 binaries (3 source files x 5 optimization levels) in `corpus/x86_64/`.

- [ ] **Step 2: Build training dataset**

```bash
cd tools/xda
uv run python -c "
from src.build_dataset import build_from_corpus_dir
from pathlib import Path
stats = build_from_corpus_dir(Path('./corpus/x86_64'), Path('./data/x86_64'))
print(f'Binaries: {stats[\"binaries\"]}')
print(f'Functions: {stats[\"functions\"]}')
print(f'Train chunks: {stats[\"train_chunks\"]}')
print(f'Val chunks: {stats[\"val_chunks\"]}')
"
```

Expected: `data/x86_64/train.json`, `data/x86_64/val.json`, `data/x86_64/stats.json` created. Small corpus — maybe 50-200 chunks. Good enough for a smoke test.

- [ ] **Step 3: Fine-tune (skip pre-training for x86_64 — use from scratch for now)**

```bash
cd tools/xda
uv run python -m src.finetune --config configs/finetune.yaml --data ./data/x86_64
```

Expected: Training runs for 5 epochs, prints per-class F1 scores. On this small corpus, results won't be great — that's fine, this validates the pipeline.

- [ ] **Step 4: Export to safetensors**

```bash
cd tools/xda
uv run python -m src.export \
    --checkpoint checkpoints/finetune/best.pt \
    --config checkpoints/finetune/config.json \
    --output ../../kong-rs/models/ \
    --arch x86_64
```

Expected: `kong-rs/models/xda_x86_64.safetensors` and `kong-rs/models/xda_x86_64_config.json` created.

- [ ] **Step 5: Run Rust integration test against trained model**

```bash
cd kong-rs
XDA_MODEL_DIR=./models cargo test --test xda_integration test_full_pipeline -- --nocapture
```

Expected: Test passes. Prints function boundaries found (may not be accurate with tiny training set — validates the pipeline works end-to-end).

- [ ] **Step 6: Commit**

```bash
git add kong-rs/models/.gitkeep
echo "*.safetensors" >> kong-rs/.gitignore
echo "*.pt" >> tools/xda/.gitignore
echo "corpus/" >> tools/xda/.gitignore
echo "data/" >> tools/xda/.gitignore
echo "checkpoints/" >> tools/xda/.gitignore
git add kong-rs/.gitignore tools/xda/.gitignore
git commit -m "feat: end-to-end XDA pipeline validated — train in Python, infer in Rust"
```

---

## Part D: Scale Up for Production Quality

### Task 17: Build Large Corpus with Debian Packages (AArch64)

This repeats the pattern for a new architecture.

- [ ] **Step 1: Build and run the Debian corpus Docker image**

```bash
cd tools/xda
docker build --platform linux/arm64 -t xda-corpus-aarch64 -f scripts/Dockerfile.corpus .
docker run --platform linux/arm64 -v $(pwd)/corpus/aarch64_debian:/out xda-corpus-aarch64
```

Expected: `corpus/aarch64_debian/pairs/` with 50-200+ binary/debug pairs.

- [ ] **Step 2: Also cross-compile custom corpus for AArch64**

```bash
# Requires: brew install aarch64-linux-gnu-gcc (or via apt on Linux)
cd tools/xda
./scripts/build_corpus.sh aarch64 ./corpus/aarch64_custom
```

- [ ] **Step 3: Build dataset from both sources**

```bash
cd tools/xda
uv run python -c "
from src.build_dataset import build_from_pairs_dir, build_from_corpus_dir
from pathlib import Path
import json

# Merge both corpus sources
all_chunks = []

# Debian packages
stats1 = build_from_pairs_dir(Path('./corpus/aarch64_debian/pairs'), Path('./data/aarch64_tmp1'))
t1 = json.loads(Path('./data/aarch64_tmp1/train.json').read_text())
v1 = json.loads(Path('./data/aarch64_tmp1/val.json').read_text())

# Cross-compiled
stats2 = build_from_corpus_dir(Path('./corpus/aarch64_custom'), Path('./data/aarch64_tmp2'))
t2 = json.loads(Path('./data/aarch64_tmp2/train.json').read_text())
v2 = json.loads(Path('./data/aarch64_tmp2/val.json').read_text())

# Merge
import random
train = t1 + t2; random.shuffle(train)
val = v1 + v2; random.shuffle(val)

out = Path('./data/aarch64'); out.mkdir(exist_ok=True)
(out / 'train.json').write_text(json.dumps(train))
(out / 'val.json').write_text(json.dumps(val))
print(f'Total train: {len(train)}, val: {len(val)}')
"
```

- [ ] **Step 4: Pre-train on AArch64 bytes (optional but recommended)**

```bash
cd tools/xda
uv run python -m src.pretrain --config configs/pretrain.yaml --data ./data/aarch64
```

Expected: Runs for 10 epochs. May take several hours depending on corpus size and hardware.

- [ ] **Step 5: Fine-tune on AArch64**

Edit `configs/finetune.yaml` to point `pretrained_checkpoint` at the pre-trained checkpoint:
```yaml
pretrained_checkpoint: "checkpoints/pretrain/pretrain_final.pt"
```

```bash
cd tools/xda
uv run python -m src.finetune --config configs/finetune.yaml --data ./data/aarch64
```

- [ ] **Step 6: Export AArch64 model**

```bash
cd tools/xda
uv run python -m src.export \
    --checkpoint checkpoints/finetune/best.pt \
    --config checkpoints/finetune/config.json \
    --output ../../kong-rs/models/ \
    --arch aarch64
```

- [ ] **Step 7: Verify in Rust**

```bash
cd kong-rs
XDA_MODEL_DIR=./models cargo test --test xda_integration -- --nocapture
```

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(xda): trained and validated AArch64 function boundary model"
```

---

## Summary: What Each Task Produces

| Task | Deliverable |
|------|-------------|
| 1 | Python project + byte tokenizer |
| 2 | DWARF ground truth extractor |
| 3 | Chunked dataset pipeline |
| 4 | XDA model definition (Python) |
| 5 | Corpus build scripts (Debian + cross-compile) |
| 6 | End-to-end dataset builder |
| 7 | Pre-training script (masked byte modeling) |
| 8 | Fine-tuning script (3-class boundary detection) |
| 9 | Safetensors export for Rust |
| 10 | Cargo workspace + kong-types crate |
| 11 | Rust byte tokenizer + config loading |
| 12 | XDA model in candle (BERT + MLP) |
| 13 | Sliding window inference + batch prediction |
| 14 | Per-arch model registry with caching |
| 15 | End-to-end Rust integration test |
| 16 | First trained model (x86_64, small corpus) |
| 17 | Production AArch64 model (large corpus) |

**Tasks 1-9** (Python) and **Tasks 10-15** (Rust) are independent workstreams that can be parallelized. Task 16 bridges them. Task 17 is the repeatable pattern for new architectures.
