from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .dataset import chunk_labeled_bytes
from .extract_labels import extract_function_boundaries, generate_byte_labels


def _serialize_chunks(chunks: list[dict], path: Path) -> None:
    serialized = [{"bytes": list(c["bytes"]), "labels": c["labels"]} for c in chunks]
    path.write_text(json.dumps(serialized))


def _split_and_save(
    all_chunks: list[dict],
    output_path: Path,
    val_split: float,
    stats: dict,
) -> dict:
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * (1 - val_split))
    train_chunks = all_chunks[:split_idx]
    val_chunks = all_chunks[split_idx:]

    output_path.mkdir(parents=True, exist_ok=True)
    _serialize_chunks(train_chunks, output_path / "train.json")
    _serialize_chunks(val_chunks, output_path / "val.json")

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
    Uses debug files for DWARF labels and .text bytes.
    """
    all_chunks: list[dict] = []
    stats: dict = {"binaries": 0, "functions": 0, "chunks": 0, "errors": []}

    debug_files = sorted(corpus_dir.glob("*_debug"))
    if not debug_files:
        debug_files = sorted(
            f for f in corpus_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and "stripped" not in f.name
        )

    for debug_file in tqdm(debug_files, desc="Processing corpus"):
        if debug_file.is_dir():
            continue

        try:
            boundaries = extract_function_boundaries(str(debug_file))
            if not boundaries:
                continue

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

    return _split_and_save(all_chunks, output_path, val_split, stats)


def build_from_pairs_dir(
    pairs_dir: Path,
    output_path: Path,
    window_size: int = 512,
    stride: int = 256,
    val_split: float = 0.1,
) -> dict:
    """Process binary/debug pairs from Debian package extraction.

    Expects pairs_dir to contain subdirectories, each with 'binary' and 'debug' files.
    """
    all_chunks: list[dict] = []
    stats: dict = {"binaries": 0, "functions": 0, "chunks": 0, "errors": []}

    for pair_dir in tqdm(sorted(pairs_dir.iterdir()), desc="Processing binaries"):
        if not pair_dir.is_dir():
            continue

        debug_file = pair_dir / "debug"
        binary_file = pair_dir / "binary"

        if not debug_file.exists():
            continue

        label_source = str(debug_file)
        byte_source = str(binary_file) if binary_file.exists() else label_source

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

    return _split_and_save(all_chunks, output_path, val_split, stats)
