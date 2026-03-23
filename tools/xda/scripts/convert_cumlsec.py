#!/usr/bin/env python3
"""Convert a CUMLSec/FairSeq XDA checkpoint to our HuggingFace BertModel format.

Usage:
    uv run python scripts/convert_cumlsec.py --input checkpoint.pt --output checkpoints/cumlsec_x86.pt

FairSeq keys (decoder.sentence_encoder.*) are remapped to HuggingFace keys
(encoder.*) so finetune.py can load them via `pretrained_checkpoint`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

LAYER_KEY_MAP = {
    "self_attn.q_proj.weight": "attention.self.query.weight",
    "self_attn.q_proj.bias": "attention.self.query.bias",
    "self_attn.k_proj.weight": "attention.self.key.weight",
    "self_attn.k_proj.bias": "attention.self.key.bias",
    "self_attn.v_proj.weight": "attention.self.value.weight",
    "self_attn.v_proj.bias": "attention.self.value.bias",
    "self_attn.out_proj.weight": "attention.output.dense.weight",
    "self_attn.out_proj.bias": "attention.output.dense.bias",
    "self_attn_layer_norm.weight": "attention.output.LayerNorm.weight",
    "self_attn_layer_norm.bias": "attention.output.LayerNorm.bias",
    "fc1.weight": "intermediate.dense.weight",
    "fc1.bias": "intermediate.dense.bias",
    "fc2.weight": "output.dense.weight",
    "fc2.bias": "output.dense.bias",
    "final_layer_norm.weight": "output.LayerNorm.weight",
    "final_layer_norm.bias": "output.LayerNorm.bias",
}

FAIRSEQ_PREFIX = "decoder.sentence_encoder."


def map_key(fairseq_key: str) -> str | None:
    if not fairseq_key.startswith(FAIRSEQ_PREFIX):
        return None

    rest = fairseq_key[len(FAIRSEQ_PREFIX):]

    if rest == "embed_tokens.weight":
        return "encoder.embeddings.word_embeddings.weight"
    if rest == "embed_positions.weight":
        return "encoder.embeddings.position_embeddings.weight"
    if rest.startswith("emb_layer_norm."):
        suffix = rest[len("emb_layer_norm."):]
        return f"encoder.embeddings.LayerNorm.{suffix}"

    layer_match = re.match(r"layers\.(\d+)\.(.*)", rest)
    if not layer_match:
        return None

    layer_idx = layer_match.group(1)
    layer_rest = layer_match.group(2)

    if layer_rest in LAYER_KEY_MAP:
        return f"encoder.encoder.layer.{layer_idx}.{LAYER_KEY_MAP[layer_rest]}"

    return None


def convert(input_path: str, output_path: str) -> None:
    print(f"Loading FairSeq checkpoint: {input_path}")
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    fairseq_state = raw["model"]

    converted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in fairseq_state.items():
        new_key = map_key(key)
        if new_key is not None:
            converted[new_key] = tensor
        else:
            skipped.append(key)

    pos_key = "encoder.embeddings.position_embeddings.weight"
    if pos_key in converted:
        pos_emb = converted[pos_key]
        target_size = 514
        if pos_emb.shape[0] != target_size:
            print(f"Position embeddings: {pos_emb.shape[0]} → {target_size} (trimming FairSeq padding offset)")
            # FairSeq prepends padding positions — take the last target_size rows
            if pos_emb.shape[0] > target_size:
                converted[pos_key] = pos_emb[pos_emb.shape[0] - target_size:]
            else:
                pad = torch.zeros(target_size - pos_emb.shape[0], pos_emb.shape[1])
                converted[pos_key] = torch.cat([pos_emb, pad])

    emb_key = "encoder.embeddings.word_embeddings.weight"
    if emb_key in converted:
        emb = converted[emb_key]
        target_vocab = 261
        if emb.shape[0] != target_vocab:
            print(f"Vocab embeddings: {emb.shape[0]} → {target_vocab}")
            if emb.shape[0] > target_vocab:
                converted[emb_key] = emb[:target_vocab]
            else:
                pad = torch.zeros(target_vocab - emb.shape[0], emb.shape[1])
                torch.nn.init.normal_(pad, std=0.02)
                converted[emb_key] = torch.cat([emb, pad])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)

    print(f"\nConverted: {len(converted)} tensors")
    print(f"Skipped:   {len(skipped)} tensors (classification head, etc.)")

    if skipped:
        print("\nSkipped keys:")
        for k in skipped:
            print(f"  {k}")

    print(f"\nSaved to: {output_path}")
    print("\nConverted keys:")
    for k in sorted(converted.keys()):
        print(f"  {k}: {list(converted[k].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CUMLSec/FairSeq checkpoint")
    parser.add_argument("--output", required=True, help="Output path for converted checkpoint")
    args = parser.parse_args()
    convert(args.input, args.output)
