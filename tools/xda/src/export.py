from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .model import XdaConfig, XdaModel


def export(checkpoint_path: str, config_path: str, output_dir: str, arch: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config_dict = json.load(f)
    config = XdaConfig(**config_dict)

    model = XdaModel(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    flat_state = {}
    for key, tensor in model.state_dict().items():
        flat_state[key] = tensor.float()

    weights_path = out / f"xda_{arch}.safetensors"
    save_file(flat_state, str(weights_path))

    config_out = out / f"xda_{arch}_config.json"
    config_out.write_text(json.dumps(config_dict, indent=2))

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
