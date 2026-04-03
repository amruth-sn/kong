#!/usr/bin/env bash
# tools/xda/scripts/export_all.sh
# Export fine-tuned checkpoints to safetensors for Rust/candle inference.
# Run from tools/xda/ on a GPU VM (or locally).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XDA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$XDA_DIR"

ARCHS=(x86_64 aarch64 arm riscv64)

mkdir -p models

echo "=== Exporting models ==="

for arch in "${ARCHS[@]}"; do
    ckpt_dir="checkpoints/finetune_${arch}"

    if [ -f "models/xda_${arch}.safetensors" ]; then
        echo "--- $arch: already exported, skipping ---"
        continue
    fi

    if [ ! -f "$ckpt_dir/best.pt" ] || [ ! -f "$ckpt_dir/config.json" ]; then
        echo "--- $arch: no checkpoint at $ckpt_dir, skipping ---"
        continue
    fi

    echo "--- $arch ---"
    uv run python -m src.export \
        --checkpoint "$ckpt_dir/best.pt" \
        --config "$ckpt_dir/config.json" \
        --output ./models/ \
        --arch "$arch"
done

echo ""
echo "=== Export complete ==="
echo "Models:"
ls -lh models/*.safetensors 2>/dev/null || echo "  (none)"
echo ""
echo "Copy to local machine:"
echo "  scp -P <port> root@<host>:$(pwd)/models/* ~/Development/kong/tools/xda/models/"
