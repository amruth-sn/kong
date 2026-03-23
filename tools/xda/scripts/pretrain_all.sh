#!/usr/bin/env bash
# tools/xda/scripts/pretrain_all.sh
# Masked byte modeling pre-training for non-x86 architectures.
# Uses the training data (only reads bytes, ignores labels).
# Run from tools/xda/ on a GPU VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XDA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$XDA_DIR"

ARCHS=(aarch64 arm riscv64)

echo "=== Pre-training: ${ARCHS[*]} ==="

for arch in "${ARCHS[@]}"; do
    ckpt_dir="checkpoints/pretrain_${arch}"

    if [ -f "$ckpt_dir/pretrain_final.pt" ]; then
        echo "--- $arch: pretrain checkpoint exists, skipping ---"
        continue
    fi

    if [ ! -f "data/${arch}/train.json" ]; then
        echo "--- $arch: no training data at data/${arch}/train.json, skipping ---"
        continue
    fi

    echo "--- $arch: pre-training ---"
    tmp=$(mktemp /tmp/pt_XXXX.yaml)
    sed "s|checkpoint_dir:.*|checkpoint_dir: ${ckpt_dir}|" configs/pretrain.yaml > "$tmp"

    uv run python -m src.pretrain --config "$tmp" --data "./data/${arch}"
    rm -f "$tmp"
done

echo ""
echo "=== Pre-training complete ==="
for arch in "${ARCHS[@]}"; do
    ckpt="checkpoints/pretrain_${arch}/pretrain_final.pt"
    if [ -f "$ckpt" ]; then
        echo "  $arch: $ckpt"
    else
        echo "  $arch: MISSING"
    fi
done
