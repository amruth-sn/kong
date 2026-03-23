#!/usr/bin/env bash
# tools/xda/scripts/finetune_all.sh
# Fine-tune all architectures for function boundary detection.
# x86_64: from CUMLSec checkpoint or from scratch.
# Others: from pretrain checkpoint (run pretrain_all.sh first).
# Run from tools/xda/ on a GPU VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XDA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$XDA_DIR"

ARCHS=(x86_64 aarch64 arm riscv64)

# Set this to the path of the CUMLSec pretrained checkpoint for x86_64.
# Leave empty to train x86_64 from scratch.
CUMLSEC_CHECKPOINT="${CUMLSEC_CHECKPOINT:-}"

echo "=== Fine-tuning: ${ARCHS[*]} ==="
if [ -n "$CUMLSEC_CHECKPOINT" ]; then
    echo "  x86_64 pretrained: $CUMLSEC_CHECKPOINT"
else
    echo "  x86_64: from scratch (set CUMLSEC_CHECKPOINT to use pretrained)"
fi

for arch in "${ARCHS[@]}"; do
    ckpt_dir="checkpoints/finetune_${arch}"

    if [ -f "$ckpt_dir/best.pt" ]; then
        echo "--- $arch: checkpoint already exists, skipping ---"
        continue
    fi

    if [ ! -f "data/${arch}/train.json" ]; then
        echo "--- $arch: no training data at data/${arch}/train.json, skipping ---"
        continue
    fi

    echo "--- $arch ---"
    tmp=$(mktemp /tmp/ft_XXXX.yaml)
    sed "s|checkpoint_dir:.*|checkpoint_dir: ${ckpt_dir}|" configs/finetune.yaml > "$tmp"

    if [ "$arch" = "x86_64" ]; then
        if [ -n "$CUMLSEC_CHECKPOINT" ]; then
            sed -i "s|pretrained_checkpoint:.*|pretrained_checkpoint: ${CUMLSEC_CHECKPOINT}|" "$tmp"
        fi
    else
        pretrain_ckpt="checkpoints/pretrain_${arch}/pretrain_final.pt"
        if [ -f "$pretrain_ckpt" ]; then
            sed -i "s|pretrained_checkpoint:.*|pretrained_checkpoint: ${pretrain_ckpt}|" "$tmp"
        else
            echo "  WARNING: no pretrain checkpoint at $pretrain_ckpt, training from scratch"
        fi
    fi

    uv run python -m src.finetune --config "$tmp" --data "./data/${arch}" --arch "$arch"
    rm -f "$tmp"
done

echo ""
echo "=== Fine-tuning complete ==="
for arch in "${ARCHS[@]}"; do
    ckpt="checkpoints/finetune_${arch}/best.pt"
    if [ -f "$ckpt" ]; then
        echo "  $arch: $ckpt"
    else
        echo "  $arch: MISSING"
    fi
done
