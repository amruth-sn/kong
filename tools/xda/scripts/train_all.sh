#!/usr/bin/env bash
# tools/xda/scripts/train_all.sh
# Production training: builds Debian corpora, datasets, trains + exports all architectures.
# Run on a Linux GPU VM with CUDA and Docker.
# Usage: ./scripts/train_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XDA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$XDA_DIR"

# arch name -> Docker platform
declare -A PLATFORMS=(
    [x86_64]=linux/amd64
    [aarch64]=linux/arm64
    [arm]=linux/arm/v7
    [mips]=linux/mipsel
)

# ---------- 1. System setup ----------
echo "=== [1/5] System setup ==="

if ! command -v docker &>/dev/null; then
    echo "Docker not found. Install: https://docs.docker.com/engine/install/"
    exit 1
fi

# QEMU for cross-platform Docker builds
docker run --privileged --rm tonistiigi/binfmt --install all 2>/dev/null || true

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# ---------- 2. Build Debian corpora ----------
echo "=== [2/5] Building Debian corpora ==="

for arch in "${!PLATFORMS[@]}"; do
    platform="${PLATFORMS[$arch]}"
    image="xda-corpus-${arch}"
    corpus_dir="./corpus/${arch}"

    if [ -d "$corpus_dir/pairs" ] && [ "$(ls -A "$corpus_dir/pairs" 2>/dev/null)" ]; then
        echo "--- $arch: corpus already exists, skipping ---"
        continue
    fi

    echo "--- $arch: building Docker image ($platform) ---"
    docker build --platform "$platform" -t "$image" -f scripts/Dockerfile.corpus .

    echo "--- $arch: extracting binary/debug pairs ---"
    mkdir -p "$corpus_dir"
    docker run --platform "$platform" -v "$(pwd)/$corpus_dir:/out" "$image"

    pair_count=$(ls -1 "$corpus_dir/pairs" 2>/dev/null | wc -l | tr -d ' ')
    echo "--- $arch: extracted $pair_count pairs ---"
done

# ---------- 3. Build datasets ----------
echo "=== [3/5] Building datasets ==="

for arch in "${!PLATFORMS[@]}"; do
    if [ -f "./data/${arch}/train.json" ]; then
        echo "--- $arch: dataset already exists, skipping ---"
        continue
    fi

    echo "--- $arch ---"
    uv run python -c "
from src.build_dataset import build_from_pairs_dir
from pathlib import Path
stats = build_from_pairs_dir(Path('./corpus/${arch}/pairs'), Path('./data/${arch}'))
print(f'  {stats[\"binaries\"]} binaries, {stats[\"functions\"]} functions, {stats[\"train_chunks\"]}+{stats[\"val_chunks\"]} chunks')
if stats[\"errors\"]:
    print(f'  {len(stats[\"errors\"])} errors')
"
done

# ---------- 4. Train ----------
echo "=== [4/5] Training models ==="

for arch in "${!PLATFORMS[@]}"; do
    ckpt_dir="checkpoints/finetune_${arch}"

    if [ -f "$ckpt_dir/best.pt" ]; then
        echo "--- $arch: checkpoint already exists, skipping ---"
        continue
    fi

    echo "--- $arch ---"

    # Point finetune.yaml's checkpoint_dir at per-arch directory
    tmp_config=$(mktemp /tmp/finetune_XXXX.yaml)
    sed "s|checkpoint_dir:.*|checkpoint_dir: ${ckpt_dir}|" configs/finetune.yaml > "$tmp_config"

    uv run python -m src.finetune --config "$tmp_config" --data "./data/${arch}" --arch "$arch"
    rm -f "$tmp_config"
done

# ---------- 5. Export ----------
echo "=== [5/5] Exporting models ==="

mkdir -p models
for arch in "${!PLATFORMS[@]}"; do
    ckpt_dir="checkpoints/finetune_${arch}"

    if [ -f "models/xda_${arch}.safetensors" ]; then
        echo "--- $arch: already exported, skipping ---"
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
echo "========================================="
echo "  Training complete. Model artifacts:"
echo "========================================="
ls -lh models/
echo ""
echo "Tensorboard logs: tensorboard --logdir runs/"
echo "Copy models: scp models/* <local>:~/Development/kong/tools/xda/models/"
