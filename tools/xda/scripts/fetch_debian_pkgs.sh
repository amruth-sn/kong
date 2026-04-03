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
