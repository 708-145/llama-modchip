#!/usr/bin/env bash
# build-dsv4-engine.sh — build the llama.cpp engine for DeepSeek-V4-Flash-0731-Fractal on Apple Silicon.
#
# patches/dsv4-64gb-cpu.patch only applies at llama.cpp d2f83055 (2026-08-10): its 18 pre-image blobs
# match between d2f83055 and dd1ea524 and nowhere else, and on current master 8 files fail to apply.
# This script checks out that commit as a worktree, applies the patch, copies the RouteCrack headers
# that the patched common/ngram-map.cpp includes, then applies patches/apple-silicon/*.patch in order.
#
# Environment:
#   LLAMA_SRC  existing llama.cpp clone (default: workspace/llama.cpp-src, cloned if missing)
#   DEST       worktree to create (default: workspace/llama.cpp-dsv4; must not exist)
#   JOBS       build parallelism (default: number of CPUs)
#   BUILD      0 = prepare the source tree only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BASE=d2f83055d6e3b379b5d34c4837122a918cf402c2
LLAMA_SRC="${LLAMA_SRC:-$REPO/workspace/llama.cpp-src}"
DEST="${DEST:-$REPO/workspace/llama.cpp-dsv4}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [[ ! -d "$LLAMA_SRC/.git" && ! -f "$LLAMA_SRC/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
fi
if ! git -C "$LLAMA_SRC" cat-file -e "${BASE}^{commit}" 2>/dev/null; then
    git -C "$LLAMA_SRC" fetch origin
fi
if [[ -e "$DEST" ]]; then
    echo "$DEST already exists; remove it (git -C $LLAMA_SRC worktree remove $DEST) or set DEST" >&2
    exit 1
fi

git -C "$LLAMA_SRC" worktree add --detach "$DEST" "$BASE"
cd "$DEST"

git apply "$REPO/patches/dsv4-64gb-cpu.patch"
cp "$REPO/speculative/cpp/rc_predictor.h" "$REPO/speculative/cpp/rc_stockfish.h" "$REPO/speculative/cpp/rc_trie_v2.h" common/

shopt -s nullglob
for p in "$REPO"/patches/apple-silicon/*.patch; do
    echo "[build-dsv4-engine] applying $(basename "$p")" >&2
    git apply "$p"
done

if [[ "${BUILD:-1}" == 0 ]]; then
    echo "[build-dsv4-engine] source tree ready in $DEST (BUILD=0)" >&2
    exit 0
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build -j "$JOBS" --target llama-server llama-bench llama-cli test-backend-ops test-quantize-fns

echo "[build-dsv4-engine] built $DEST/build/bin" >&2
