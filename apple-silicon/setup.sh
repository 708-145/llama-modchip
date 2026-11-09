#!/usr/bin/env bash
# apple-silicon/setup.sh — Build patched llama-modchip (Metal + ARM64 NEON SDOT Fractal-64) on macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-${REPO_ROOT}/llama.cpp}"

echo "[*] Building llama-modchip on Apple Silicon (Metal=${GGML_METAL:-ON}, BLAS=${GGML_BLAS:-ON}, ARM64 NEON SDOT + 1x{8,4,3,2} Fractal-64 tiling)..."
"${REPO_ROOT}/scripts/build_zen5_llama.sh" "$LLAMA_DIR"

echo ""
echo "[+] Apple Silicon setup complete!"
echo "    Binary   : ${BUILD_DIR:-${LLAMA_DIR}/build}/bin/llama-server"
echo "    Metal    : MODEL=/path/to/model.gguf ./apple-silicon/run-server.sh"
echo "    CPU-only : BUILD_DIR=${LLAMA_DIR}/build-cpu GGML_METAL=OFF GGML_BLAS=OFF ./scripts/build_zen5_llama.sh"
