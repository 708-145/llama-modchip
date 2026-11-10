#!/usr/bin/env bash
# build_zen5_llama.sh — Build llama.cpp with 256K Hybrid SWA + Fractal-64 (AVX-512 / AVX2 / ARM64 SDOT / Metal) + RouteCrack Engine
set -euo pipefail

LLAMA_DIR="${1:-./llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_DIR}/build}"
LLAMA_COMMIT="d2f83055d6e3b379b5d34c4837122a918cf402c2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/zen5-nemotron-256k-optimizations.patch"
# patches stacked on top of PATCH_FILE, applied in file-name order
SERIES_DIR="${REPO_ROOT}/patches/nemotron-apple-silicon"
SPEC_CPP_DIR="${REPO_ROOT}/speculative/cpp"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"

if [[ "$OS_NAME" == "Darwin" ]]; then
    if [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
        echo "[!] WARNING: Running inside a Rosetta x86_64 shell on Apple Silicon! Re-run in a native arm64 terminal ('arch -arm64 bash') for NEON SDOT + Metal performance." >&2
    fi
fi

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "[*] Cloning llama.cpp into $LLAMA_DIR..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
else
    CUR_COMMIT="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$CUR_COMMIT" != "$LLAMA_COMMIT" ]]; then
        echo "[*] Existing $LLAMA_DIR is at ${CUR_COMMIT:-unknown}; checking out pinned commit $LLAMA_COMMIT..."
        git -C "$LLAMA_DIR" fetch --depth=1 origin "$LLAMA_COMMIT"
        git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
    fi
fi

# Patches stacked on top of PATCH_FILE (Q2_0 Metal decode kernel, gguf-py block size, Q2_0 tests). They are applied on
# macOS by default and left out elsewhere, so a Linux/zen5 source tree is the same as before; APPLE_SILICON_PATCHES=1
# applies them on any OS, =0 never.
APPLE_SILICON_PATCHES="${APPLE_SILICON_PATCHES:-auto}"
if [[ "$APPLE_SILICON_PATCHES" == "auto" ]]; then
    if [[ "$OS_NAME" == "Darwin" ]]; then APPLE_SILICON_PATCHES=1; else APPLE_SILICON_PATCHES=0; fi
elif [[ "$APPLE_SILICON_PATCHES" != "0" && "$APPLE_SILICON_PATCHES" != "1" ]]; then
    echo "[!] ERROR: Invalid APPLE_SILICON_PATCHES='$APPLE_SILICON_PATCHES'. Use 'auto', '0', or '1'." >&2
    exit 1
fi
SERIES_PATCHES=()
if [[ "$APPLE_SILICON_PATCHES" == "1" ]]; then
    shopt -s nullglob
    SERIES_PATCHES=("${SERIES_DIR}"/[0-9][0-9][0-9][0-9]-*.patch)
    shopt -u nullglob
fi

# The stacked patches change lines the base patch added, so once they are applied PATCH_FILE can no longer be checked on
# its own. Count the series patches that reverse-apply: all of them means PATCH_FILE and the series are in the tree; some
# of them means an interrupted run: the script stops, or resets the tree to the pinned commit if RESET_LLAMA_TREE=1.
SERIES_APPLIED=0
if [[ ${#SERIES_PATCHES[@]} -gt 0 ]]; then
    N_SERIES_IN_TREE=0
    for p in "${SERIES_PATCHES[@]}"; do
        if git -C "$LLAMA_DIR" apply --reverse --check "$p" 2>/dev/null; then
            N_SERIES_IN_TREE=$((N_SERIES_IN_TREE + 1))
        fi
    done
    if [[ "$N_SERIES_IN_TREE" -eq "${#SERIES_PATCHES[@]}" ]]; then
        SERIES_APPLIED=1
    elif [[ "$N_SERIES_IN_TREE" -gt 0 ]]; then
        # A partly applied series must not be built, and the script cannot tell the patches' changes from the user's own
        # edits, so it only resets the tree when asked to.
        if [[ "${RESET_LLAMA_TREE:-0}" == "1" ]]; then
            echo "[*] Only $N_SERIES_IN_TREE of ${#SERIES_PATCHES[@]} patches from patches/nemotron-apple-silicon/ are in $LLAMA_DIR; RESET_LLAMA_TREE=1: resetting its working tree to clean $LLAMA_COMMIT (uncommitted changes are discarded)..."
            git -C "$LLAMA_DIR" checkout HEAD -- .
            while read -r created_file; do
                [[ -n "$created_file" ]] && rm -f "$LLAMA_DIR/$created_file"
            done < <(git -C "$REPO_ROOT" apply --summary "$PATCH_FILE" 2>/dev/null | awk '/create mode/{print $4}')
        else
            echo "[!] ERROR: Only $N_SERIES_IN_TREE of ${#SERIES_PATCHES[@]} patches from patches/nemotron-apple-silicon/ are in $LLAMA_DIR (an interrupted run?)." >&2
            echo "[!] Nothing was changed. To discard the uncommitted changes in $LLAMA_DIR and apply everything again, run:" >&2
            echo "[!]   RESET_LLAMA_TREE=1 $0 $LLAMA_DIR     (or: git -C $LLAMA_DIR checkout -- .  and run again)" >&2
            exit 1
        fi
    fi
fi

echo "[*] Applying 256K Hybrid SWA + Fractal-64 SIMD/Metal + L3 NTA + 2MB HugePages patch..."
if [[ "$SERIES_APPLIED" == "1" ]]; then
    echo "[i] Patch and patches/nemotron-apple-silicon/ already applied cleanly in $LLAMA_DIR."
elif git -C "$LLAMA_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "[i] Patch already applied cleanly in $LLAMA_DIR."
elif git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$LLAMA_DIR" apply "$PATCH_FILE"
    echo "[+] Kernel & KV patches applied cleanly."
else
    echo "[*] Resetting $LLAMA_DIR working tree to clean $LLAMA_COMMIT before applying updated patch..."
    git -C "$LLAMA_DIR" checkout HEAD -- .
    while read -r created_file; do
        [[ -n "$created_file" ]] && rm -f "$LLAMA_DIR/$created_file"
    done < <(git -C "$REPO_ROOT" apply --summary "$PATCH_FILE" 2>/dev/null | awk '/create mode/{print $4}')
    if git -C "$LLAMA_DIR" apply --check "$PATCH_FILE"; then
        git -C "$LLAMA_DIR" apply "$PATCH_FILE"
        echo "[+] Updated kernel & KV patches applied cleanly."
    else
        echo "[!] ERROR: Failed to apply $PATCH_FILE onto $LLAMA_DIR (expected commit $LLAMA_COMMIT)." >&2
        exit 1
    fi
fi

echo "[*] Installing native RouteCrack + Stockfish-PVS C++ speculative engine into $LLAMA_DIR/common/..."
cp -f "${SPEC_CPP_DIR}/rc_predictor.h" "$LLAMA_DIR/common/"
cp -f "${SPEC_CPP_DIR}/rc_trie_v2.h" "$LLAMA_DIR/common/"
cp -f "${SPEC_CPP_DIR}/rc_stockfish.h" "$LLAMA_DIR/common/"
cp -f "${SPEC_CPP_DIR}/ngram-map.cpp" "$LLAMA_DIR/common/"

if [[ "$SERIES_APPLIED" != "1" && ${#SERIES_PATCHES[@]} -gt 0 ]]; then
    echo "[*] Applying patches/nemotron-apple-silicon/ (Q2_0 tests, Metal Q2_0 decode kernel, gguf-py Q2_0 block size)..."
    for p in "${SERIES_PATCHES[@]}"; do
        if git -C "$LLAMA_DIR" apply --check "$p"; then
            git -C "$LLAMA_DIR" apply "$p"
            echo "[+] Applied $(basename "$p")"
        else
            echo "[!] ERROR: Failed to apply $p onto $LLAMA_DIR." >&2
            echo "[!] $LLAMA_DIR has local changes in the files this patch touches; use a fresh clone of llama.cpp at $LLAMA_COMMIT." >&2
            exit 1
        fi
    done
fi

CMAKE_ISA_ARGS=(-DGGML_NATIVE="${GGML_NATIVE:-ON}" -DBUILD_SHARED_LIBS=ON)

if [[ "$OS_NAME" == "Darwin" ]]; then
    METAL_OPT="${GGML_METAL:-ON}"
    OPENMP_OPT="${GGML_OPENMP:-OFF}"
    BLAS_OPT="${GGML_BLAS:-ON}"
    if [[ "$ARCH_NAME" == "arm64" ]]; then
        echo "[*] Detected macOS Apple Silicon (arm64) — enabling ARM64 NEON SDOT 1x{8,4,3,2} Fractal-64 + Metal=${METAL_OPT} + BLAS=${BLAS_OPT}..."
    else
        echo "[*] Detected macOS ($ARCH_NAME) — Metal=${METAL_OPT}, BLAS=${BLAS_OPT}..."
    fi
    CMAKE_ISA_ARGS+=(
        -DGGML_OPENMP="${OPENMP_OPT}"
        -DGGML_METAL="${METAL_OPT}"
        -DGGML_BLAS="${BLAS_OPT}"
    )
elif [[ "$ARCH_NAME" == "aarch64" || "$ARCH_NAME" == "arm64" ]]; then
    OPENMP_OPT="${GGML_OPENMP:-ON}"
    echo "[*] Detected Linux ARM64 ($ARCH_NAME) — enabling ARMv8 NEON SDOT 1x{8,4,3,2} Fractal-64 kernels..."
    CMAKE_ISA_ARGS+=(-DGGML_OPENMP="${OPENMP_OPT}")
else
    # Auto-detect x86_64 CPU SIMD ISA extensions so both Zen 4/5 (AVX-512) and Zen 2/3 / Intel (AVX2) build safely
    CPU_FLAGS=""
    if [[ -r /proc/cpuinfo ]]; then
        CPU_FLAGS="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
    fi
    OPENMP_OPT="${GGML_OPENMP:-ON}"
    CMAKE_ISA_ARGS+=(-DGGML_OPENMP="${OPENMP_OPT}")

    if echo "$CPU_FLAGS" | grep -q "avx512_vnni" && echo "$CPU_FLAGS" | grep -q "avx512_bf16"; then
        echo "[*] Detected AVX-512 VNNI + BF16 (AMD Zen 4/5 or Intel Xeon) — enabling full AVX-512 tile kernels..."
        CMAKE_ISA_ARGS+=(
            -DGGML_AVX512=ON
            -DGGML_AVX512_VNNI=ON
            -DGGML_AVX512_VBMI=ON
            -DGGML_AVX512_BF16=ON
        )
    elif echo "$CPU_FLAGS" | grep -q "avx2"; then
        echo "[*] Detected AVX2 + FMA — enabling AVX2 native kernels..."
        CMAKE_ISA_ARGS+=(
            -DGGML_AVX2=ON
            -DGGML_FMA=ON
            -DGGML_F16C=ON
        )
    else
        echo "[*] Using native architecture auto-detection..."
    fi
fi

VULKAN_OPT="${GGML_VULKAN:-OFF}"
CUDA_OPT="${GGML_CUDA:-OFF}"
case "$VULKAN_OPT" in
    OFF|off|0|FALSE|false|NO|no|"") ;;
    *) echo "[*] Enabling Vulkan GPU backend (-DGGML_VULKAN=${VULKAN_OPT})..." ;;
esac
case "$CUDA_OPT" in
    OFF|off|0|FALSE|false|NO|no|"") ;;
    *) echo "[*] Enabling CUDA GPU backend (-DGGML_CUDA=${CUDA_OPT})..." ;;
esac
CMAKE_ISA_ARGS+=(
    -DGGML_VULKAN="${VULKAN_OPT}"
    -DGGML_CUDA="${CUDA_OPT}"
)

# BUILD_TESTS=1 (default off) also builds test-quantize-fns and test-backend-ops
BUILD_TARGETS=(llama-server llama-quantize llama-bench)
if [[ "${BUILD_TESTS:-0}" == "1" ]]; then
    CMAKE_ISA_ARGS+=(-DLLAMA_BUILD_TESTS=ON)
    BUILD_TARGETS+=(test-quantize-fns test-backend-ops)
fi

echo "[*] Configuring CMake build in ${BUILD_DIR}..."
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ISA_ARGS[@]}"

echo "[*] Building ${BUILD_TARGETS[*]} with ${NPROC} threads..."
cmake --build "$BUILD_DIR" --config Release -j "${NPROC}" --target "${BUILD_TARGETS[@]}"

echo "[+] Build complete: ${BUILD_DIR}/bin/ (${BUILD_TARGETS[*]})"
