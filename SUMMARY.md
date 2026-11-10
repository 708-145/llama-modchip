# llama-modchip Summary (v2.4.0)

## Ideas

- use ExpertSelectReducedMantissa global scale: save 1B
- use 3bit LUT for 4 bit block scale: save 1B
- overal 28B == 3.5bpw

## What is llama-modchip?

`llama-modchip` is a set of `ggml`/`llama.cpp` kernel patches, quantization utilities, and a unified CLI for running large language models (primarily **Nemotron-3.5-30B-A3B** and **DeepSeek-V4-Flash 284B**) on CPUs **without GPUs**. It targets **CPU-first inference** for 284B and 30B Mixture-of-Experts models on 16–64 GB RAM, breaking the traditional "enterprise GPU required" bottleneck.

## Improvements over standard llama / llama.cpp

| Aspect | Standard llama.cpp | llama-modchip (v2.4.0) |
|---|---|---|
| Target model | Dense or small MoE on CPU | Hybrid **Mamba-2 SSM + MoE** (46 SSM layers + 6 GQA + routed experts) |
| Models supported | ~7B–13B dense | **284B DeepSeek-V4-Flash** (13.17B active) & 30B Nemotron |
| Throughput (16C/32T Zen 5, no GPU) | ~2-4 tok/s (dense) | **44.36 tok/s prefill**, **9.06-10.74 tok/s pure decode** (+77%+ vs baseline) |
| Context window | ~128K (dense) | **256K** (262,144 tokens) with zero swap |
| Memory footprint (256K ctx) | ~7.08 GB VmRSS | **57.61 GiB VmRSS** but usable only 62.5 GiB total; 384 MB raw KV cache (Q4_0) or 432 MB symmetric (Q8_0/Q4_0) |
| MoE expert quantization | Q4_0 (4.25 bpw) fallback | **Fractal-64 (Q2_0)** -> **3.75 bpw** (block-64 2-level fractal) |
| Key innovations | None beyond basic ggml | 4-stream Sinkhorn-20 hyper-connections; AVX-512 ZMM register tiler (ZMM `_1x8/_1x4...`); L1D/L2 cache-blocked GEMM (L2 rows 128, inner L1D tiles {8,4,3,2}); Non-temporal prefetch `_MM_HINT_NTA`; RouteCrack + Stockfish-PVS draftless speculative decoding; Entropy-guided EAKV & Virtual Positional Indexing; 8-slot Q8_K activation cache |

## Can it be compiled / run standalone?

Yes. `llama-modchip` ships with:

- **`install.sh`** - automated Linux installer.
- **`modchip`** CLI - setup, start, chat, bench, eval, quantize, web, claude, stop.
- **`scripts/llama_modchip.py`** - header/binary inspector (<5 ms), topology profiler, architecture safety locks, runtime controller (`/dev/shm/llama_modchip.bin`), Curses TUI.
- **`build_fractal_moe_gguf.sh`** - builds asymmetric Dyadic Fractal-MoE GGUF (profiles: `fractal64`, `mxfp4`, `mxfp4_q5`).
- Manual CMake build instructions documented in README ("Repository Structure & Manual CMake Build Instructions").
- Patched `llama-quantize` binary supports writing `Q2_0` Fractal-64 blocks (run `./modchip setup --skip-model` first or pass `--quantize-bin /path/to/patched/llama-quantize`).

### Build process (CMake + patched llama.cpp)

`llama-modchip` patches and builds a custom version of `llama.cpp` that includes the Fractal-64 quantizer, Q2_0 CPU optimizations, and Zen 5/Apple Silicon support. The main build script is `scripts/build_zen5_llama.sh`. Here's the step-by-step workflow:

1. **Clone/check out llama.cpp**  
   The script clones `https://github.com/ggml-org/llama.cpp.git` or checks out a pinned commit (`d2f83055d6e3b379b5d34c4837122a918cf402c2`) into a local directory (`LLAMA_DIR`).

2. **Apply stacked patches** (in file-name order) from `patches/`  
   - **Main patch**: `zen5-nemotron-256k-optimizations.patch` — the core set of optimizations (Q2_0 Metal decode kernel, gguf-py block size, Q2_0 tests, Fractal-64 quantization, dynamic MoE Top-K, L2-quadrature tail pruning, RouteCrack speculative decoding).  
   - **Series patches** (Apple Silicon only by default, controllable via `APPLE_SILICON_PATCHES` env var): patches in `patches/nemotron-apple-silicon/` (Q2_0 tests, Metal SDOT, gguf-py Q2_0 block size).  
   - **Additional CPU patches**: `dsv4-64gb-cpu.patch`, `deepseek-v4-flash-fractal-256k.patch` (Fractal-64 quantization for DeepSeek-V4-Flash 284B).  
   - Patches are stacked; each subsequent patch changes lines the previous one added, so the tree is checked for consistency after application.

3. **Configure with CMake** using `CMAKE_ISA_ARGS` that detect and enable the appropriate SIMD ISA extensions:  
   - **AVX-512** (Zen 4/5 or Intel Xeon): `-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON`  
   - **AVX2** (Zen 2/3 or Intel): `-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`  
   - **Metal** (Apple Silicon): `-DGGML_METAL=ON` for Metal GPU backend + Metal SDOT kernels  
   - **ARM64 NEON**: `-DGGML_OPENMP=ON` for NEON SDOT 1x{8,4,3,2} Fractal-64 kernels  
   - **CUDA/Vulkan**: optional GPU backends (`-DGGML_CUDA`, `-DGGML_VULKAN`)  
   - **OpenMP**: `-DGGML_OPENMP=ON` for multi-threaded CPU execution  

4. **Build** with `cmake --build` using `Release` configuration and `NPROC` parallelism (number of CPU cores). Build targets: `llama-server`, `llama-quantize`, `llama-bench` (and `test-quantize-fns`, `test-backend-ops` if `BUILD_TESTS=1`).

5. **Result** — A custom `llama.cpp` binary that includes:  
   - **Fractal-64 (Q2_0) quantizer**: custom 30-byte block-64 quantizer for MoE weight superblocks, enabling 3.75 bpw instead of 4.25 bpw for Q4_0.  
   - **Dynamic MoE expert selection**: `LLAMA_MOE_TOP_K` env var (default Top-6 + 1 shared expert) to skip inactive experts per layer.  
   - **L2-quadrature MoE tail pruning**: preserves expert balance using quadratic norm rescaling instead of linear sum.  
   - **RouteCrack draftless speculative decoding**: predicts tokens without full forward pass, only computing necessary up/down projections.  
   - **Efficient AVX-512/AVX2 kernels**: tile GEMM, non-temporal prefetch `_MM_HINT_NTA`, token-to-core L1D/L2 affinity, 8-slot Q8_K activation cache.  
   - **Q2_0 fused kernel** (`ggml_vec_dot_q2_0_q8_0_generic`): unpacks 64 3-bit indices and translates via `_mm256_shuffle_epi8` in 1 cycle, multiplies sub-blocks by integer scales inside `vpmaddwd` at zero extra SIMD cost.

### Building the Fractal-64 quantized GGUF

Separate from the llama.cpp build, `scripts/build_fractal_moe_gguf.sh` asymmetricly dyadic-fractal-quantizes a GGUF model:

- **Input**: a base GGUF (e.g., Nemotron-3.5-30B-A3B BF16/F16 at 17.59 GiB, or DeepSeek-V4-Flash-0731 IQ1_S).  
- **Output**: Fractal-64 quantized model at 14.35 GiB (Nemotron) or 56.56 GiB (DeepSeek-V4).  
- **Profiles**:  
  - `fractal64`: routed experts → Block-64 2-Level Fractal (Q2_0, 30 bytes / 64 weights = 3.75 bpw). SSM & shared experts → Q4_0; GQA attn/head → Q8_0.  
  - `mxfp4`: routed experts → Dyadic MXFP4 (4.25 bpw); SSM & shared → Q4_0; attn/head → Q8_0.  
- **Requirements**: patched `llama-quantize` binary (auto-downloaded via `./modchip setup --skip-model` or passed via `--quantize-bin`). An `imatrix` file may be needed for best results.  
- **Quality gate**: 100/100 on tool-calling, needle retrieval, long-context math (benchmarks/FRACTAL_MOE_QUALITY_GATE.md).

### Quick start

```bash
# 1. Build llama.cpp with all optimizations
./scripts/build_zen5_llama.sh    # auto-detects CPU, applies patches, cmake + make

# 2. (Optional) Build a Fractal-64 quantized GGUF
./scripts/build_fractal_moe_gguf.sh \
    --input /path/to/input-model.gguf \
    --output /path/to/output-fractal-moe.gguf \
    --profile fractal64 \
    --threads 16
```bash
# 1. Build llama.cpp with all optimizations
./scripts/build_zen5_llama.sh    # auto-detects CPU, applies patches, cmake + make

# 2. (Optional) Build a Fractal-64 quantized GGUF
./scripts/build_fractal_moe_gguf.sh \\
    --input /path/to/input-model.gguf \\
    --output /path/to/output-fractal-moe.gguf \\
    --profile fractal64 \\
    --threads 16
```

### Clarification: Top-K MoE, Tail Pruning, Dynamic Expert Selection (NOT Partial Tensor Processing)

The terms **Top-K MoE**, **Tail Pruning**, and **Dynamic Expert Selection** in llama-modchip do **NOT** mean partial up/down projection or partial tensor computation. They refer to **complete-expert selection and correction**:

- **Top-K MoE (`LLAMA_MOE_TOP_K`)**: Selects a complete set of K *entire* experts per layer. If K = 4, the engine computes **all the activations of exactly those 4 expert projection matrices** (`up_exps`, `gate_exps`, `down_exps`) for each token. The other 2 routed experts are **omitted entirely** from the forward pass.

- **Dynamic Expert Selection**: The K experts are chosen **per token** (or per layer) based on the router's logits, but once selected the computation for those experts proceeds **to completion** — there is no "partial" computation on the selected experts.

- **L2-quadrature Tail Pruning**: After the K selected experts have been fully computed, their output vectors (each `d`-dimensional) are **rescaled using the L2-norm** (`α_L2 = sqrt( Σᵢ wᵢ² / Σᵢ wᵢ² )`) to preserve the exact L2 norm of the full expert ensemble. This is a **post-hoc scaling correction**, not a reduction in the amount of computation for the selected experts.

- **CSA/LID Skip (`GGML_TENSOR_FLAG_DSV4_SKIP`)**: The CSA/LID compressor nodes that implement Sinkhorn-20 hyper-connections are tagged as "skip" on 7/8 decode steps, **eliminating whole-graph rebuilds** rather than partially computing graph nodes.

These mechanisms **avoid computing the other (non-selected) experts altogether** and **reconstruct the missing contribution mathematically** (L2-quadrature scaling) so that the final output is **numerically equivalent** to processing all experts, while delivering **significant speedups** by cutting both compute and DRAM traffic.

Thus, llama-modchip's MoE optimization is **expert-selection-plus-correction**, not **partial tensor processing**.
```

## Fractal 64 (Q2_0) quant type - how it works

`Fractal-64` is a **custom 30-byte block-64 quantizer** for MoE weight superblocks (`q2_0`). It implements a **two-level fractal hierarchy**:

1. **Level 1 (super-block)**: 64-weight block, master FP16 scale `d = max_b(s_opt[b]) / 19.0f`.
2. **Level 2 (sub-blocks)**: Eight 4-bit shifted sub-scales `{sc in {0..15}}` mapped to integer multipliers `mult = 4 + sc in {4..19}`. Because local kurtosis across 8 weights is low, the ratio `max/min` stays within 4.75x, making `{0..3}` dead codes; shifting to `{4..19}` boosts resolution by +31%.

A third level adds a **non-linear dyadic codebook** `{-14, -8, -4, -1, +1, +4, +8, +14}` (step spacing ~0.71x near zero for Laplacian/Gaussian MoE weights, reaching +-14 for outliers).

**Zero-overhead fused kernel** (`ggml_vec_dot_q2_0_q8_0_generic`):
- Unpacks 64 3-bit indices via `_mm_unpacklo_epi64(qh_lo, _mm_srli_epi16(qh_lo, 1))` and translates via `_mm256_shuffle_epi8(lut256, idx)` in **1 cycle**.
- Computes `p01 = _mm256_maddubs_epi16(_mm256_abs_epi8(q01), _mm256_sign_epi8(q8_0, q01))` using `_mm256_madd_epi16(p01, vpmaddwd)`.
- Multiplies each 8-weight sub-block by its sub-scale inside `vpmaddwd` at **zero extra SIMD cost**.
- Streams with `_MM_HINT_NTA` cache-bypass hints.

**Results** (from `benchmarks/FRACTAL_MOE_QUALITY_GATE.md`):
- Disk size **minus 3.24 GiB** (-18.4%) vs Q4_0 (17.59 GiB).
- Effective bits/weight **3.75 bpw** vs 4.50/4.78 baseline.
- SQNR **17.78 dB** vs 21.02 dB (Q3_K).
- Quality gate **100/100** on tool-calling, needle retrieval, long-context math.
- Decode **+33.5% (MXFP4) / +28.0% (Fractal-64)** vs Baseline-Q4_0.

Used for **DeepSeek-V4-Flash 284B** (routed experts -> Fractal-64) and **Nemotron-3.5-30B-A3B** (routed experts -> Fractal-64).

**Note on lineage:** `Fractal-64` is an **IQ4_NL-family quantizer** — two-level hierarchical non-linear scaling (FP16 master scale per 64-weight super-block + per-sub-block integer scales) decoded through an **8-value non-linear LUT** (`{-14,-8,-4,-1,+1,+4,+8,+14}`). The specific twist is **8-weight sub-blocks** (8×8 micro-grid) with **shifted 4-bit sub-scales** (`mult ∈ {4..19}`) fused inside AVX2 `vpmaddwd` at zero extra SIMD cost, giving **3.75 bpw** where stock `Q3_K`/`IQ3_S`/`IQ3_XXS` (256-block K-quants) fall back to `Q4_0` on Nemotron's `1856 × 2688` expert dims.

## Quick links
- Overview: `README.md`
- Architecture: `ARCHITECTURE.md`
- Technical paper: `paper/llama-modchip-paper.pdf`
- FAQ: `FAQ.md`
- Release notes: `RELEASE_NOTES.md` (v2.4.0, Nov 2026)
- Install: `install.sh`
- Mod chip CLI: `modchip`
- Benchmarks: `benchmarks/eval_quality_and_tools.py`, `FRACTAL_MOE_QUALITY_GATE.md`

---

## Deep Dive: CSA/LID Skip Mechanism

### What is CSA/LID?
**CSA/LID** = **Compressed Sparse Attention / Latent Indexer Decoder**. A core component in the **DeepSeek-V4-Flash 284B** architecture implementing **sparse indexing** for Mixture-of-Experts (MoE) layers. Part of the **Sinkhorn-20 Hyper-Connections** system.

- **CSA (Compressed Sparse Attention)**: Handles the attention computation efficiently
- **LID (Latent Indexer Decoder)**: Manages expert routing/indexing
- **HCA (Hyper-Connection Architecture)**: Connects everything via 4-stream Sinkhorn-20 mixing

### The CSA/LID Skip Optimization

#### Problem It Solves
During **autoregressive decoding** (generating tokens one-by-one), the DeepSeek-V4 model previously rebuilt the **entire 4,551-node compute graph** on every single token. This includes the CSA/LID compressor nodes that perform sparse attention indexing — expensive work that's only needed when *new* context is being compressed, not during regular decode steps.

#### How It Works

**1. Dummy Compressor Nodes Tagged with Skip Flag**
The compute graph contains dummy CSA/LID compressor nodes that are **always present** in the graph structure. These nodes are tagged at graph-build time with:
```cpp
flags |= 0x100  // GGML_TENSOR_FLAG_DSV4_SKIP
```

**2. Runtime Skip Decision**
During decoding, a global flag is evaluated:
```cpp
extern int ggml_dsv4_skip_csa_comp;

ggml_dsv4_skip_csa_comp = (plan.state_write_pos.size() == 1 && plan.state_write_pos[0] == -1) ? 1 : 0;
```

**Condition for skipping**: `plan.state_write_pos.size() == 1 && plan.state_write_pos[0] == -1`
- Means: "We're reading from a single snapshot position that's -1 (no new state to write)"
- This occurs on **7 out of every 8 decode steps** (non-compressor steps)
- The **1st step** (compressor step) writes new state → skip = 0
- The **next 7 steps** just read existing state → skip = 1

**3. Execution Flow**
```cpp
// In graph execution:
if (skip_csa_comp) {
    // Skip NOPs or dynamically skipped dummy CSA/LID compression node
    // Just reuse existing recurrent state without recomputing
} else {
    // Full CSA/LID computation needed (compressor step)
}
```

#### What Gets Eliminated

| Without Skip | With Skip |
|---|---|
| 4,551-node graph rebuild every token | Graph rebuild **only on compressor steps** (1/8 tokens) |
| CSA/LID attention computed every step | CSA/LID attention **skipped 7/8 steps** |
| Full recurrent state recomputation | **State reuse** from previous compressor step |

#### Performance Impact (from ARCHITECTURE.md / RELEASE_NOTES.md)

| Metric | Before Skip | After Skip | Improvement |
|---|---|---|---|
| Single-token decode latency | 169.3 ms/tok | 100.5–110.3 ms/tok | **1.5–1.7× faster** |
| Decode throughput | 5.91 tok/s | 9.06–10.74 tok/s | **+53% to +68%** |
| Prefill throughput | 24.70–25.05 tok/s | 43.16–44.36 tok/s | **+77%** |
| Draft acceptance (speculative) | — | 94.5% | Enables high-quality speculative decoding |

#### Interaction with Other Optimizations
The CSA/LID Skip works synergistically with:
- **8-slot Q8_K/Q8_0 activation cache** (`s_q8_cache_buf`) — caches quantized inputs
- **MUL_MAT_ID row-group reuse** (`g_tls_mmid_ids_valid`) — shares work across `up_exps` and `gate_exps`
- **Token-to-core L1D/L2 affinity** — pins `it` to thread `ith = it % nth` across `DSV4_HC_POST → DSV4_HC_PRE → RMS_NORM`
- **Non-temporal prefetch** (`_MM_HINT_NTA`) — protects L3 cache for hot KV/SSM state

#### Weights and Data Used
- **Weights**: The CSA/LID mechanism itself doesn't "select" weights. It operates on **recurrent state tensors** (the compressed attention/hyper-connection state), not on MoE expert weights.
- **Inputs**: Previous step's recurrent state (`kv_csa`, `kv_lid`, `kv_hca`)
- **Outputs**: Updated recurrent state (only on compressor steps)
- **Router logits**: The MoE router still runs normally; CSA/LID Skip only affects the *hyper-connection* state update path

#### Key Insight
This is **not** a weight-selection or pruning technique. It's a **graph-execution optimization** that recognizes the recurrent state only needs updating on **compressor steps** (1/8 tokens) and can be **carried forward unchanged** on the other 7/8 steps. The "skip" eliminates an entire subgraph of attention computations, not individual weights.

---

## RouteCrack + Stockfish-PVS: Draftless Speculative Decoding

### What Is It?

**RouteCrack + Stockfish-PVS** is llama-modchip's **primary speculative decoding engine** — a **draftless** (no separate draft model) approach that uses **batch verification** to generate multiple tokens per forward pass. Unlike classic MTP (which requires a separate draft model), RouteCrack uses the **existing main model** with a lightweight **statistical predictor** that proposes candidate tokens, which are then verified in a single batch forward pass.

### How It Works

**1. RouteCrack Prediction (Draft Generation)**
- **8-Level FNV-64 Cascade**: A hash-based predictor that analyzes the current context and proposes multiple candidate tokens
- **11,382-Entry Multi-Token Macro Codebook**: Pre-computed token sequences for common patterns (96.44% precision)
- **Stockfish-PVS Transposition Tables**: 192 KB L2-resident Zobrist hash tables (`tt5/tt4/tt3`) that cache promising token sequences
- **RouteCrack** = the routing/prediction engine that decides which tokens to propose

**2. Stockfish-PVS (Principal Variation Search)**
- **1-Ply Quiescence PVS**: Evaluates candidate token sequences with 1-ply lookahead
- **Singular Horizon Extension** (`M=7`): Extends search when `true_consensus_hits == d` (high confidence)
- **Refutation Tracking**: Tracks which token sequences fail, avoiding repeated failures
- **Stockfish History Gravity**: `h += bonus - h * |bonus| / 512` — prioritizes historically successful token sequences

**3. Batch Verification**
- **Multiple tokens proposed** (typically 3-7 per step)
- **Single forward pass** verifies all proposed tokens simultaneously
- **Tokens accepted** if they match the model's predictions
- **Tokens rejected** → rollback to last accepted position, continue from there

### Key Performance Characteristics

| Metric | Value |
|---|---|
| Draft acceptance rate | **96.39%** (best case), **59-66%** (typical) |
| Mean speculative length | **4.18-6.90 tok/step** |
| Decode throughput (16C Zen 5) | **48.39 tok/s avg / 60.40 tok/s peak** |
| Decode throughput (Threadripper 7985WX) | **111.62 tok/s warm** |
| Prefill throughput | **386.00 tok/s** (Threadripper) |

### RouteCrack + Stockfish-PVS vs. Classic MTP

| Aspect | RouteCrack + Stockfish-PVS | Classic MTP |
|---|---|---|
| **Draft model** | None (uses existing main model) | Separate MTP head (~1.16 GB) |
| **Verification** | Batch forward pass (single model) | Separate MTP model forward pass |
| **Memory overhead** | 192 KB (TT tables only) | ~1.16 GB (MTP model) |
| **Best for** | CPU inference (no extra model needed) | GPU scenarios with VRAM to spare |
| **Acceptance rate** | 59-96% | Varies by model quality |

### Is There Partial Execution / Neuron Skipping in Up/Down Projections?

**No.** RouteCrack + Stockfish-PVS does **NOT** perform partial execution or neuron skipping in the up/down projections. Here's the distinction:

**What RouteCrack does:**
- Proposes **candidate token sequences** using statistical prediction
- Verifies **entire token sequences** in a single batch forward pass
- **Each token** goes through **complete** up/down projections (all experts, all layers)
- **No neurons are skipped** — the full model computes for each proposed token

**What it does NOT do:**
- **No partial up-projection**: The `up_exps` matrix is fully computed for each token
- **No partial down-projection**: The `down_exps` matrix is fully computed for each token
- **No neuron skipping**: All neurons in the FFN layers are computed
- **No weight pruning**: All expert weights are loaded and used

**The "speculation" is at the token level, not the neuron level:**
- RouteCrack predicts **which tokens** to generate
- The main model **fully computes** each proposed token
- If a token is accepted, its computation was "wasted" (but amortized across multiple tokens)
- If a token is rejected, the model rolls back and tries a different token

### Bottom Line

RouteCrack + Stockfish-PVS is a **token-level speculative decoding** technique that:
- Uses **statistical prediction** (not a separate draft model)
- Verifies **entire token sequences** in batch forward passes
- Achieves **48-112 tok/s** decode on CPU
- Does **NOT** perform partial execution or neuron skipping — each proposed token goes through the **complete** up/down projections

---

## Why Is It Called "modchip"?

The name **`llama-modchip`** (also called **"Priiloader / Modchip"**) comes from its design as a **runtime microarchitecture patcher** for CPU LLM inference — inspired by **console modchips** and **Priiloader** (`hacks.ini`).

### What "Modchip" Means Here

A **console modchip** is a small firmware file that patches/overrides a game's hardware drivers so they run correctly on a specific platform — essentially a microcode-level patch for the host OS/hardware.

**`llama-modchip` does the same thing for GGUF models:**

| Console Modchip | llama-modchip |
|---|---|
| Inspects console hardware | Inspects `.gguf` model header + Linux CPU/cache topology (<5 ms) |
| Patches game drivers at runtime | Patches `ggml-cpu`/`ggml-quants` kernel parameters at runtime |
| Classifies hacks by safety | Classifies microarchitectural hacks into safety tiers: 🟢 SAFE / 🔵 AUTO-ADAPTIVE / 🟡 EXPERIMENTAL / ⚪ LOCKED |
| Lives in firmware | Lives in `/dev/shm/llama_modchip.bin` (48-byte POSIX shared-memory struct) |
| Applied once at boot | Applied per-forward-pass (live override) |

### Dual Operation Levels

1. **Startup & Graph-Construction Profile** — Injects load-time parameters:
   - `LLAMA_MOE_TOP_K`, `--repack`, `MADV_HUGEPAGE`
   - Deterministic per-CCD `taskset -c` core pinning
   - Quantization cache types (`-ctk q8_0 -ctv q4_0`)

2. **Live Per-Forward-Pass Control** — Writes to `/dev/shm/llama_modchip.bin` read on **every forward/draft pass** by:
   - `ggml-cpu/ops.cpp` → attention window, sinks, SWA stride, L3 window
   - `common/ngram-map.cpp` → RouteCrack enabled, cert thresholds (`rc_cert_d0/d1/d2`)

### The "Priiloader" Alias

The interactive TUI (`scripts/llama_modchip.py --tui`) is called **"Priiloader"** — a custom curses interface (from `hacks.ini`) that lets you **toggle microarchitectural hacks live at runtime**, just like Priiloader on the Wii lets you toggle system patches.

### Command Surface

When you run `./modchip chat`, `./modchip quantize`, `./modchip bench`, or `./modchip eval`, you're invoking the **Modchip** — the microarchitecture-adaptive patching layer that makes any GGUF model run optimally on your specific CPU.

---

*Project: `synchopate/llama-modchip` · Version 2.4.0 · License: Apache 2.0*