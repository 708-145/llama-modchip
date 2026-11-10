# llama-modchip Summary (v2.4.0)

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

It runs **offline on CPU** (no CUDA/GPU needed); heavy lifting is custom SIMD kernels (AVX-512BW, AVX2) and cache-aware scheduling. For Windows there is a hybrid GPU+CPU pack (`windows/README.md`); Apple Silicon has Metal/NEON/M4 scripts.

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
*Project: `synchopate/llama-modchip` · Version 2.4.0 · License: Apache 2.0*