# 🚀 NVIDIA Nemotron (`30B-A3B` & `H-8B`) — Deep-Dive Optimization Guide

> **Back to:** [Main Overview (`README.md`)](../README.md) · [Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](HARDWARE_OPTIMIZATIONS.md) · [DeepSeek-V4-Flash 284B Guide (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](DEEPSEEK_V4_OPTIMIZATIONS.md) · [Full Mathematical Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md)

---

## 1. Architectural Profile: Why `Nemotron-3.5-30B-A3B` Is Ideal for 256K CPU & Hybrid Inference

`NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`nemotron_h_moe`) is a **52-layer Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE** model (`30.2B` total parameters, **`3.0B` active per token**):
- **23 Mamba-2 State-Space Layers (`SSM`)**: Maintain a constant-size $O(1)$ recurrent state (`18.4 MB` total across all 23 layers) regardless of whether context is `1K` or `256K` tokens.
- **6 Grouped-Query Attention Layers (`GQA`, `2` KV heads per layer = `12` KV heads total)**: Only **6 out of 52 layers** allocate a growing KV cache. At a full **`262,144`-token (`256K`) context**, the KV cache is only **`384 MB` (`Q4_0`) to `432 MB` (`EAKV Q8_0/Q4_0`)** — compared to `7.1–16.0 GB` on standard 48-layer GQA Transformers.
- **23 Sparse MoE Layers (`128` routed experts/layer, `top-6` active + `1` double-width shared expert)**: `~1.8 GB` of active weights read per token during `N=1` decode.

---

## 2. Summary of Optimizations Applied to `Nemotron-3.5-30B-A3B`

| Optimization Layer | Technique Implemented | Target Bottleneck Solved | Measured Gain (`Zen 5` / `Threadripper` / `Vulkan`) |
| :--- | :--- | :--- | :--- |
| **1. Quantization (`Fractal-64`)** | Block-64 2-Level Hierarchical Non-Linear `Q2_0` (`30B / 64w = 3.75 bpw`) from `BF16` master weights ([`scripts/build_fractal_moe_gguf.sh`](../scripts/build_fractal_moe_gguf.sh)) | `1856 × 2688` expert dimensions are not divisible by `256`, breaking standard `Q3_K`/`IQ3_S` in `llama.cpp` | **`17.59 GiB -> 14.35 GiB` (`-6.01 GiB` idle RAM)**, **`+28.0%` decode (`25.02 tok/s`)**, **`100/100` Quality Gate** |
| **2. Quantization (`Fractal-MXFP4`)** | Asymmetric Block-32 `MXFP4` (`4.25 bpw`) on routed MoE experts + `Q8_0` on GQA/SSM backbone + `_MM_HINT_NTA` | Bandwidth bottleneck on CPU & VRAM capacity on `16 GB` GPUs | **`15.20–15.95 GiB`**, **`+33.5%` CPU decode (`26.08 tok/s`)**, **`75.1–80.8 tok/s` on AMD Vulkan Hybrid** |
| **3. SIMD Kernel (`dot_q2_0_q8_0`)** | Single-cycle AVX2 variable bit-shifts (`_mm256_sllv_epi32` / `_mm256_srlv_epi32`) + sub-scale rescaling fused into `vpmaddwd` | Scalar bit-unpacking overhead in 3-bit hierarchical blocks | **`+55%` kernel speedup** (`86.5 ms -> 55.8 ms` per 10M blocks), **zero extra SIMD instructions** |
| **4. Speculative Engine (`Stockfish-PVS v2`)** | `1.64 MB` L2/L3 Zobrist Transposition Tables (`tt5/tt4/tt3`) + 1-Ply Quiescence PVS + `RouteCrack` 8-Level FNV-64 Cascade + Macro Codebook (`M=7`) | 1B neural draft models read `3.0 GB` from DDR5 (`40 ms`) and flush L3 cache | **`48.39 tok/s` avg / `60.40 tok/s` peak** on 16C Zen 5 (`96.4%` acc, `len 6.90`); **`111.62 tok/s` warm** on Threadripper 7985WX |
| **5. Cache Isolation (`_MM_HINT_NTA`)** | Bidirectional Non-Temporal L3 bypass on cold MoE weights and KV tokens older than `LLAMA_KV_L3_WINDOW=2048` | Streaming `1.8 GB/tok` of MoE weights evicts hot Mamba-2 state and recent KV tokens from `64 MB` L3 | Keeps `18.4 MB` Mamba-2 state + `4.88 MiB` hot KV window permanently in L3 cache |
| **6. Virtual Memory (`2MB HugePages`)** | `MADV_HUGEPAGE` on weight mmap + compute arenas + `2MB`-aligned `posix_memalign` | `17.6 GiB` in `4KB` pages = `4.5M` PTEs, thrashing the `3,072`-entry Zen 5 L2 STLB (`14 ms/tok` page-walk stall) | Expands L2 STLB coverage to **`6.14 GB`**, cutting dTLB page-walk latency from `14.0 ms` to **`0.8 ms/tok` (`-99.6%` misses)** |
| **7. Weight Tile Repacking (`q4_0_8x8`)** | In-memory AVX-512 VNNI (`VPDPBUSD`) / AVX2 interleaved `8×8` matrix tile repacking at model load | Cache-line splits and nibble-unpacking stalls during GEMV/GEMM | **`+19%` raw GEMV throughput** and **`241–306.8 tok/s` (`pp512`) prefill** on 16C Zen 5 |
| **8. KV Cache (`EAKV + VPI`)** | Entropy-Guided Asymmetric KV (`Q8_0` Keys for RoPE phase precision, `Q4_0` Values) + Virtual Positional Indexing | FP16 KV cache growth and RoPE degradation at `128K–256K` context | Full **`256K` context in `432 MB`**, `100%` passkey needle retrieval |
| **9. Agentic Loop (`HARP-Cache` Slot-0)** | `/dev/shm` memory-mapped KV + Mamba-2 recurrent state snapshot + SSE Gateway auto-restore | Re-evaluating Claude Code's `22.2K`-token 40-tool system prompt takes `130.8 s` on CPU | **`18.9 ms` warm turn restore** (`~6,800x` faster TTFT) |
| **10. Dynamic MoE Pruning (`Top-K 6->4`)** | Opt-in norm-preserving dynamic MoE router truncation (`LLAMA_MOE_TOP_K=4` via `/dev/shm/llama_modchip.bin`) | Reading 6 routed experts per layer on bandwidth-constrained DDR4/DDR5 | **`-33%` routed MoE DRAM traffic** (switchable live via TUI) |

---

## 3. Verified Quality & Speed Gate (`100 / 100` Score)

All three quantization profiles (`Baseline-Q4_0`, `Fractal-MXFP4`, and `Fractal-64-BF16`) are verified by the automated grading suite [`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py) (full report in [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](../benchmarks/FRACTAL_MOE_QUALITY_GATE.md)):

| Metric / Suite ([`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py)) | `Baseline-Q4_0` (`17.59 GiB`) | `Fractal-MXFP4` (`15.95 GiB`) | **`Fractal-64-BF16` (`14.35 GiB`)** | Engineering Delta |
| :--- | :---: | :---: | :---: | :---: |
| **1. Agentic Tool-Calling (`30` pts — 5 multi-tool JSON calls)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | `100.0%` parity |
| **2. Executable Python Code Gen (`30` pts — sandbox-tested)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | `100.0%` parity |
| **3. Long-Context Needle (`8.5K`) & Symbolic Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** | `100.0%` parity |
| **Single-Turn Decode Throughput (`N=1` base)** | `19.54 tok/s` | **`26.08 tok/s`** | **`25.02 tok/s`** | **`+28.0%` (`+33.5%` MXFP4)** |
| **Multi-Turn In-Context Refactoring (`Stockfish-PVS M=7`)** | `40.91 tok/s` | **`48.39–60.40 tok/s`** | **`48.39–60.40 tok/s`** | **`96.39%` draft acceptance** |
| **Idle / Peak Resident RAM (`VmRSS` @ `256K` Context)** | `21.41 / 21.71 GiB` | `17.46 / 18.86 GiB` | **`15.40 / 17.27 GiB`** | **`-6.01 GiB` idle RAM saved** |

### 3.1 Empirical Paired KL-Divergence & Logit Fidelity (`Q4_0` Reference Baseline)

Independent paired evaluation conducted by `@khairullinr` on an Apple M4 Max (`36 GB` LPDDR5X-8533, Metal) using `llama-perplexity --kl-divergence` on a Python source-code corpus (`12` chunks × `2048` ctx, `~12.3k` scored tokens, `-ngl 99 -fa 1 -b 256 -ub 256`, [Issue #16](https://github.com/synchopate/llama-modchip/issues/16#issuecomment-6752457578)):

- **Reference Model**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (`17.59 GiB`, nominal `4.50 bpw`). Note this measures fidelity relative to `Q4_0`, not unquantized `BF16`.
- **Target Quantization**: `Fractal-64-BF16` (`14.35 GiB`, nominal `3.75 bpw`, `-18.42%` file footprint).
- **Top-1 (Argmax) Token Agreement**: **`94.347 ± 0.208%`** (reported as `Same top p` by `llama-perplexity`). In ~19 out of 20 token positions, `Fractal-64` selects the exact same greedy token as the `Q4_0` baseline.
- **Kullback-Leibler Divergence (KLD)**:
  - **Mean KLD**: **`0.056606 ± 0.001964 nats`**
  - **90th Percentile KLD**: **`0.139480 nats`**
  - **95th Percentile KLD**: **`0.280968 nats`**
  - **99th Percentile KLD**: **`0.845352 nats`**
  - **Median KLD**: **`0.000829 nats`**
  - *Distribution profile*: On the vast majority of tokens, the predicted logit distributions are nearly identical (median KLD `< 0.001 nats`), with divergence concentrated in a tail of roughly 5–10% of positions.
- **Per-Token Log-Likelihood Correlation**: **`97.28%`** (`Cor(ln PPL(Q), ln PPL(base))`).
- **Paired Perplexity & Probability Shifts**:
  - **Mean PPL(base)**: `2.082848 ± 0.037092`
  - **Mean PPL(Q)**: `2.123941 ± 0.039133`
  - **Mean Paired $\ln(\text{PPL}(Q)/\text{PPL}(\text{base}))$**: `0.019537 ± 0.004266` (**`+1.97%`**)
  - **Mean Paired $\text{PPL}(Q) - \text{PPL}(\text{base})$**: `0.041092 ± 0.009110`
  - **Median $\Delta p$**: **`-0.000%`** (indicating no systematic probability bias)
  - **RMS $\Delta p$**: **`8.133 ± 0.197%`**
- *Methodological Caveats*: Single Python code corpus (which has low entropy, making median KLD and top-1 agreement look favorable compared to open prose); tokens within context windows are correlated, so true asymptotic variance may exceed i.i.d. estimates.

---

## 4. Quick Commands for Nemotron

```bash
# 1. Default 1-command launch (Linux / macOS / Windows Git Bash)
./modchip chat

# 2. Build Fractal-64 (14.35 GiB from BF16) or Fractal-MXFP4 (15.95 GiB from Q4_0)
./modchip quantize --profile fractal64 --input /path/to/bf16.gguf --output /path/to/Fractal-64.gguf
./modchip quantize --profile mxfp4     --input /path/to/q4_0.gguf --output /path/to/Fractal-MXFP4.gguf

# 3. Run the 100-Point Quality & Tool-Calling Gate
./modchip eval
```
