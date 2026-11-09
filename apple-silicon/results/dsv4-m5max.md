# DeepSeek-V4-Flash-0731-Fractal 284B on an M5 Max (64 GB)

**27.0 tok/s decode in `llama-bench`, 20.7–23.6 tok/s in `llama-server`, 0 swapouts.**

Machine: Apple M5 Max (6 Super + 12 Performance cores), 64 GB unified memory, macOS 27.2,
`iogpu.wired_limit_mb=57344`. Model: `DeepSeek-V4-Flash-0731-Fractal-0000{1,2,3}-of-00003.gguf`
(60,729,886,080 bytes, 56.56 GiB). Engine: built by [`../build-dsv4-engine.sh`](../build-dsv4-engine.sh)
(llama.cpp `d2f83055` + `patches/dsv4-64gb-cpu.patch` + `patches/apple-silicon/0001`–`0003`).
Launch with [`../run-dsv4.sh`](../run-dsv4.sh).

## Results

`llama-bench`, `-fa 1 -ngl 99 -t 12`, experts of layers 10, 12, 14, 16 and 18 on the CPU, everything else on Metal:

| Test | t/s |
| :--- | ---: |
| pp512 | 169.4 ± 29.6 |
| tg128 | 27.0 ± 0.4 |
| tg64 @ d8192 | 25.9 ± 0.6 |
| tg64 @ d16384 | 22.8 ± 2.3 |

`llama-server` with the same placement, `bench_server.py` workloads, thinking off, greedy decoding:

| Workload | Prompt | Generation | Swapouts |
| :--- | ---: | ---: | ---: |
| code_edit (3,322-token prompt, 1,200 generated) | 69.2 t/s | 20.7 t/s | 0 |
| code_new | – | 23.3 t/s | 0 |
| free_text | – | 23.6 t/s | 0 |

An earlier live session on the `--n-cpu-moe 4` build generated 11,667 tokens at an average of
16.6 tok/s, falling to 12.6 tok/s at the end as the context grew.

## Which layers go to the CPU matters more than how many

The model does not fit the Metal working set (57,916 MiB of weights against 53,084 MiB by default,
57,344 MiB with the raised limit), so some routed experts must stay on the CPU. Which layers
they come from changed decode by 2x, with no corresponding change in the number of bytes:

| CPU-resident experts | Quant types | tg128 |
| :--- | :--- | ---: |
| layers 0–3 (`--n-cpu-moe 4`) | IQ2_XXS, IQ1_M, IQ1_S | 13.6 |
| layers 10,12,14 | IQ1_S | 28.5 ± 2.6 |
| layers 10,12,14,16 | IQ1_S | 28.2 |
| **layers 10,12,14,16,18** | **IQ1_S** | **27.0** |
| layers 10–20, even | IQ1_S | 24.2 |
| layers 10–24, even | IQ1_S | 23.7 |

`--n-cpu-moe N` always takes layers 0..N-1. Toggling `GGML_METAL_FUSION_DISABLE` (13.04) and
`GGML_METAL_CONCURRENCY_DISABLE` (13.21) did not change decode at `--n-cpu-moe 4`, and neither did a
7% faster Metal Q2_0 kernel (12.95 → 12.96), so the GPU was not on the critical path. A 10 s sample
of the process puts the time in CPU dot products for the experts of layers 0–3.

**Why layers 0–3 are slow is not established.** They hold this GGUF's IQ2_XXS/IQ1_M boundary
quants, but dividing the samples by bytes per type shows the CPU kernels cost about the same per byte:

| Type in layers 0–3 | Bytes | Samples | Samples per GiB |
| :--- | ---: | ---: | ---: |
| IQ1_S | 2.73 GiB | 19,501 | 7,132 |
| IQ2_XXS | 2.06 GiB | 17,698 | 8,581 |
| IQ1_M | 0.44 GiB | 3,952 | 9,033 |

Layers 0–3 hold 5.23 GiB of experts and the five IQ1_S layers hold 5.86 GiB, so neither the byte
count nor a 20–27% per-byte kernel difference explains a 2x decode gap. Not yet tested: whether
layers 0–3 route more tokens through their experts than the middle layers do, and whether a few
early IQ1_S layers (3, 6, 7, 9) on the CPU are fast or slow. The second test would separate
"early layer" from "boundary quant type".

Three IQ1_S layers is marginally faster, but it put enough on the GPU to cause 132K swapouts during
the sweep.

CPU threads matter: at `--n-cpu-moe 6`, tg128 was 10.3 / 11.3 / 12.9 / 14.0 / 16.6 / 14.4 / 12.6 / 4.1
tok/s at 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18 threads. At 18 threads the compute threads take every core
from the Metal driver. Eight IQ1_S layers (~49 GB on the GPU) fits the default Metal limit without `sudo`, at
23.7 tok/s (`PROFILE=nosudo`, which has not been checked in `llama-server`).

## What had to change for this to run on a Mac at all

1. **The patch only applies at upstream `d2f83055`.** Its 18 pre-image blobs match only between
   `d2f83055` and `dd1ea524`, both from 2026-08-10. On current master 8 files fail to apply.
2. **arm64 did not link.** `ggml_get_vec_dot_multi()` references `*_1x8/_1x4/_1x3/_1x2` kernels that
   exist only in `arch/x86/quants.c`.
3. **Q2_0 was silently wrong on ARM and Metal.** The patch redefines `block_q2_0` from 18 to 30 bytes
   but rewrites only the x86 kernels. On Metal, `test-backend-ops -o MUL_MAT -p q2_0` passed 0/45
   with the upstream kernels and passes 45/45 after the port (MUL_MAT_ID 75/75).
4. **Metal mapped the whole file.** With mmap, llama.cpp wraps one `[first, last]` span per file, and
   `output.weight` sits at byte 0 of shard 2, so every split mapped all 57.9 GB and hit
   `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Mapping each contiguous run of GPU tensors
   separately is what allows experts to be placed on the CPU at all.
5. **A NEON kernel for Fractal-64 Q2_0** doubled decode (4.2 → 8.4 tok/s) while whole layers were
   still on the CPU.

## What does not work

- **Speculative decoding is slower.** `ngram-mod` accepted 1,037/1,170 drafts on code_edit and still
  dropped it from 20.7 to 19.1 tok/s (code_new 17.6, free_text 15.8), with 41K swapouts. Verifying
  several tokens at once activates more experts per step, including the CPU-resident ones.
- **`llama-server` reports "model loaded" after a Metal out-of-memory error.** Output after that line
  is invalid. Check the log for `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
- **Copying weights into device buffers swaps.** The opt-in `LLAMA_GPU_COPY_WEIGHTS=1` held both the
  mmap pages and the copies: 4.95M swapouts, 45 GB of swap, and Metal OOM anyway.
- **Deep prefills can still swap.** The 8K/16K depth benchmark caused 70K swapouts.
