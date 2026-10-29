# Muse BF16 projection implementation report

Date: 2026-10-27, Europe/Moscow. Final scored job: `86e5f6c3e21d41eb`.

## Delivered result

The standalone [submission.py](../submission.py) computes every required projection and router output, including the sliced unpooler, with a measured **1.2202× speedup** over the pinned SGLang baseline on H200. The official 95% interval is **[1.2019, 1.2385]**. The normalized model-step score is **3958.89 µs**, versus **4830.80 µs**, a reduction of **18.05%**. These model-step times are not individual GEMM latencies.

All **9 preflight tests and 30 scored cases passed**. The previous cleaned check measured 1.2329×; the scored run measured 1.2202×, consistent with the reported uncertainty. The final artifact is byte-identical to [the submitted snapshot](../experiments/v15_final.py).

The leaderboard snapshot places this entry **582 of 1157** at 1.2202×. The leader was 1.6367×, so this is **not a first-place result**. Positions are provisional. This work does not claim a global optimum.

## Exact baseline paths

SGLang is pinned to `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. The generated `baseline.py` and the pinned source agree on these paths:

| Projection | M=1,8 | M=16,32,128,256 |
|---|---|---|
| qkv_gate | BF16 `_bf16_gemm_dispatch_impl` → `F.linear` → cuBLAS/cuBLASLt | Same |
| o | Same BF16 library path | Same |
| pooler | Same BF16 library path | Same |
| unpooler | Same BF16 library path, `W[:1024]` | Same library path, all 8192 output rows |
| router | FP32 `F.linear(x.float(), W)` | Two BF16 hi/lo GEMMs with FP32 outputs, then FP32 addition |

Hopper GEMV exists in SGLang but is an explicit backend, not the default H200 path in this challenge. Automatic CuTe DSL dispatch at this revision is for Blackwell. [Pinned dispatch source](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/unquant.py). More source findings and library eligibility details are in [research.md](research.md).

## Final implementation

The dispatcher selects kernels only from public shape/stride metadata. [dispatch.json](dispatch.json) records every scored case's backend and parameters. Tuple parameters for custom GEMMs are `(BM, BN, BK, split_K, warps, pipeline_stages)`; GEMV parameters are `(output_rows_per_CTA, warps)`.

- **One-token projections:** Triton FP32-reduction GEMV for qkv, o and sliced unpooler; SGLang Hopper GEMV for pooler. Qkv and unpooler use one output row per CTA to expose enough parallelism. The router reads its original FP32 weights and reduces in FP32.
- **Small BF16 batches:** retain cuBLAS when it matches or beats custom kernels. Use selected split-K Triton GEMMs for qkv at eight tokens, the pooler at eight tokens with KN weights, and the sliced unpooler at eight tokens. All partial outputs are FP32 and are reduced before the final BF16 cast.
- **Larger BF16 batches:** use native DeepGEMM SM90 for selected qkv/pooler/unpooler cases and cuBLAS for the remaining ones. Selected cuBLAS paths use an untimed contiguous KN weight layout.
- **Router above one token:** fuse BF16 hi/lo products in the same split-K Tensor Core kernel, using separate FP32 accumulators. A second kernel sums FP32 partials. BK128 and adequate K splitting substantially improve small batches; larger M uses BM64. The scored eight-token router improves from 26.27 to 6.43 µs, **4.08×**.
- **Other valid shapes:** BF16 projections use a general PyTorch GEMM fallback; the router applies the same masked hi/lo family at nearby M values. Column-strided inputs use the library path. The original weights are retained and the unpooler slices the first `n` rows exactly.

The native DeepGEMM path is deliberately enabled through its supported deterministic-algorithm API; the default package route otherwise selects cuBLASLt. Native scheduling is specialized for selected SM counts and tile-N multiples. Official profiles show `sm90_bf16_gemm_impl`, confirming actual native execution. [Pinned DeepGEMM package source](https://github.com/sgl-project/DeepGEMM/tree/e885b74a23eb511c09a0c440e74f257d0e6b681f).

Preparation took **16.11 seconds** in the final scored run, inside the 300-second budget. It retains **340 MiB** of weight tensors versus **188 MiB** for the originals: a **1.809×** storage ratio. Extra storage consists of three exact BF16 transposes and the allowed router hi/lo pair. This excludes JIT binaries, library caches and allocator reserves. The largest scored FP32 split-K workspace is **4 MiB**. Details are in [preparation.json](preparation.json).

Preparation warms selected library specializations with synthetic zero activations to compile them. No actual invocation input is available there, and no product, synthetic input or graph output is retained. The final submission contains no autotuning, network/process calls, reference imports, input-value host synchronization, output caching or benchmark-mode detection.

## Measurement and complete results

The primary measurements are official paired CUDA-graph replays on a single H200. The judge clears L2 before each replay, alternates candidate and SGLang order, uses median paired ratios, and removes its fixed graph-replay overhead. The score aggregates each load's five projection times with weight 62 and then takes a geometric mean across the six loads. It does not average the 30 case speedups directly.

| Load | Baseline step µs | Candidate step µs | Speedup |
|---|---:|---:|---:|
| decode 1 | 3698.18 | 2804.51 | 1.319× |
| decode 16 | 4724.90 | 4072.38 | 1.160× |
| decode 32 | 4848.90 | 4180.21 | 1.160× |
| verify 1 | 4662.40 | 3264.43 | 1.428× |
| verify 16 | 5333.98 | 4474.91 | 1.192× |
| verify 32 | 6324.99 | 5788.58 | 1.093× |

The complete shape-by-shape baseline latency, candidate latency, speedup, theoretical floor, correctness status, backend and parameters are in [benchmark.csv](benchmark.csv); a readable 30-row table is in [benchmark.md](benchmark.md). The complete official result, including raw timing samples, hardware state, correctness messages and evaluator image/source digests, is in [v15_result.json](v15_result.json). The GPU was NVIDIA H200 with SM clocks reported at 1980 MHz and memory clocks at 3201 MHz in the machine snapshot.

The source SHA-256 of the delivered file is:

`9f87ee5344c4507e9161cbe3784972f8cdfd0f9c90ace6d692cdfdd5deffe269`

The service's content identifier is separately retained in [final_summary.json](final_summary.json); it is not treated as the raw file SHA-256. The report generator verifies byte equality with the exact local submitted snapshot.

## Correctness and portability evidence

The official preflight covers M=1/7/33/130/5/8/9/512/1000, including normal/outlier inputs, router precision and exact-weight probe rows. All scored cases also pass reference checks. The router output is FP32 with a 1e-4 relative-L2 bound; BF16 outputs have a 1e-2 bound. Probe rows have their separate 1e-3 bound. We do not detect probes or alter their computation.

Local validation on **NVIDIA GeForce RTX 3080 Ti Laptop GPU**, PyTorch **2.11.0+cu128**, passed **74 CUDA-graph replay checks** across 37 scored/nearby/outlier cases. Each captured input buffer was overwritten with two independently generated inputs before replay, then compared with the FP64 reference. Additional checks passed for row-strided and column-strided activations, all-zero inputs and empty batches. See [local_final.json](local_final.json). Local timing is not used to rank H200 kernels. Optional SGLang/DeepGEMM paths are validated by the official H200 runs; the local machine tests their correct library fallbacks.

The final file depends only on packages supplied in the challenge image: Torch, Triton, SGLang and DeepGEMM. It does not import any development/report file. Optional native-library import failures have correct Torch/Triton fallbacks.

## What did not work and what remains

[optimization_history.md](optimization_history.md) explains every experiment, rejected approach, timing-tool repair and remaining priority. [experiments.csv](experiments.csv) summarizes official outcomes; [all_cases.csv](all_cases.csv) preserves all available per-case measurements. Native C++ cuBLASLt tuning, Tiny GEMM, alternative split-K/packed layouts, concatenated router GEMM, SIMT routing and exact BF16 packing were investigated. Passing candidates are preserved, including the best preparation-time searches.

The strongest remaining bottleneck is large BF16 projection throughput, especially the pooler/unpooler. At 256 tokens those two alone account for about 57% of candidate projection time; optimizing the now-fast router further has less overall impact. The most useful next experiment is a directly controlled skinny-M TMA/WGMMA scheduler, assessed against the retained native/library choices. A safe single-launch split-K reduction may remove another launch in narrow cases, but that has not been demonstrated here.

Nsight Compute was attempted but access to performance counters was denied (`ERR_NVGPUCTRPERM`); no interactive H200 shell is provided. Consequently no DRAM/L2 utilization or occupancy figures are invented. Official per-kernel profiles support the dispatch conclusions. Theoretical bandwidth/compute floors are useful lower bounds, not calibrated achievable performance.

## Reproduction

From the repository root, with a signed-in `okc` account:

```bash
okc check muse_bf16_proj/submission.py --json
okc status 86e5f6c3e21d41eb --json
```

Local correctness, using an environment with Torch/Triton and CUDA:

```bash
.venv/bin/python muse_bf16_proj/validate.py --full --output muse_bf16_proj/reports/local_final.json
```

Inside the challenge image on an H200, `benchmark.py` can compare the candidate and generated baseline with cold-cache event timing. Its cache flush and event placement differ from the official harness, so its results are diagnostics rather than a competition score:

```bash
python muse_bf16_proj/benchmark.py --output muse_bf16_proj/reports/local_benchmark.csv
```

To regenerate the tables and source identity from the saved official run:

```bash
python3 muse_bf16_proj/summarize_results.py --final muse_bf16_proj/reports/v15_result.json --snapshot muse_bf16_proj/experiments/v15_final.py
```
