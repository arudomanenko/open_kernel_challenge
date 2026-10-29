# Muse projections: CUDA implementation and results

Revision completed 2026-10-30, Europe/Moscow. Scored job: `c014ae47ef1e478b`.

## Outcome

The [standalone submission](../submission.py) now uses CUDA C++ for every custom
kernel. It contains no Triton import or Triton kernel. The readable
[kernels.cu](../kernels.cu) is mirrored inside the Python file, so the competition
submission does not read another project file at runtime.

The final scored run passes **9 preflight tests and 30 scored cases**, measuring
**1.2322×** over SGLang, with 95% interval
**[1.2138, 1.2507]**. Normalized model-step time
is **3920.31 µs**, versus **4830.80 µs**, a
**18.85%** reduction. These are aggregate model-step scores,
not individual GEMM latencies.

The prior scored Triton-based submission measured 1.2202× and the reviewed check
measured 1.2250×. Their intervals overlap the new result. The CUDA rewrite
preserves aggregate performance; the measurements do **not** establish a
statistically significant improvement over the previous implementation.

The refreshed leaderboard snapshot places this entry **701 of
1402**, at 1.2322×. The leader is **1.6498×**.
Another **33.89%** aggregate speed improvement would
be needed to match that snapshot. **This is not a winning result.** Positions
remain provisional. See [leaderboard.json](leaderboard.json).

## Production implementation

The final dispatcher is frozen from cold-cache H200 CUDA-graph experiments. It
branches on layer, dimensions, stride and alignment metadata only. There is no
runtime search, input inspection, output caching or diagnostic timing in the
production file.

| Regime | Selected implementation |
|---|---|
| One-token qkv/unpooler/router | CUDA vector prefetch, independent FP32 accumulators and block reduction |
| One-token output projection/pooler | CUDA shared-activation GEMV with streamed weight loads |
| Selected small BF16 batches | Reverse-orientation CUDA WGMMA, 128-byte shared-memory swizzle, double-buffered asynchronous copies and FP32 split-K reduction |
| Eight-token router | CUDA FP32 SIMD kernel reusing weight vectors across four tokens |
| Router M=16/32/128 | CUDA WGMMA hi/lo products with separate FP32 accumulators and FP32 reduction |
| Router M=256 | One BF16 cuBLAS GEMM for concatenated hi/lo output columns, then CUDA FP32 addition |
| Other scored BF16 cases | cuBLAS/cuBLASLt or native DeepGEMM CUDA where those measured better |
| Nearby shapes / unsupported layouts | General library fallback; nearby router shapes use the combined hi/lo family when eligible |

[dispatch.json](dispatch.json) records all 30 choices. WGMMA tuples are
`(token_tile, K_tile, K_splits)`; the broad 64-row WGMMA dimension covers weight
rows. GEMV tuples encode the measured prefetch or streaming specialization.
DeepGEMM SM-count and deterministic settings are saved and restored in `finally`;
a lock serializes configuration and dispatch among this module's callers.
Unrelated DeepGEMM callers changing the same settings concurrently still require
application-level coordination.

BF16 weights remain at their original values. Only the router uses the permitted
BF16 hi/lo representation. Hi and lo accumulate separately in FP32 before their
sum. Every split-K workspace is FP32; final casts occur after reduction. The
unpooler uses the first requested `n` weight rows, including its 1024-row slice.

## Validation

- Final official scored run: **9 preflight tests + 30 scored cases passed**;
  static checks passed with no flagged or rejected items.
- Frozen off-board check: **1.2246×**, interval [1.2063, 1.2430], all 39 cases
  passed. Its source is byte-identical to the scored submission.
- Separate H200 audit: **264** fresh-input graph replays over 33 shapes and four
  layouts passed. It includes normal/outlier inputs, weight-column probes,
  dense/aligned-padded/unaligned-padded/offset views, and configuration restoration.
  See [v33_result.json](v33_result.json) and its
  [development audit source](../experiments/v33_cuda_audit.py).
- Local RTX 3080 Ti Laptop GPU: **104** fresh-input graph checks
  over 52 cases, **38** selected-layout checks and
  extra zero/empty/column-stride cases passed. Hopper WGMMA/DeepGEMM fallbacks
  execute locally; actual native execution is independently audited on H200.
- Four backend regression tests pass: restore settings after success and failure,
  serialize concurrent configuration/launch, and bypass native kernels for
  unsupported layouts.

The accuracy contract remains the official FP64-reference bounds: BF16 relative
L2 1e-2, router 1e-4, probe rows 1e-3, and finite outputs. Official tests also
cover larger outlier batches. No special treatment of actual probe rows exists
in production.

## Official measurements

Official paired CUDA-graph replays start with cold L2, alternate execution order,
use median paired ratios and subtract fixed graph-replay overhead. Each load
aggregates five projections with weight 62; the final speedup is a geometric
mean over six loads.

| Load | Baseline step µs | CUDA step µs | Speedup |
|---|---:|---:|---:|
| decode 1 | 3685.28 | 2624.60 | 1.404× |
| decode 16 | 4628.67 | 3928.13 | 1.178× |
| decode 32 | 4769.54 | 4102.07 | 1.163× |
| verify 1 | 4699.10 | 3225.40 | 1.457× |
| verify 16 | 5292.32 | 4499.41 | 1.176× |
| verify 32 | 6228.77 | 5865.21 | 1.062× |

All 30 per-case baseline/candidate latencies, speedups, theoretical floors and
backends are in [benchmark.csv](benchmark.csv) and [benchmark.md](benchmark.md).
The full scored response is [v34_result.json](v34_result.json), with identity,
machine and evaluator metadata in [final_summary.json](final_summary.json).

## Experiments and limitations

Fifteen off-board checks and one scored run covered the CUDA rewrite. Two checks
failed compilation because the initial prefetched GEMV declared zero-sized
arrays at K=1024 with oversized thread counts; the corrected implementation
rounds up the vector count and masks inactive threads. Local per-layer checks
passed before the corrected combined search.

[optimization_history.md](optimization_history.md) records every version,
including negative results. [cuda_tuning.json](cuda_tuning.json) preserves the
available diagnostic top candidates; their event overhead differs from the
official harness and those values are not official case latencies.

Measured approaches include shared-memory WMMA, register MMA, safe CTA-cluster
split-K reduction, reversed and standard CUTLASS tiles, K splitting, ping-pong
schedules, multicast clusters, streamed and prefetched GEMV, exact BF16 packing,
SIMD routing, concatenated hi/lo GEMM, WGMMA orientation, shared-memory swizzling
and double buffering. The first port measured only 0.8199×. The best search
measured 1.2359×; freezing it measured 1.2246× and scoring it measured 1.2322×,
consistent with measurement uncertainty.

No cluster reduction, weight compression, preparation search or CUTLASS header
dependency remains in production. Native cuBLAS and optional DeepGEMM still win
many BF16 cases. The tested custom large-BF16 kernels did not establish a better
path, so no broad performance win is claimed for rewriting in CUDA.

Retained weight tensors occupy **384 MiB**, compared with
**188 MiB** originally and **324 MiB** in the reviewed Triton revision. The CUDA
artifact retains four exact BF16 transposes plus router hi/lo and concatenated
weights: a **2.043×** storage ratio. This is an explicit memory
tradeoff. Maximum scored split-K workspace is **4 MiB**; selected dynamic shared
memory reaches **160 KiB per CTA**. Preparation took **13.44 s**,
inside the 300-second limit. These figures exclude JIT binaries and allocator
reserves. See [preparation.json](preparation.json).

Large BF16 pooler/unpooler throughput remains the main opportunity. Future work
needs a demonstrably better large-BF16 scheduler/data path, or a lossless decode
path whose bandwidth savings exceed decode cost. The tested options did not
prove one. Nsight Compute hardware counters remain unavailable; occupancy,
DRAM utilization and stall measurements are not claimed.

## Reproduction and source identity

The competition artifact uses Torch's supported CUDA extension loader during
preparation, plus Torch/CUDA and optional installed DeepGEMM. The readable CUDA
file is for review; its contents are embedded in the standalone submission.
Local builds need a supported CUDA compiler, C++ compiler and Ninja on PATH.
Use `sync_cuda.py --write` after editing the readable CUDA file, then revalidate
and resubmit the changed artifact; running it without flags checks the mirror.

```bash
source .venv/bin/activate
python muse_bf16_proj/test_backend.py
python muse_bf16_proj/validate.py --full --output /tmp/muse-validation.json
okc status c014ae47ef1e478b --json
okc check muse_bf16_proj/submission.py --json
python muse_bf16_proj/summarize_results.py --final muse_bf16_proj/reports/v34_result.json --snapshot muse_bf16_proj/experiments/v32_cuda_frozen.py
```

Source SHA-256:
`67982fe239b900bdaac86823ee95e4af888968e38158d6a7565ae75cc5716686`.

The [frozen source snapshot](../experiments/v32_cuda_frozen.py) is byte-identical
to the submitted file. The service content identifier is retained separately
and is not treated as the raw file SHA. [research.md](research.md) cites the pinned
baseline and CUDA/CUTLASS sources. [v15_report.md](v15_report.md) and
[v18_report.md](v18_report.md) preserve the previous implementation reports.
