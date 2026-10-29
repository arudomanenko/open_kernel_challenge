# Optimization history

Date: 2026-10-27. All performance decisions use official H200 checks, not RTX 3080 Ti timings. Raw official results are `vNN_result.json`; complete per-case comparisons are in `all_cases.csv`. Recoverable Python candidates and official kernel profiles are under `../experiments/`. These experiments do not establish a global optimum or a first-place solution.

## Experiments

| Version | Experiment | Finding / disposition |
|---|---|---|
| v00 | Unmodified PyTorch starter | Passed. 0.907× overall. Its FP32 router above eight tokens is slower than the baseline's two BF16 products. Establishes why the starter is not equivalent in speed to SGLang. |
| v01 | Triton GEMV; raw NK split-K BF16 GEMM; fused router hi/lo dots | Passed. 0.976× overall. Router improves, but BK64 split-K BF16 loses on small batches. Retain numerical method and reduction, reject this universal dispatch. |
| v02 | SGLang Hopper GEMV; contiguous KN weights for cuBLAS; different large-router tile | Passed. 1.052× overall. GEMV and some transposed projections improve; the larger router tile is a substantial regression. |
| v03 | Explicit cuBLASLt heuristic search in a C++ extension | Official execution passed, but the preparation tuner incorrectly used internal graph events for elapsed-time queries. Every plan was skipped, leaving a PyTorch fallback. This result does **not** evaluate successful cuBLASLt algorithm tuning. |
| v04 | Single-launch packed KN BF16 GEMM; concatenated router weights; GEMV with more CTAs | Passed. 0.841× overall. Small-router packing and BN1 GEMV help. Universal single-launch BF16 GEMM and the larger-router configuration regress badly. |
| v05 | Preparation-time search over Torch, SGLang, DeepGEMM and Triton | Setup failed: same CUDA event-capture mistake as v03. No performance conclusion is drawn from this run. |
| v06 | SGLang tiny GEMM with explicit N split=4, max M=8 | Passed. 1.043× overall. At eight tokens, qkv and pooler are about 1.8× slower than the baseline. Reject tiny GEMM for these workloads. |
| v07 | Lossless blocked `[N/64,K,64]` BF16 layout; expanded-router library GEMM | Passed. 0.936× overall. Blocked small-pooler GEMM gains about 10%; large BF16 cases regress. Router activation concatenation plus one FP32-output BF16 GEMM is better than the earlier large-router tiling. |
| v08 | Corrected CUDA-graph event search with `external=True` | Passed. 1.209× overall. Shape-specific dispatch avoids broad regressions. Router gains across every batch size; qkv and sliced unpooler gain from BN1 GEMV. Default DeepGEMM selects cuBLASLt here. |
| v09 | Corrected cuBLASLt heuristic search, both NK/KN layouts, 32 heuristics/layout, 32 MiB workspace | Passed. 1.052× overall with the deliberately unchanged, slow large-router path. BF16 gains are generally small; no broad new advantage justifies retaining the extension in the final submission. |
| v10 | Force native DeepGEMM SM90 with its deterministic-algorithm API; broaden router/blocked-layout search | Passed. 1.225× overall. Native DeepGEMM benefits the 128-token pooler and some smaller pooler cases. Profiles verify `sm90_bf16_gemm_impl`, unlike default DeepGEMM. |
| v11 | Exact 12-bit BF16 exponent/mantissa packing with escape reads from original weights | Local full correctness passed. Official preflight passed, but scored GEMM compilation failed on Hopper: Triton attempted an unsupported i1 shared allocation in the pipelined masked escape load. GEMV was also roughly twice as slow on qkv/pooler. Reject the approach in this implementation. |
| v12 | Combined search, including exact packing and FP32 SIMT router variants | Passed. 1.220× overall. Packing and additional SIMT router variants were not selected. This difference from v10 is within the official uncertainty interval, so it is not a meaningful regression or improvement. |
| v13 | Native DeepGEMM tile-N constraints and active-SM count search | Passed. 1.231× overall. Targeted scheduling gives further modest gains; retained in the frozen dispatcher. |
| v14 | Frozen dispatcher, removed search and unused layouts | Passed. 1.233× in the final check. Preserves selected kernels with much shorter preparation and fewer retained buffers. |
| v15 | Final source cleanup and scored submission | Passed. 1.220× overall, 95% interval [1.202, 1.239]. Final scored artifact; all 9 preflight and 30 scored cases pass. |

Candidate parameters are retained directly in each source file. Raw NK and packed/blocked GEMMs vary BM=16/32/64, BN=16/32/64/128, BK=64/128/256, K splits=1/2/4/8/16/32, four/eight warps, and two/three pipeline stages; only combinations actually listed in the candidate generators were run. The router uses separate FP32 hi/lo accumulators and an FP32 partial-sum reduction. Large BF16 paths include native WGMMA/TMA from DeepGEMM and cuBLAS/cuBLASLt. Persistent scheduling was evaluated through those library implementations rather than a new standalone persistent kernel. Stream-K, a custom single-launch last-arriver split-K reduction, TileLang and ThunderKittens were researched but not independently implemented or benchmarked; no claim of their performance is made.

## Numerical and preparation details

Preparation-time tuning uses newly generated synthetic activations solely to select kernels. It retains weights, layouts and public-shape algorithm metadata. Synthetic inputs, output values and their captured graphs are not retained. The final submission freezes measured choices and removes the search. It does not cache real-input products or compute real-input work in preparation.

The router's allowed BF16 split is `hi=W.bfloat16()` and `lo=(W-hi.float()).bfloat16()`. Both dots accumulate in FP32. Test errors for this family are typically around a few parts per million against FP64, comfortably inside the 1e-4 router limit. Probe rows use the same path, with no probe detection or special treatment.

The rejected lossless representation stores one mantissa/sign byte per value and one exponent nibble per value. Escape code 15 loads the original BF16 word; it is not a quantized approximation. `lossless_audit.json` records exact round trips for all 65,536 BF16 bit patterns and all four independently generated projection weights. The packed buffers are 75% of the original weight size, but originals remain allocated for escapes, so total storage increases. In-kernel unpacking and escape handling erased the bandwidth benefit. Passing on Ampere did not predict successful Hopper compilation.

## Profiling limits

Official profiles identify kernel names and GPU durations for every scored case. Offline compilation with local Triton 3.6 targeting SM90 confirmed WGMMA instructions in the BM64 raw GEMM, so its poor performance cannot be explained simply as an absence of Tensor Core instructions. This is a compiler inspection, not an H200 hardware-counter measurement.

Nsight Compute was attempted locally but returned `ERR_NVGPUCTRPERM`; performance counters are not accessible. No interactive H200 host or remote shell is supplied by `okc check`, and submissions may not launch external processes. Consequently no measured DRAM utilization, L2 hit rate, occupancy or stall counters are claimed. The official profiler and cold-cache timings guide the conclusions.

## Remaining priorities

1. Focus on the large BF16 pooler/unpooler. They dominate load totals, while router launch/reduction improvements now affect a much smaller share of step time. Compare native scheduling and layout choices with the retained cuBLAS paths before rewriting them.
2. Investigate a Hopper TMA/WGMMA implementation with an explicit skinny-M scheduler and tile configuration. General library heuristics leave measurable room relative to the workload's theoretical bounds, but those bounds are optimistic and are not a measured attainable roofline.
3. A safe single-launch split-K final reduction could remove a launch for the router and sliced unpooler. This is a proposed experiment, not an observed improvement; global spin synchronization is not acceptable.
4. Further lossless packing would need efficient vectorized unpacking and a cheaper escape strategy. The measured implementation is a negative result, so do not deploy it merely because its byte count is smaller.
5. Rerun close dispatch decisions on a controlled H200. Differences of one or two percent are inside or near the uncertainty interval; ranking them by one preparation timing is unreliable.


## Review follow-up: v16–v18

The initial implementation and reports were committed and pushed first as
`4a6bcd4` on `codex/muse-bf16-proj`, as requested.

- **v16 — scoped native backend:** Save and restore DeepGEMM SM-count and
  deterministic settings around each launch in a `finally` block. Serialize
  configuration and launch among this module's callers. Remove block-size
  mutation because the pinned public interface has no matching getter. Guard
  native dispatch with dense-layout and 16-byte pointer-alignment checks.
  Remove the output-projection transpose, reducing retained storage by 16 MiB.
  H200 check `25c2d00e3c2e4228` passed all 39 official cases at 1.21898×,
  statistically indistinguishable from the previous 1.22024× scored result.
- **v17 — native layout and neighborhood audit:** Development-only synthetic
  checks in preparation exercised every native-selected shape with dense,
  padded-row and offset-base inputs; all 21 graph-replay checks passed, with
  DeepGEMM configuration restored after preparation and calls. Diagnostic
  cold-cache CUDA-graph timing compared nearby shapes. It showed small pooler
  gains for packed M=4/7 and native M=15/17, but large native regressions at
  M=33/129. The latter routes were rejected. Qkv and sliced-unpooler gains were
  too small to justify extending their specialized ranges. See
  [nearby_v17.json](nearby_v17.json) for raw diagnostic times, which include
  different event overhead from the official benchmark. The audit job
  `2cd2938cbe074b01` also passed all official cases at 1.21798×.
- **v18 — final bounded dispatch:** Extend pooler packed execution to M=2–8,
  native execution to M=9–32, and unpooler transposed-library execution to
  M=9–32. Preserve exact scored selections. Retain the generic library fallback
  beyond these ranges. The cleaned production source has no diagnostic timing
  or audit code. See the current [final report](final_report.md) for the final
  official result.

Four host-side regression tests cover settings restoration after success and
failure, serialization of concurrent calls, and native-layout fallback. Expanded
local validation passes 104 fresh-input CUDA-graph checks over 52 cases plus 14
native-selected layout checks. The local GPU cannot exercise DeepGEMM itself;
the v17 H200 audit supplies that separate evidence.

The lock only coordinates callers using this module. Unrelated DeepGEMM users
that mutate shared configuration concurrently still require application-level
coordination. No claim of a new aggregate speed improvement follows from these
changes: demonstrated benefits are lower memory, settings restoration, layout
handling and bounded nearby-shape dispatch. Direct TMA/WGMMA scheduler development
and single-launch split-K remain future experiments.
