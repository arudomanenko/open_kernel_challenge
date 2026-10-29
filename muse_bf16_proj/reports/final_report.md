# Muse BF16 projections: reviewed implementation

Date: 2026-10-27. Final H200 check: `cbbd6faa6ca74248`.

## Result

The current [submission](../submission.py) passes all **9 preflight tests and
30 scored-shape checks**, measuring **1.2250×** over SGLang. Its 95%
interval is **[1.2067, 1.2434]**. The normalized
model-step time is **3943.41 µs** versus **4830.80 µs**.

The previous scored artifact measured 1.2202×. The new result is statistically
consistent with that measurement; this is **not evidence of a significant new
aggregate speedup**. This final run is an off-board check, not a new leaderboard
submission. The prior scored submission remains unchanged on the leaderboard.

The initial implementation and reports were pushed first as `4a6bcd4` on
`codex/muse-bf16-proj`. The review follow-up preserves measured performance while
reducing retained weight storage from **340 MiB to 324 MiB** and improving backend
isolation, layout handling and nearby-shape dispatch.

## Changes after review

- DeepGEMM SM-count and deterministic settings are read before dispatch and
  restored in `finally`, including when dispatch raises. A module lock serializes
  configuration and launch among this module's callers. Preparation no longer
  leaves deterministic mode enabled globally.
- Block-size configuration is no longer changed: the pinned public interface has
  no matching getter with which to restore an arbitrary caller's previous value.
- Native dispatch requires dense, 16-byte-aligned activations and aligned weights.
  Unsupported views use the general library path without copying activations.
- Output projection uses original weights instead of retaining a separate 16 MiB
  transpose. The remaining exact BF16 transposes are pooler and unpooler.
- Nearby pooler batches use the packed kernel for M=2–8 and native DeepGEMM for
  M=9–32; nearby full unpooler batches use the transposed library path for M=9–32.
  Exact scored selections remain in the measured tactic table. Larger nearby
  pooler shapes retain the general fallback: native M=33 and M=129 regressed in
  the diagnostic experiment.

The underlying arithmetic remains unchanged: original BF16 weights, FP32
accumulation/reduction, and the permitted router BF16 hi/lo pair. One-token
router execution reads original FP32 weights. Outputs have the required BF16 or
FP32 dtype, and unpooler slicing uses the requested number of weight rows.

## Validation and measurements

| Evidence | Result |
|---|---|
| Final official H200 check | 9 preflight + 30 scored cases passed |
| H200 native-path audit | 21 dense/padded/offset graph checks passed |
| Local fresh-input CUDA-graph checks | 104 checks over 52 cases passed |
| Additional local native-selected layout checks | 14 passed |
| Backend regression tests | 4 passed: success, exception, concurrency, layout fallback |
| Static submission checks | Passed, no flagged or rejected items |

The native audit also confirms that preparation and each tested call restore
DeepGEMM settings. Local testing uses an RTX 3080 Ti Laptop GPU and therefore
covers optional-backend fallbacks; native execution is separately covered by
H200. The official profiles retain native `sm90_bf16_gemm_impl` kernels.

Official results use paired cold-L2 CUDA-graph replay timing. The overall score
is the geometric mean of six load speedups, each aggregating five projections.
[benchmark.csv](benchmark.csv) and [benchmark.md](benchmark.md) contain all 30
case results; [v18_result.json](v18_result.json) preserves the full official
response. [final_summary.json](final_summary.json) records identity and result
metadata.

The separate [nearby-shape experiment](nearby_v17.json) uses cold-cache graph
CUDA events. Its event placement and overhead differ from the official harness,
so these timings are diagnostic comparisons, not official per-case scores.
Representative observations:

| Shape | Original library µs | Selected alternative µs |
|---|---:|---:|
| pooler M=4 | 29.95 | 28.64 packed |
| pooler M=7 | 30.18 | 28.93 packed |
| pooler M=15 | 30.11 | 28.54 native |
| pooler M=17 | 30.37 | 28.90 native |
| unpooler M=17 | 28.32 | 27.23 transposed library |

Native pooler M=33 took 37.73 µs versus 30.88 µs for the original library;
M=129 took 46.40 µs versus 31.33 µs. These regressions are why the new native
range stops at 32. Small qkv and sliced-unpooler differences were insufficient
to justify extending their specialized ranges.

## Storage, preparation and limitations

Retained tensors total **324 MiB**, compared with **188 MiB** of original weights
(1.7234×). Extra storage is **136 MiB**. The largest scored FP32 split-K workspace
is **4 MiB**. Preparation took **19.37 seconds**, within the
300-second budget. See [preparation.json](preparation.json).

The lock coordinates only callers of this module. Unrelated DeepGEMM users
changing shared settings concurrently still need application-level coordination.
This change does not make the underlying library configuration per-instance.

There is no new custom TMA/WGMMA scheduler or single-launch split-K implementation
in this revision. Those remain possible performance experiments. Pooler/unpooler
throughput remains the largest aggregate opportunity. Hardware performance
counters remain unavailable; no occupancy or DRAM-utilization measurements are
claimed. Theoretical floors are lower bounds rather than achievable targets.

## Artifacts and reproduction

- [Optimization history](optimization_history.md): all versions and rejected experiments.
- [Research](research.md): pinned primary sources and backend findings.
- [Original scored report](v15_report.md): historical v15 results and original research context.
- [Local validation](local_v18.json): expanded fresh-input and layout checks.
- [Native audit source](../experiments/v17_native_audit.py): reproducible development diagnostics.
- [Final source snapshot](../experiments/v18_final.py): byte-identical to the checked submission.

```bash
okc status cbbd6faa6ca74248 --json
okc check muse_bf16_proj/submission.py --json
.venv/bin/python muse_bf16_proj/test_backend.py
.venv/bin/python muse_bf16_proj/validate.py --full --output /tmp/muse-validation.json
python3 muse_bf16_proj/summarize_results.py --final muse_bf16_proj/reports/v18_result.json --snapshot muse_bf16_proj/experiments/v18_final.py
```

Current source SHA-256:
`98c913426e8eaa9c81cac7b3daa18be19d303185d5cd52f89860400810f66d4b`.
The service content identifier is recorded separately and is not treated as the
raw source hash. Diagnostic code is confined to experiment files; the production
submission performs no timing, input inspection, reference imports or output caching.
