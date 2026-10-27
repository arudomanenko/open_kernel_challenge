# Research: Muse BF16 projections on H200

Research date: 2026-10-27. This report separates verified source behavior from unmeasured implementation hypotheses. Performance measurements and final dispatch decisions belong to the experiment/benchmark reports; no external benchmark numbers are adopted as challenge results.

## Authoritative workload

The local `TASK.md`, `baseline.py`, `reference.py`, and `checks.py` were inspected. The output is `X @ W[:n].T`, with BF16 activations, BF16 projection weights/output, and FP32 router weights/output. The unpooler computes only the first 1024 weight rows for `m <= 8`. Scored token counts are 1, 8, 16, 32, 128, and 256; preflight checks additionally include 5, 7, 9, 33, 130, 512, and 1000 tokens. A public-metadata dispatcher therefore needs a general fallback.

The correctness contract makes weight precision substantive: probe activations expose individual weight columns. Router hi/lo BF16 splitting is explicitly allowed, while BF16-only router weights are insufficient. Timing uses CUDA graph replay with cold L2 before each replay. Measurements on an Ampere desktop GPU or with hot weights cannot decide H200 dispatch. These are local challenge facts, not inferred from general GEMM documentation.

## Exact SGLang baseline

At commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, `_bf16_gemm_dispatch_impl` tries configured split-K, explicitly enabled Hopper GEMV, and configured CuTe DSL paths, before falling back to `F.linear`. Initialization enables automatic CuTe DSL only on SM100. Its explicit CuTe DSL backend rejects non-SM100/SM103 devices. Thus the challenge baseline's stated cuBLAS path on H200 is consistent with the pinned source; the existence of a Hopper kernel does not mean that kernel is used by the baseline. [Pinned unquantized dispatch](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/unquant.py).

The router builds `hi = W.to(bfloat16)` and `lo = (W - hi.float()).to(bfloat16)`, stores transposed contiguous versions, and computes two BF16 matrix products with FP32 outputs followed by addition. The challenge uses native FP32 linear for up to eight tokens and this split above eight. Fusing the two products in one kernel is mathematically compatible with this representation, but numerical and timing validation remain necessary. [Pinned Muse router](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L256).

## Existing kernels worth testing

### SGLang Hopper GEMV

Direct entry point: `sglang.kernels.ops.gemm.hopper_bf16_gemv.hopper_bf16_gemv(x, w)`. The support predicate requires one activation row, K divisible by 512, K between 512 and 17408, N divisible by eight, and supported N ranges. All four BF16 challenge shapes at `m=1`, including the sliced unpooler, qualify. The Python wrapper allocates BF16 output and uses the SGLang JIT loader. For N below 8192 it selects one output row per warp; otherwise two, with eight warps per block and unroll two. [Pinned wrapper](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py).

The CUDA implementation stages the activation in shared memory, streams 16-byte weight vectors using `__ldcs`, accumulates scalar FMAs in FP32, reduces with warp shuffles, and stores BF16 output. It uses one kernel without a split-K reduction. This is a useful minimal implementation model if the installed Python module is absent. It is not directly suitable for FP32 router weights or multiple activation rows. [Pinned CUDA source](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/hopper_bf16_gemv.cuh).

### SGLang tiny GEMM

Direct API: `tiny_gemm_bf16(x, w, out=None, out_dtype=None, n_split=None, max_m=16)` in `sglang.kernels.ops.gemm.tiny_gemm`; test `can_use_tiny_gemm(n, k, max_m)` first. It chooses the N variant when N<K and the K variant otherwise. The N variant accepts challenge qkv, pooler, sliced unpooler, and BF16 router halves with small M. The K variant's warp-lane condition rejects the challenge output projection and full unpooler (their K/8 exceeds a warp). Explicit `n_split` controls N grouping; default aims for a wave of CTAs. [Pinned tiny GEMM API](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/gemm/tiny_gemm.py).

### DeepGEMM

Current upstream exports `bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk", ...)`, matching `[M,K] @ [N,K].T`, with BF16 operands and BF16 or FP32 output. The implementation dispatches to SM90/SM100 and may use cuBLASLt when its heuristic permits. Consequently, merely calling the function does not establish that a novel DeepGEMM kernel runs. The installed challenge revision must be probed before relying on the API. A useful experiment is a preallocated `d`, plus a router second product accumulated through `c`, if supported. [Current API source](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/gemm.hpp), [upstream BF16 tests](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_bf16.py).

Current upstream declares SM90/SM100, CUDA toolkit 12.9+, PyTorch 2.3+, C++20 format support, and CUTLASS 4.0+; kernels compile at runtime. These are current upstream requirements, not proof of what the challenge image installs. The user supplied image description lists DeepGEMM, but its exact version/API remains a remote validation question. [DeepGEMM requirements](https://github.com/deepseek-ai/DeepGEMM#requirements).

### CUTLASS, TileLang, ThunderKittens

CUTLASS's Hopper collective-builder example provides a conventional route to BF16 Tensor Core GEMM with SM90 scheduling and pipeline choices. It is most relevant to larger M; a single Python submission can compile an embedded extension only if the necessary headers and compiler are already available. Header discovery and compilation must fit the challenge preparation and total job budgets. [Official Hopper example](https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu).

TileLang offers GEMM examples and ThunderKittens offers tile primitives and example kernels. Both are reasonable adaptation sources when already installed. Their existence alone is not a reason to port a complete framework into the submission; direct SGLang kernels or a compact Triton/CUDA implementation have a smaller integration surface. Neither framework was benchmarked in this source-research task. [TileLang repository](https://github.com/tile-ai/tilelang), [ThunderKittens repository](https://github.com/HazyResearch/ThunderKittens).

## Algorithmic experiment priorities

The following are hypotheses and proposed experiments, not measured conclusions:

| Regime | First candidates | Reason and tradeoff |
|---|---|---|
| BF16 M=1 | SGLang Hopper GEMV; custom vectorized GEMV; cuBLAS fallback | Streaming the weight once is the primary goal; avoid extra launch/reduction traffic unless measured worthwhile. |
| BF16 M=8/16/32 | Small-M Tensor Core GEMM; small split-K; supported tiny GEMM | Enough reuse to benefit from MMA, but output tile grids may be too small for full device utilization. |
| BF16 M=128/256 and larger | cuBLAS; DeepGEMM; conventional Hopper Tensor Core GEMM | More output tiles and activation reuse; compare before paying implementation complexity. |
| Router small M | Direct FP32 SIMT reduction; fused hi/lo BF16 MMA | Direct FP32 preserves weight precision; split representation permits Tensor Cores and launch fusion. |
| Router large M | Fused hi/lo GEMM; baseline two-product FP32 accumulation | Narrow N=256 can limit CTA parallelism, making K splitting attractive. |

CUTLASS documents split-K as a way to expose parallelism when M/N tiling yields too few thread blocks, at the cost of a reduction and workspace. This motivates testing it on small M rather than assuming universal benefit. [Official GEMM optimization discussion](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html). Hopper tuning guidance emphasizes coalescing, avoiding redundant global loads, and enough active work; these support the streaming/reuse priorities above. [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

Keep router high and low partial sums in separate FP32 accumulators when practical, then add, rather than folding a tiny correction repeatedly into a much larger running sum. For split-K, write FP32 partial outputs and reduce to the final dtype. These are numerical design suggestions requiring checks against FP64 and the challenge probe rows.

## Environment and evidence limitations

The local project `.venv` had no torch, Triton, SGLang, DeepGEMM, CUTLASS Python package, or FlashInfer at the time of this research probe. This says nothing about the H200 judge image. Upstream source availability also does not guarantee that the corresponding source/JIT dependencies ship in that image. Therefore first remotely probe imports/version information and compile availability, then retain a correct fallback for every optional route.

No source-research result establishes a latency improvement. Use only measured cold-L2 H200 CUDA-graph results to select dispatch, and preserve passing candidates. The official checks remain authoritative for correctness and the scored geometric mean.

## Follow-up: force native DeepGEMM SM90 BF16

The remote experiment found that default `bf16_gemm_nt` ran cuBLASLt on every scored shape. The pinned SGLang dependency is `sgl-deep-gemm==0.2.1.post1`; its fork tag resolves to `e885b74a23eb511c09a0c440e74f257d0e6b681f`. [Pinned dependency manifest](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/pyproject.toml), [fork release tag](https://github.com/sgl-project/DeepGEMM/tree/e885b74a23eb511c09a0c440e74f257d0e6b681f).

That exact version exports `use_deterministic_algorithms` and `get_deterministic_algorithms`. Its BF16 native dispatch is reached when deterministic algorithms are enabled: the Lt branch is guarded by `not get_deterministic_algorithms()` and Lt availability. On SM90, native execution rejects a non-None `alpha`; omit it. Inputs are BF16; output can be BF16 or FP32. This is an ordinary public library configuration, independent of PyTorch's deterministic setting. [Pinned Python exports](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/deep_gemm/__init__.py), [pinned BF16 dispatch](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/csrc/apis/gemm.hpp#L584).

SGLang's `gemm_nt_bf16bf16f32` wrapper only adds its compilation execution hook before calling `deep_gemm.bf16_gemm_nt`; it does not independently select native SM90. Calling this wrapper therefore does not avoid Lt. [Pinned SGLang wrapper](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py#L266).

A minimal standalone experiment, after creating real BF16 x/w and preallocated d, is:

```python
old_deterministic = deep_gemm.get_deterministic_algorithms()
deep_gemm.use_deterministic_algorithms(True)
try:
    deep_gemm.bf16_gemm_nt(x, w, d)  # warm up / compile first
    # Validate and measure graph replay here; profile kernel names.
finally:
    deep_gemm.use_deterministic_algorithms(old_deterministic)
```

For a candidate submission, configuration can instead be fixed once in `prepare()`, and only selected shapes call DeepGEMM. First test the otherwise-unimproved BF16 M=128/256 cases, retaining the established paths elsewhere. Confirm that the profile actually names `sm90_bf16_gemm`, rather than a cuBLAS kernel, before interpreting the experiment.

The pinned normal SM90 BF16 path has no explicit M>=64 restriction. Its heuristics offer block-M16 for M<=16 and block-M32 for M<=32, alongside 64/128; BF16 output also permits block-M256. Thus small M is intentionally represented in native tiling, although this does not promise speed. The normal path constructs TMA descriptors and persistent launches; do not apply stricter grouped-GEMM-only assertions to it. [Pinned SM90 heuristics](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/csrc/jit_kernels/heuristics/sm90.hpp#L18), [pinned normal SM90 implementation](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp#L77).

TMA tensor bases and noncontiguous byte strides require 16-byte alignment; the challenge's contiguous K/N dimensions meet that ordinary layout requirement. As a second option, reducing `set_num_sms` below the hardware total makes `is_cublaslt_available()` false, also bypassing Lt. Prefer the deterministic switch initially because reducing SM count changes the performance experiment. No documented force-native environment variable was found in the inspected pinned runtime; `DG_USE_PYTORCH_CUBLASLT_HANDLE` and `DG_USE_TEMP_CUBLASLT_WORKSPACE` configure Lt resources rather than disabling dispatch. [Pinned runtime](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/csrc/runtime/runtime.hpp), [TMA utilities](https://github.com/sgl-project/DeepGEMM/blob/e885b74a23eb511c09a0c440e74f257d0e6b681f/csrc/jit_kernels/impls/runtime_utils.hpp).
