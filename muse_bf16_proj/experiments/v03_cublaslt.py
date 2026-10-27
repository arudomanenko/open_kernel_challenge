# okc: problem=muse_bf16_proj
"""BF16 projections with split-K tensor cores and an FP32 router."""
from typing import TypedDict

import torch
import triton as tr
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


@tr.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    val = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + n, val, n < N)


@tr.jit
def _split_gemm(X, W, WL, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                SX: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, SPLIT: tl.constexpr, ROUTER: tl.constexpr):
    mi, ni, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    low = tl.zeros((BM, BN), tl.float32)
    for it in range(tr.cdiv(K, BK * SPLIT)):
        k = (it * SPLIT + si) * BK + r
        a = tl.load(X + m[:, None] * SX + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(W + n[None, :] * K + k[:, None],
                    (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
        if ROUTER:
            bl = tl.load(WL + n[None, :] * K + k[:, None],
                         (n[None, :] < N) & (k[:, None] < K), 0)
            low = tl.dot(a, bl, low)
    tl.store(P + si * M * N + m[:, None] * N + n[None, :], acc + low,
             (m[:, None] < M) & (n[None, :] < N))


@tr.jit
def _finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SPLIT)
    v = tl.load(P + s[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < SIZE)


_LT_SOURCE = '#include <torch/extension.h>\n#include <c10/cuda/CUDAStream.h>\n#include <cublasLt.h>\n#include <vector>\n#include <stdexcept>\n\nstatic void ck(cublasStatus_t s) {\n  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLt error ", int(s));\n}\nstruct Plan {\n  cublasLtHandle_t handle;\n  cublasLtMatmulDesc_t desc;\n  cublasLtMatrixLayout_t a, b, c;\n  std::vector<cublasLtMatmulHeuristicResult_t> algos;\n  Plan(int m, int n, int k, bool packed) {\n    ck(cublasLtCreate(&handle));\n    ck(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));\n    cublasOperation_t op = packed ? CUBLAS_OP_N : CUBLAS_OP_T;\n    ck(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));\n    ck(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, packed ? n : k, packed ? k : n, packed ? n : k));\n    ck(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));\n    ck(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));\n    cublasLtMatmulPreference_t pref;\n    ck(cublasLtMatmulPreferenceCreate(&pref));\n    size_t bytes = 32 * 1024 * 1024;\n    ck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));\n    algos.resize(32);\n    int count = 0;\n    ck(cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, pref, 32, algos.data(), &count));\n    algos.resize(count);\n    ck(cublasLtMatmulPreferenceDestroy(pref));\n  }\n  ~Plan() {\n    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b);\n    cublasLtMatrixLayoutDestroy(c); cublasLtMatmulDescDestroy(desc);\n    cublasLtDestroy(handle);\n  }\n  int count() const { return algos.size(); }\n  void run(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor ws, int index) {\n    float alpha = 1.f, beta = 0.f;\n    ck(cublasLtMatmul(handle, desc, &alpha, w.data_ptr(), a, x.data_ptr(), b,\n        &beta, y.data_ptr(), c, y.data_ptr(), c, &algos.at(index).algo,\n        ws.data_ptr(), ws.numel(), c10::cuda::getCurrentCUDAStream()));\n  }\n};\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  pybind11::class_<Plan>(m, "Plan").def(pybind11::init<int,int,int,bool>())\n    .def("count", &Plan::count).def("run", &Plan::run);\n}\n'

def _tune_lt(weights, transposed):
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(name="muse_lt_v1", cpp_sources=_LT_SOURCE,
                      functions=None, with_cuda=True,
                      extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)
    plans = {}
    # Synthetic activations only: preparation retains algorithms and weights,
    # never activations, computed outputs, or graphs of those outputs.
    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    ws = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for layer, w in weights.items():
        if layer == "router":
            continue
        k = w.shape[1]
        for m in [8, 16, 32, 128, 256]:
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.zeros((m, k), device=w.device, dtype=w.dtype)
            y = torch.empty((m, n), device=w.device, dtype=w.dtype)
            best = None
            for packed in [False, True]:
                pw = transposed[layer] if packed else w
                # The narrow unpooler slice needs its own contiguous KN layout.
                if packed and n != w.shape[0]:
                    pw = w[:n].T.contiguous()
                plan = ext.Plan(m, n, k, packed)
                for index in range(plan.count()):
                    try:
                        plan.run(x, pw, y, ws, index)
                        graph = torch.cuda.CUDAGraph()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.graph(graph):
                            flush.zero_()
                            start.record()
                            plan.run(x, pw, y, ws, index)
                            end.record()
                        values = []
                        for _ in range(3):
                            graph.replay()
                            end.synchronize()
                            values.append(start.elapsed_time(end))
                        score = sorted(values)[1]
                        if best is None or score < best[0]:
                            best = (score, plan, index, pw)
                    except RuntimeError:
                        continue
            if best is not None:
                plans[(layer, m, n, k)] = best[1:]
                print("LT", layer, m, n, "us", round(best[0] * 1000, 3),
                      "algo", best[2], "packed", best[3].stride(0) != k)
    return plans

def prepare(weights: Weights):
    state = dict(weights)
    state["transposed"] = {name: w.T.contiguous() for name, w in weights.items() if name != "router"}
    state["gemv"] = None
    if torch.cuda.get_device_capability()[0] == 9:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        for name, w in weights.items():
            if name == "router":
                continue
            n = 1024 if name == "unpooler" else w.shape[0]
            hopper_bf16_gemv(torch.zeros((1, w.shape[1]), device=w.device, dtype=w.dtype), w[:n])
        state["gemv"] = hopper_bf16_gemv
    hi = weights["router"].to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["plans"] = _tune_lt(weights, state["transposed"])
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    layer, n = data["config"]["layer"], data["config"]["n"]
    m, k = x.shape
    router = layer == "router"
    w = state[layer]
    if m == 1 and not router and state["gemv"] is not None:
        return state["gemv"](x, w[:n])
    if m == 1 and x.is_contiguous():
        out = torch.empty((m, n), device=x.device,
                          dtype=torch.float32 if router else torch.bfloat16)
        _gemv[(tr.cdiv(n, 4),)](x, w, out, n, k, 4, tr.next_power_of_2(k), num_warps=4)
        return out
    if not router:
        plan_info = state["plans"].get((layer, m, n, k))
        if plan_info is not None and x.is_contiguous():
            plan, index, pw = plan_info
            out = torch.empty((m, n), device=x.device, dtype=x.dtype)
            ws = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.uint8)
            plan.run(x, pw, out, ws, index)
            return out
        return torch.mm(x, w[:n].T)
    split = 16 if m <= 32 else 4
    bm, bn, bk = (16, 32, 64) if m <= 32 else (32, 64, 64)
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    _split_gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x, state["router_hi"] if router else w,
        state["router_lo"] if router else w, p, m, n, k, x.stride(0),
        bm, bn, bk, split, router, num_warps=4, num_stages=2)
    _finish[(tr.cdiv(m * n, 256),)](p, out, m * n, split, 256)
    return out
