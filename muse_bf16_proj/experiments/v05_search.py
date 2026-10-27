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


@tr.jit
def _packed_gemm(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 WN: tl.constexpr, SX: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                 EXPAND: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    total: tl.constexpr = K * 2 if EXPAND else K
    for it in range(tr.cdiv(total, BK * SPLIT)):
        k = (it * SPLIT + si) * BK + r
        kk = k % K if EXPAND else k
        a = tl.load(X + m[:, None] * SX + kk[None, :],
                    (m[:, None] < M) & (k[None, :] < total), 0)
        b = tl.load(W + k[:, None] * WN + n[None, :],
                    (n[None, :] < N) & (k[:, None] < total), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + si * M * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))

def _run_tactic(x, layer, n, state, tactic):
    m, k = x.shape
    router = layer == "router"
    kind = tactic[0]
    if kind == "torch":
        w = state["transposed"][layer][:, :n] if tactic[1] else state[layer][:n].T
        return torch.mm(x, w)
    if kind == "fp32":
        return torch.mm(x.float(), state[layer][:n].T)
    if kind == "pair":
        a = torch.mm(x, state["router_hi_t"][:, :n], out_dtype=torch.float32)
        a.add_(torch.mm(x, state["router_lo_t"][:, :n], out_dtype=torch.float32))
        return a
    if kind == "cat":
        return torch.mm(torch.cat((x, x), dim=1), state["router_packed"][:, :n], out_dtype=torch.float32)
    if kind == "sglang":
        return state["gemv"](x, state[layer][:n])
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    if kind == "deep":
        state["deep"](x, state[layer][:n], out)
        return out
    if kind == "gemv":
        _, bn, warps = tactic
        _gemv[(tr.cdiv(n, bn),)](x, state[layer], out, n, k, bn, tr.next_power_of_2(k), num_warps=warps)
        return out
    _, bm, bn, bk, split, warps, stages = tactic
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else out
    if kind in ("packed", "expand"):
        w = state["router_packed"] if router else state["transposed"][layer]
        _packed_gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
            x, w, p, m, n, k, w.shape[1], x.stride(0),
            bm, bn, bk, split, router, num_warps=warps, num_stages=stages)
    else:
        _split_gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
            x, state["router_hi"] if router else state[layer],
            state["router_lo"] if router else state[layer], p,
            m, n, k, x.stride(0), bm, bn, bk, split, router,
            num_warps=warps, num_stages=stages)
    if split > 1:
        _finish[(tr.cdiv(m * n, 256),)](p, out, m * n, split, 256)
    return out


def _candidates(layer, m, state):
    if layer == "router":
        choices = [("fp32",), ("pair",), ("cat",)]
        if m == 1:
            choices += [("gemv", 1, 4), ("gemv", 1, 8), ("gemv", 4, 4)]
        if m <= 32:
            choices += [("raw", 16, 32, 64, 16, 4, 2),
                        ("raw", 16, 32, 128, 16, 4, 3),
                        ("expand", 16, 32, 128, 16, 4, 3),
                        ("expand", 16, 32, 256, 16, 4, 3)]
        else:
            choices += [("raw", 16, 32, 64, 16, 4, 2),
                        ("raw", 32, 32, 128, 16, 4, 3),
                        ("raw", 64, 32, 64, 16, 4, 3),
                        ("expand", 16, 32, 128, 8, 4, 3),
                        ("expand", 32, 64, 128, 8, 4, 3),
                        ("expand", 64, 32, 64, 8, 4, 3)]
        return choices
    choices = [("torch", False), ("torch", True)]
    if state["deep"] is not None:
        choices += [("deep",)]
    if m == 1:
        choices += [("gemv", 1, 4), ("gemv", 4, 4)]
        if state["gemv"] is not None:
            choices += [("sglang",)]
        choices += [("raw", 16, 64, 128, 8, 4, 3)]
    elif m <= 32:
        choices += [("packed", 16, 16, 256, 1, 4, 3),
                    ("packed", 16, 32, 128, 4, 4, 3),
                    ("packed", 16, 64, 128, 4, 4, 3),
                    ("raw", 16, 32, 256, 4, 4, 3),
                    ("raw", 16, 64, 256, 4, 4, 3)]
    else:
        choices += [("raw", 32, 64, 64, 4, 4, 3),
                    ("raw", 64, 64, 64, 4, 4, 3),
                    ("raw", 64, 128, 64, 4, 8, 3),
                    ("raw", 64, 64, 128, 2, 4, 3),
                    ("packed", 32, 64, 128, 4, 4, 3)]
    return choices


def prepare(weights: Weights):
    state = dict(weights)
    state["transposed"] = {name: w.T.contiguous() for name, w in weights.items() if name != "router"}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state.update(router_hi=hi, router_lo=lo, router_hi_t=hi.T.contiguous(),
                 router_lo_t=lo.T.contiguous(),
                 router_packed=torch.cat((hi, lo), dim=1).T.contiguous())
    state["gemv"] = None
    if torch.cuda.get_device_capability()[0] == 9:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        state["gemv"] = hopper_bf16_gemv
    state["deep"] = None
    try:
        import deep_gemm
        state["deep"] = getattr(deep_gemm, "bf16_gemm_nt", None)
    except ImportError:
        pass
    state["tactics"] = {}
    # These are synthetic activations for kernel selection only. None of their
    # values, products, or graphs is retained in the prepared state.
    flush = torch.empty(64 * 1024 * 1024, device=weights["router"].device, dtype=torch.uint8)
    for layer, w in weights.items():
        for m in [1, 8, 16, 32, 128, 256]:
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.randn((m, w.shape[1]), device=w.device, dtype=torch.bfloat16)
            timings = []
            for tactic in _candidates(layer, m, state):
                try:
                    _run_tactic(x, layer, n, state, tactic)
                    graph = torch.cuda.CUDAGraph()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.graph(graph):
                        flush.zero_()
                        start.record()
                        _run_tactic(x, layer, n, state, tactic)
                        end.record()
                    samples = []
                    for _ in range(5):
                        graph.replay()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                    timings.append((sorted(samples)[2], tactic))
                except (RuntimeError, ValueError, AssertionError, TypeError) as error:
                    print("candidate unavailable", layer, m, tactic, str(error)[:160])
            timings.sort()
            state["tactics"][(layer, m, n, w.shape[1])] = timings[0][1]
            print("SELECT", layer, m, timings[:3])
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    layer, n = data["config"]["layer"], data["config"]["n"]
    m, k = x.shape
    tactic = state["tactics"].get((layer, m, n, k))
    if tactic is None or x.stride(1) != 1:
        return torch.mm(x.float() if layer == "router" else x, state[layer][:n].T)
    if m == 256 and layer == "router":
        print("TACTICS", repr(state["tactics"]))
    return _run_tactic(x, layer, n, state, tactic)
