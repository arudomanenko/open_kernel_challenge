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
        b = tl.load(W + tl.program_id(1) * K * (2 if EXPAND else 1) * BN + k[:, None] * BN + tl.arange(0, BN)[None, :],
                    (n[None, :] < N) & (k[:, None] < total), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + si * M * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))

def prepare(weights: Weights):
    state = dict(weights)
    state["transposed"] = {name: w.T.contiguous() for name, w in weights.items() if name != "router"}
    state["blocked"] = {name: w.reshape(-1, 64, w.shape[1]).transpose(1, 2).contiguous()
                        for name, w in weights.items() if name != "router"}
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
    state["router_packed"] = torch.cat((hi, state["router_lo"]), dim=1).T.contiguous()
    state["router_blocked"] = torch.cat((hi, state["router_lo"]), dim=1).reshape(-1, 64, hi.shape[1] * 2).transpose(1, 2).contiguous()
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
        _gemv[(n,)](x, w, out, n, k, 1, tr.next_power_of_2(k), num_warps=4)
        return out

    if router and m > 32:
        return torch.mm(torch.cat((x, x), dim=1), state["router_packed"][:, :n], out_dtype=torch.float32)
    split = 16 if router else 4
    bm, bn, bk = (16, 64, 128) if m <= 32 else (64, 64, 64)
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else out
    w = state["router_blocked"] if router else state["blocked"][layer]
    _packed_gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x, w, p, m, n, k, w.shape[1], x.stride(0),
        bm, bn, bk, split, router, num_warps=4, num_stages=3)
    if split > 1:
        _finish[(tr.cdiv(m * n, 256),)](p, out, m * n, split, 256)
    return out
