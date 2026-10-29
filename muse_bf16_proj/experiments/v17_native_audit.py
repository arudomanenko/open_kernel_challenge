# okc: problem=muse_bf16_proj
"""H200 dense projections: measured GEMV, split-K router, and library GEMM paths.

BF16 weights retain their original values. Only the router uses the explicitly
permitted BF16 hi/lo representation, with separate FP32 accumulators.
"""
from typing import TypedDict
from threading import RLock

import torch
import triton as tr
import triton.language as tl


# Serialize configuration and launch among callers of this module. External
# DeepGEMM users must coordinate separately if they change runtime settings.
_DEEP_LOCK = RLock()


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
                 BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for it in range(tr.cdiv(K, BK * SPLIT)):
        k = (it * SPLIT + si) * BK + r
        a = tl.load(X + m[:, None] * SX + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(W + k[:, None] * WN + n[None, :],
                    (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + si * M * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


# Choices frozen from cold-L2 H200 CUDA-graph experiments.
_TACTICS = {('qkv_gate', 1, 2304, 8192): ('gemv', 1, 4),
 ('qkv_gate', 8, 2304, 8192): ('raw', 16, 32, 256, 4, 4, 3),
 ('qkv_gate', 16, 2304, 8192): ('torch', False),
 ('qkv_gate', 32, 2304, 8192): ('torch', False),
 ('qkv_gate', 128, 2304, 8192): ('deepcfg', 128),
 ('qkv_gate', 256, 2304, 8192): ('deep',),
 ('o', 1, 8192, 1024): ('gemv', 4, 4),
 ('o', 8, 8192, 1024): ('torch', False),
 ('o', 16, 8192, 1024): ('torch', False),
 ('o', 32, 8192, 1024): ('torch', False),
 ('o', 128, 8192, 1024): ('torch', False),
 ('o', 256, 8192, 1024): ('torch', False),
 ('pooler', 1, 4096, 8192): ('sglang',),
 ('pooler', 8, 4096, 8192): ('packed', 16, 64, 128, 4, 4, 3),
 ('pooler', 16, 4096, 8192): ('deepcfg', 128),
 ('pooler', 32, 4096, 8192): ('deep',),
 ('pooler', 128, 4096, 8192): ('deep',),
 ('pooler', 256, 4096, 8192): ('torch', True),
 ('unpooler', 1, 1024, 4096): ('gemv', 1, 4),
 ('unpooler', 8, 1024, 4096): ('raw', 16, 32, 256, 4, 4, 3),
 ('unpooler', 16, 8192, 4096): ('torch', True),
 ('unpooler', 32, 8192, 4096): ('torch', True),
 ('unpooler', 128, 8192, 4096): ('torch', True),
 ('unpooler', 256, 8192, 4096): ('deepcfg', 132),
 ('router', 1, 256, 8192): ('gemv', 1, 8),
 ('router', 8, 256, 8192): ('raw', 16, 32, 128, 16, 4, 3),
 ('router', 16, 256, 8192): ('raw', 16, 32, 128, 16, 4, 3),
 ('router', 32, 256, 8192): ('raw', 16, 32, 128, 16, 4, 3),
 ('router', 128, 256, 8192): ('raw', 64, 32, 128, 16, 4, 3),
 ('router', 256, 256, 8192): ('raw', 64, 32, 64, 16, 4, 3)}


def _execute(x, layer, n, state, tactic):
    m, k = x.shape
    router = layer == "router"
    kind = tactic[0]
    if kind == "torch":
        w = state["transposed"][layer][:, :n] if tactic[1] else state[layer][:n].T
        return torch.mm(x, w)
    # Native kernels require aligned, dense activations and weights. Route
    # unusual views through the general library path without copying inputs.
    native_layout = (x.is_contiguous() and x.data_ptr() % 16 == 0
                     and state[layer].data_ptr() % 16 == 0)
    if kind in ("deep", "deepcfg", "sglang") and not native_layout:
        return torch.mm(x, state[layer][:n].T)
    if kind == "sglang" and state["gemv"] is not None:
        return state["gemv"](x, state[layer][:n])
    if kind in ("deep", "deepcfg") and state["deep"] is None:
        return torch.mm(x, state[layer][:n].T)
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    if kind in ("deep", "deepcfg"):
        sms = state["sms"] if kind == "deep" else min(tactic[1], state["sms"])
        dg = state["deep_module"]
        with _DEEP_LOCK:
            old_sms = dg.get_num_sms()
            old_deterministic = dg.get_deterministic_algorithms()
            try:
                dg.set_num_sms(sms)
                dg.use_deterministic_algorithms(True)
                state["deep"](x, state[layer][:n], out)
            finally:
                dg.use_deterministic_algorithms(old_deterministic)
                dg.set_num_sms(old_sms)
        return out
    if kind in ("gemv", "sglang"):
        bn, warps = (1, 4) if kind == "sglang" else tactic[1:]
        _gemv[(tr.cdiv(n, bn),)](x, state[layer], out, n, k, bn,
                                tr.next_power_of_2(k), num_warps=warps)
        return out
    _, bm, bn, bk, split, warps, stages = tactic
    partials = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    grid = (tr.cdiv(m, bm), tr.cdiv(n, bn), split)
    if kind == "packed":
        w = state["transposed"][layer]
        _packed_gemm[grid](x, w, partials, m, n, k, w.shape[1], x.stride(0),
                           bm, bn, bk, split, num_warps=warps, num_stages=stages)
    else:
        _split_gemm[grid](x, state["router_hi"] if router else state[layer],
                          state["router_lo"] if router else state[layer], partials,
                          m, n, k, x.stride(0), bm, bn, bk, split, router,
                          num_warps=warps, num_stages=stages)
    _finish[(tr.cdiv(m * n, 256),)](partials, out, m * n, split, 256)
    return out


def prepare(weights: Weights):
    # Keep normalized, lossless original weights for library and general paths.
    state = {name: w.contiguous() for name, w in weights.items()}
    packed_layers = {key[0] for key, tactic in _TACTICS.items()
                     if tactic[0] == "packed" or (tactic[0] == "torch" and tactic[1])}
    state["transposed"] = {name: state[name].T.contiguous() for name in packed_layers}
    hi = state["router"].to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (state["router"] - hi.float()).to(torch.bfloat16)
    state["sms"] = torch.cuda.get_device_properties(state["router"].device).multi_processor_count
    state["gemv"], state["deep"] = None, None
    if torch.cuda.get_device_capability(state["router"].device)[0] == 9:
        try:
            from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
            state["gemv"] = hopper_bf16_gemv
        except ImportError:
            pass
        try:
            import deep_gemm
            # Check all required public APIs before enabling this backend.
            for api in ("get_num_sms", "set_num_sms",
                        "get_deterministic_algorithms", "use_deterministic_algorithms"):
                getattr(deep_gemm, api)
            state["deep_module"] = deep_gemm
            state["deep"] = deep_gemm.bf16_gemm_nt
        except (ImportError, AttributeError):
            pass
    # Compile selected library specializations using synthetic activations only.
    # Nothing derived from an actual invocation is available or retained here.
    for (layer, m, n, k), tactic in _TACTICS.items():
        if tactic[0] in ("sglang", "deep", "deepcfg"):
            x = torch.zeros((m, k), device=state[layer].device, dtype=torch.bfloat16)
            _execute(x, layer, n, state, tactic)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    layer, n = data["config"]["layer"], data["config"]["n"]
    m, k = x.shape
    router = layer == "router"
    if m == 0 or n == 0:
        return torch.empty((m, n), device=x.device,
                           dtype=torch.float32 if router else torch.bfloat16)
    if x.stride(1) != 1:
        return torch.mm(x.float() if router else x, state[layer][:n].T)
    tactic = _TACTICS.get((layer, m, n, k))
    if tactic is None:
        if router:
            tactic = (("gemv", 1, 8) if m == 1 else
                      ("raw", 16, 32, 128, 16, 4, 3) if m <= 32 else
                      ("raw", 64, 32, 128, 16, 4, 3))
        else:
            return torch.mm(x, state[layer][:n].T)
    return _execute(x, layer, n, state, tactic)

# Development-only diagnostics: synthetic inputs, no retained products.
_prepare_candidate = prepare

def prepare(weights):
    import json
    import statistics
    import deep_gemm as dg
    before = (dg.get_num_sms(), dg.get_deterministic_algorithms())
    state = _prepare_candidate(weights)
    assert before == (dg.get_num_sms(), dg.get_deterministic_algorithms())
    layout_checks = 0
    for (layer, m, n, k), tactic in _TACTICS.items():
        if tactic[0] not in ("deep", "deepcfg", "sglang"):
            continue
        for layout in ("dense", "padded", "offset"):
            if layout == "dense":
                x = torch.empty((m, k), device=weights[layer].device, dtype=torch.bfloat16)
            elif layout == "padded":
                x = torch.empty((m, k + 1), device=weights[layer].device, dtype=torch.bfloat16)[:, :k]
            else:
                x = torch.empty(m * k + 1, device=weights[layer].device, dtype=torch.bfloat16)[1:].view(m, k)
            x.normal_()
            data = {"x": x, "config": {"layer": layer, "n": n}}
            kernel_fn(data, state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = kernel_fn(data, state)
            x.normal_()
            graph.replay()
            ref = (x.double() @ weights[layer][:n].double().T).bfloat16()
            error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            assert error < .01, (layer, m, layout, error)
            assert before == (dg.get_num_sms(), dg.get_deterministic_algorithms())
            layout_checks += 1
    print('NATIVE_LAYOUT_AUDIT', layout_checks, 'configuration_restored', flush=True)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=weights['router'].device)
    for layer, ms in [('qkv_gate', (4, 7)), ('pooler', (4, 7, 15, 17, 33, 129)),
                      ('unpooler', (4, 7, 17, 129))]:
        for m in ms:
            n, k = weights[layer].shape
            if layer == 'unpooler' and m <= 8:
                n = 1024
            x = torch.randn((m,k), device=weights[layer].device, dtype=torch.bfloat16)
            choices = [('torch', False)]
            if layer in state['transposed']:
                choices.append(('torch', True))
            if m <= 8:
                choices.append(('packed',16,64,128,4,4,3) if layer=='pooler' else ('raw',16,32,256,4,4,3))
            elif layer == 'pooler':
                choices.append(('deep',))
            records=[]
            graphs=[]
            for tactic in choices:
                _execute(x,layer,n,state,tactic)
                graph=torch.cuda.CUDAGraph()
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                with torch.cuda.graph(graph):
                    flush.zero_()
                    start.record()
                    out=_execute(x,layer,n,state,tactic)
                    end.record()
                graphs.append((graph,start,end,out))
            timings=[[] for _ in choices]
            for rep in range(21):
                for i in (range(len(choices)) if rep%2==0 else reversed(range(len(choices)))):
                    graph,start,end,out=graphs[i]
                    graph.replay()
                    end.synchronize()
                    timings[i].append(start.elapsed_time(end)*1000)
            ref=(x.double() @ weights[layer][:n].double().T).bfloat16()
            for tactic,times,(_,_,_,out) in zip(choices,timings,graphs):
                error=((out.float()-ref.float()).norm()/ref.float().norm()).item()
                assert error<.01,(layer,m,tactic,error)
                records.append({'tactic':tactic,'us':statistics.median(times)})
            print('NEARBY',json.dumps({'layer':layer,'m':m,'results':records}),flush=True)
    assert before == (dg.get_num_sms(), dg.get_deterministic_algorithms())
    return state
