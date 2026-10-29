"""Local correctness and CUDA-graph replay checks, independent of the judge.

Usage: .venv/bin/python muse_bf16_proj/validate.py --module submission.py
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reference


def load(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default=str(ROOT / 'submission.py'))
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    mod = load(args.module)
    weights = reference.generate_weights(20261027)
    state = mod.prepare(weights)
    cases = [(m, layer, 'normal') for m in [1, 8, 16, 32, 128, 256]
             for layer in reference.LAYERS]
    cases += [(7, 'pooler', 'normal'), (33, 'o', 'normal'),
              (130, 'unpooler', 'normal'), (5, 'unpooler', 'normal'),
              (9, 'router', 'outliers')]
    if args.full:
        cases += [(512, 'qkv_gate', 'outliers'), (1000, 'unpooler', 'outliers')]
    cases += [(m, "pooler", "normal") for m in [2, 3, 4, 5, 6, 9, 15, 17, 31, 33, 129]]
    cases += [(m, "unpooler", "normal") for m in [9, 15, 17, 31]]
    results = []
    for index, (m, layer, dist) in enumerate(cases):
        data = reference.generate_input(m, layer, 7100 + index, dist)
        mod.kernel_fn(data, state)
        torch.cuda.synchronize()
        # Capture and then change the existing graph input twice. This catches
        # cached activations, missing partial writes and stale output buffers.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mod.kernel_fn(data, state)
        for replay in range(2):
            fresh = reference.generate_input(m, layer, 9300 + index * 2 + replay, dist)
            data['x'].copy_(fresh['x'])
            graph.replay()
            ok, message = reference.check_implementation(data, out, weights)
            record = dict(m=m, layer=layer, dist=dist, replay=replay,
                          passed=ok, message=message)
            results.append(record)
            print(json.dumps(record), flush=True)
            if not ok:
                raise AssertionError(record)
    # A row-strided BF16 activation view and an all-zero input are valid too.
    for layer in ['qkv_gate', 'router']:
        data = reference.generate_input(9, layer, 10101)
        backing = torch.empty((9, data['x'].shape[1] * 2), device='cuda', dtype=torch.bfloat16)
        view = backing[:, :data['x'].shape[1]]
        view.copy_(data['x'])
        data['x'] = view
        out = mod.kernel_fn(data, state)
        ok, message = reference.check_implementation(data, out, weights)
        assert ok, message
        data['x'].zero_()
        out = mod.kernel_fn(data, state)
        assert bool((out == 0).all()), 'zero input must give zero output'
        fresh = reference.generate_input(9, layer, 10102)
        columns = backing[:, ::2]
        columns.copy_(fresh['x'])
        data['x'] = columns
        out = mod.kernel_fn(data, state)
        ok, message = reference.check_implementation(data, out, weights)
        assert ok, message
        data['x'] = columns[:0]
        out = mod.kernel_fn(data, state)
        assert out.shape == (0, data['config']['n'])
    # Exercise every native-selected shape with views that fail its layout
    # contract, including contiguous tensors with an unaligned base pointer.
    layout_checks = 0
    for (layer, m, n, k), tactic in mod._TACTICS.items():
        if tactic[0] not in ("deep", "deepcfg", "sglang"):
            continue
        for layout in ["padded_rows", "offset_base"]:
            data = reference.generate_input(m, layer, 14100 + m)
            if layout == "padded_rows":
                backing = torch.empty((m, k + 1), device="cuda", dtype=torch.bfloat16)
                view = backing[:, :k]
            else:
                backing = torch.empty(m * k + 1, device="cuda", dtype=torch.bfloat16)
                view = backing[1:].view(m, k)
            view.copy_(data["x"])
            data["x"] = view
            mod.kernel_fn(data, state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = mod.kernel_fn(data, state)
            fresh = reference.generate_input(m, layer, 15100 + m)
            view.copy_(fresh["x"])
            graph.replay()
            ok, message = reference.check_implementation(data, out, weights)
            assert ok, (layer, m, layout, message)
            layout_checks += 1
    summary = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                   candidate=args.module, passed=True, checks=results, native_layout_checks=layout_checks,
                   additional=['row-strided activations', 'column-strided activations',
                               'all-zero inputs', 'empty batches'])
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
