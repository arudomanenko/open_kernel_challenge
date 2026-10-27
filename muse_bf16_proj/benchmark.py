"""Reproduce cold-cache CUDA-graph measurements outside the official judge.

Use the challenge image on an H200. This diagnostic does not replace okc check:
its event placement and cache flush differ from the competition harness.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reference


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(module, data, state, flush):
    module.kernel_fn(data, state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        flush.zero_()
        start.record()
        output = module.kernel_fn(data, state)
        end.record()
    return graph, start, end, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default=str(ROOT / 'submission.py'))
    parser.add_argument('--baseline', default=str(ROOT / 'baseline.py'))
    parser.add_argument('--repetitions', type=int, default=21)
    parser.add_argument('--output', default=str(ROOT / 'reports' / 'local_benchmark.csv'))
    args = parser.parse_args()
    print('Device:', torch.cuda.get_device_name(), 'PyTorch:', torch.__version__)
    candidate = load('candidate', args.candidate)
    baseline = load('baseline', args.baseline)
    weights = reference.generate_weights(20261027)
    states = [baseline.prepare(weights), candidate.prepare(weights)]
    flush = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    rows = []
    for m in [1, 8, 16, 32, 128, 256]:
        for layer in reference.LAYERS:
            data = reference.generate_input(m, layer, 12000 + m)
            graphs = [capture(mod, data, state, flush)
                      for mod, state in zip([baseline, candidate], states)]
            timings = [[], []]
            ratios = []
            for repetition in range(args.repetitions):
                pair = [None, None]
                for side in ([0, 1] if repetition % 2 == 0 else [1, 0]):
                    graph, start, end, out = graphs[side]
                    graph.replay()
                    end.synchronize()
                    pair[side] = start.elapsed_time(end) * 1000
                    timings[side].append(pair[side])
                ratios.append(pair[0] / pair[1])
            ok, message = reference.check_implementation(data, graphs[1][3], weights)
            assert ok, message
            row = dict(layer=layer, m=m, n=data['config']['n'], k=data['config']['k'],
                       baseline_us=statistics.median(timings[0]),
                       candidate_us=statistics.median(timings[1]),
                       paired_speedup=statistics.median(ratios), correct=ok)
            rows.append(row)
            print(row, flush=True)
    with open(args.output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
