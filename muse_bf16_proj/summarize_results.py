"""Build benchmark CSVs and the experiment index from saved official results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
SHAPES = {'qkv_gate': (2304, 8192), 'o': (8192, 1024),
          'pooler': (4096, 8192), 'unpooler': (8192, 4096), 'router': (256, 8192)}


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def case_rows(document, version):
    rows = []
    for entry in (document.get('result') or {}).get('records', []):
        if entry.get('event') != 'benchmark' or 'baseline_ns' not in entry:
            continue
        case = entry['case']
        m, layer = case['m'], case['layer']
        n, k = SHAPES[layer]
        if layer == 'unpooler' and m <= 8:
            n = 1024
        rows.append(dict(version=version, job=document['id'], layer=layer,
                         load=case['load'], m=m, n=n, k=k,
                         baseline_us=entry['baseline_ns'] / 1000,
                         candidate_us=entry['median_ns'] / 1000,
                         speedup=entry['paired_ratio'],
                         theoretical_floor_us=entry['speed_of_light_ns'] / 1000,
                         correct=entry['status'] == 'pass'))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', help='Saved JSON from the final official check or submission')
    parser.add_argument('--snapshot', help='Exact local source snapshot sent for that run')
    args = parser.parse_args()
    experiments, cases = [], []
    for path in sorted(REPORTS.glob('v*_result.json')):
        if not path.stat().st_size:
            continue
        document = json.loads(path.read_text())
        result = document.get('result') or {}
        version = path.name.split('_')[0]
        setup = next((v for v in result.get('records', []) if v.get('event') == 'setup'), {})
        ci = result.get('speedup_ci') or ['', '']
        experiments.append(dict(version=version, job=document['id'], status=document['status'],
                                passed=result.get('passed'), speedup=result.get('speedup'),
                                ci_low=ci[0], ci_high=ci[1], prepare_s=setup.get('prepare_s'),
                                measured_cases=len(case_rows(document, version))))
        cases.extend(case_rows(document, version))
    write_csv(REPORTS / 'experiments.csv', experiments)
    write_csv(REPORTS / 'all_cases.csv', cases)
    if not args.final:
        return
    document = json.loads(Path(args.final).read_text())
    result = document['result']
    assert result['passed'], 'Do not label a failing run as the final result'
    rows = case_rows(document, 'final')
    assert len(rows) == 30 and all(r['correct'] for r in rows)
    dispatch = {(d['layer'], d['m']): d
                for d in json.loads((REPORTS / 'dispatch.json').read_text())}
    for row in rows:
        choice = dispatch[(row['layer'], row['m'])]
        row['backend'] = choice['backend']
        row['parameters'] = json.dumps(choice['parameters'])
    write_csv(REPORTS / 'benchmark.csv', rows)
    source = (ROOT / 'submission.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if args.snapshot:
        assert Path(args.snapshot).read_bytes() == source, 'Delivered file differs from submitted snapshot'
    summary = dict(job=document['id'], mode=document.get('mode'),
                   speedup=result['speedup'], speedup_ci=result['speedup_ci'],
                   score_us=result['score_us'], baseline_step_us=result['baseline_us'],
                   gpu=result['gpu'], machine=result.get('machine'),
                   evaluator=result.get('evaluator'),
                   passed_tests=sum(e.get('event') == 'test' and e.get('status') == 'pass'
                                    for e in result['records']),
                   passed_scored_cases=len(rows),
                   submission_sha256=digest,
                   service_content_hash=document['content_hash'],
                   submission_bytes=len(source))
    (REPORTS / 'final_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = ['| Layer | M | N | K | SGLang µs | Candidate µs | Speedup |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append('| {layer} | {m} | {n} | {k} | {baseline_us:.2f} | {candidate_us:.2f} | {speedup:.3f}× |'.format(**row))
    (REPORTS / 'benchmark.md').write_text(
        '# Official H200 benchmark\n\n'
        f'Job `{document["id"]}` ({document.get("mode")}). All 9 preflight tests and all 30 scored cases passed.\n\n'
        f'Overall: **{result["speedup"]:.4f}×** over SGLang, 95% interval '
        f'**[{result["speedup_ci"][0]:.4f}, {result["speedup_ci"][1]:.4f}]**. '
        f'Normalized step time: {result["score_us"]:.2f} µs versus {result["baseline_us"]:.2f} µs. '
        'This is a model-step score, not the latency of one GEMM.\n\n'
        'Each row reports the official paired cold-L2 CUDA-graph measurement. '
        'The candidate time is the baseline median divided by the median paired speedup, as specified by the judge. '
        'The overall score aggregates weighted projection times within each of six loads, then takes the geometric mean of load speedups; '
        'it is not the geometric mean of these 30 individual ratios.\n\n'
        + '\n'.join(table) + '\n\n'
        f'Artifact SHA-256: `{summary["submission_sha256"]}`. '
        'Full timings, samples, correctness messages, evaluator image digest and machine details are retained in the result JSON.\n')


if __name__ == '__main__':
    main()
