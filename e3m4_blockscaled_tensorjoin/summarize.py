"""Rebuild the frozen four-process block-scaled Stage 1 timing summary."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ['int8', 'fp16', 'e4m3', 'e3m4', 'mx_e4m3', 'mx_e3m4']
CASES = ['cifar4096', 'synthetic_clustered', 'synthetic_outlier']
PHASES = ['transfer_allocate_prepare_ms', 'stage1_ms', 'host_counts_fp32_ms', 'fp64_ms']


def ratio_stats(comparator, candidate):
    ratios = [a/b for a, b in zip(comparator, candidate)]
    logs = list(map(math.log, ratios)); mean = statistics.mean(logs)
    half = 3.182446305 * statistics.stdev(logs)/2
    return {'process_ratios': ratios, 'geomean': math.exp(mean),
            'arithmetic_mean': statistics.mean(ratios),
            'log_t95_interval': [math.exp(mean-half), math.exp(mean+half)],
            'process_wins': sum(r > 1 for r in ratios),
            'all_processes_gt_1_10': all(r > 1.10 for r in ratios),
            'forward_geomean': math.sqrt(ratios[0]*ratios[2]),
            'reverse_geomean': math.sqrt(ratios[1]*ratios[3])}


def summarize(root=ROOT):
    files = [root/'results'/f'bench_v1_{i}.json' for i in range(4)]
    data = [json.loads(p.read_text()) for p in files]
    assert all(d['pass'] and d['ptxas_version'] == '13.1' for d in data)
    assert [d['order'] for d in data] == [0, 1, 2, 3]
    assert all(d['source_sha256'] == data[0]['source_sha256'] for d in data)
    result = {'raw_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'source_sha256': data[0]['source_sha256'],
              'scope': 'pageable host input to sorted canonical host IDs; warm JIT; all preparation and cascade work charged',
              'estimator': 'four paired process medians, comparator/candidate, Student-t log interval df=3',
              'cases': {}, 'sustained': {}}
    for case in CASES:
        groups = {}
        hashes = set()
        for method in METHODS:
            samples = [[r for r in d['records'] if r['case'] == case and r.get('format') == method
                        and r['role'] == 'sample' and not r['warmup']] for d in data]
            assert all(len(s) == 5 for s in samples), (case, method)
            assert all(r['exact_ids'] for s in samples for r in s)
            raw = [r for s in samples for r in s]
            hashes.update(r['output_sha256'] for r in raw)
            med = [float(np.median([r['e2e_ms'] for r in s])) for s in samples]
            groups[method] = {'n': raw[0]['n'], 'threshold_d2': raw[0]['threshold_d2'],
                'process_medians_ms': med,
                'p10_p50_p90_ms': np.percentile([r['e2e_ms'] for r in raw], [10, 50, 90]).tolist(),
                'fp32_pairs': sorted({r['fp32_pairs'] for r in raw}),
                'fp64_pairs': sorted({r['fp64_pairs'] for r in raw}),
                'output_count': raw[0]['output_count'], 'output_sha256': raw[0]['output_sha256'],
                'input_sha256': raw[0]['input_sha256'],
                'phase_medians_ms': {key: float(np.median([r[key] for r in raw])) for key in PHASES}}
        assert len(hashes) == 1
        for method, row in groups.items():
            row['ratios'] = {base: ratio_stats(groups[base]['process_medians_ms'], row['process_medians_ms'])
                             for base in ('int8', 'fp16', 'mx_e4m3', 'e3m4')}
            row['marginal_int8_ratio'] = groups['int8']['p10_p50_p90_ms'][1]/row['p10_p50_p90_ms'][1]
            row['short_screen_gate'] = all(row['ratios'][b]['all_processes_gt_1_10']
                                          and row['ratios'][b]['log_t95_interval'][0] > 1
                                          for b in ('int8', 'fp16'))
        result['cases'][case] = groups
    batch = {}
    for method in METHODS:
        samples = [[r for r in d['records'] if r['case'] == 'cifar4096'
                    and r.get('format') == method and r['role'] == 'sustained'] for d in data]
        assert all(len(s) == 32 and all(r['exact_ids'] for r in s) for s in samples)
        envelope = [[r for r in d['records'] if r.get('format') == method and r['role'] == 'batch_wall'] for d in data]
        assert all(len(s) == 1 for s in envelope)
        batch[method] = {'process_medians_ms': [float(np.median([r['e2e_ms'] for r in s])) for s in samples],
                         'process_batch_mean_ms': [s[0]['mean_call_wall_ms'] for s in envelope],
                         'p10_p50_p90_ms': np.percentile([r['e2e_ms'] for s in samples for r in s], [10, 50, 90]).tolist()}
    for method, row in batch.items():
        row['ratios'] = {b: ratio_stats(batch[b]['process_medians_ms'], row['process_medians_ms'])
                         for b in ('int8', 'fp16', 'mx_e4m3', 'e3m4')}
        row['enclosing_batch_ratios'] = {b: ratio_stats(batch[b]['process_batch_mean_ms'], row['process_batch_mean_ms'])
                                         for b in ('int8', 'fp16')}
        row['non_regression_gate'] = all(all(r >= 1 for r in row['ratios'][b]['process_ratios']) and
                                          all(r >= 1 for r in row['enclosing_batch_ratios'][b]['process_ratios'])
                                          for b in ('int8', 'fp16'))
    result['sustained'] = batch
    result['mx_e3m4_real_data_screen_pass'] = (result['cases']['cifar4096']['mx_e3m4']['short_screen_gate']
                                             and batch['mx_e3m4']['non_regression_gate'])
    return result


if __name__ == '__main__':
    output = summarize()
    (ROOT/'results'/'SUMMARY.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({case: {fmt: {'ms': row['p10_p50_p90_ms'][1], 'fp32': row['fp32_pairs'],
                                  'fp64': row['fp64_pairs'], 'int8_ratio': row['ratios']['int8']['geomean']}
                            for fmt, row in methods.items()} for case, methods in output['cases'].items()}, indent=2))
