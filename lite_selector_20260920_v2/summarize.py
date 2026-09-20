"""Recompute the preregistered G1 gate from all four train/validation processes."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
METHODS = ['int8', 'e3m4', 'fp16']


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def quantile(values, q):
    x = sorted(values); pos = (len(x)-1)*q; lo = int(pos); hi = math.ceil(pos)
    return x[lo] + (x[hi]-x[lo])*(pos-lo)


def interval(logs):
    center = statistics.mean(logs); radius = 3.182446305*statistics.stdev(logs)/2
    return {'process_ratios': [math.exp(v) for v in logs], 'geomean': math.exp(center),
            'log_t95_interval': [math.exp(center-radius), math.exp(center+radius)]}


def grouped_ratio(rows, numerator, denominator):
    parents = sorted({r['parent_id'] for r in rows}); groups = {}
    for parent in parents:
        subset = [r for r in rows if r['parent_id'] == parent]
        assert len(subset) == 4
        groups[parent] = [statistics.mean(math.log(r['process_medians_ms'][numerator][p] /
                                                  r['process_medians_ms'][denominator][p]) for r in subset)
                          for p in range(4)]
    return dict(interval([statistics.mean(groups[parent][p] for parent in parents) for p in range(4)]),
                parent_ratios={parent: gm([math.exp(v) for v in vals]) for parent, vals in groups.items()})


def compute(root=ROOT):
    paths = [root/'results'/f'g1_bench_{i}.json' for i in range(4)]
    runs = [json.loads(p.read_text()) for p in paths]
    assert all(r['pass'] and r['process'] == i for i, r in enumerate(runs))
    assert all(r['source_sha256'] == runs[0]['source_sha256'] and
               r['environment'] == runs[0]['environment'] and
               r['contract_sha256'] == runs[0]['contract_sha256'] for r in runs)
    manifest = [json.loads(s) for s in (root/'workloads.jsonl').read_text().splitlines()]
    assert len(manifest) == 96 and sum(r['split'] == 'test' for r in manifest) == 32
    configs = [r for r in manifest if r['split'] != 'test']
    rows = []
    for cfg in configs:
        med = {}; diagnostics = {}
        for method in METHODS:
            samples = [[r for r in run['records'] if r['config_id'] == cfg['config_id'] and
                        r['method'] == method and r['role'] == 'sample' and not r['warmup']] for run in runs]
            assert all(len(s) == 5 and all(r['correct'] for r in s) for s in samples)
            for p, sample in enumerate(samples):
                refs = [r for r in runs[p]['records'] if r['config_id'] == cfg['config_id'] and r['role'] == 'reference']
                assert len(refs) == 1
                assert all(r['output_sha256'] == refs[0]['output_sha256'] and r['input_sha256'] == cfg['input_sha256']
                           and r['threshold_hex'] == cfg['threshold_hex'] for r in sample)
            med[method] = [statistics.median(r['e2e_ms'] for r in sample) for sample in samples]
            flat = sum(samples, [])
            diagnostics[method] = {'p10_p50_p90_ms': [quantile([r['e2e_ms'] for r in flat], q) for q in (.1,.5,.9)],
                'fp32_pairs': sorted({r['fp32_pairs'] for r in flat}), 'fp64_pairs': sorted({r['fp64_pairs'] for r in flat}),
                'output_count': flat[0]['output_count'], 'output_sha256': flat[0]['output_sha256'],
                'peak_memory_bytes_max': max(r['peak_memory_bytes'] for r in flat),
                'feature_ms': 0., 'inference_ms': 0.}
        labels = {m: gm(med[m]) for m in METHODS}
        winner = min(('int8','e3m4'), key=labels.get)
        other = 'e3m4' if winner == 'int8' else 'int8'
        consistent = all(a < b for a,b in zip(med[winner],med[other]))
        tie = labels[other]/labels[winner] <= 1.03 or not consistent
        med['O2'] = [min(med['int8'][p],med['e3m4'][p]) for p in range(4)]
        med['O3'] = [min(med[m][p] for m in METHODS) for p in range(4)]
        rows.append(dict(cfg, process_medians_ms=med, label_ms=labels,
                         label='tie' if tie else winner,
                         sample_weight=abs(labels['int8']-labels['e3m4'])/min(labels['int8'],labels['e3m4']),
                         observed_free_O2_ms=min(labels['int8'],labels['e3m4']),
                         observed_free_O3_ms=min(labels.values()), diagnostics=diagnostics))
    split_summary = {}
    for split in ('train','validation'):
        part = [r for r in rows if r['split'] == split]
        # Four configurations per parent, equal parent weight. Balanced counts
        # make the scalar label GM equivalent to the explicit two-level GM.
        by_parent = {p: [r for r in part if r['parent_id'] == p] for p in {r['parent_id'] for r in part}}
        fixed = {m: gm([gm([r['label_ms'][m] for r in group]) for group in by_parent.values()]) for m in METHODS}
        oracle = gm([gm([r['observed_free_O2_ms'] for r in group]) for group in by_parent.values()])
        best = min(fixed,key=fixed.get)
        split_summary[split] = {'fixed_geomean_ms':fixed, 'best_fixed':best, 'free_O2_geomean_ms':oracle,
            'best_fixed_over_free_O2':fixed[best]/oracle,
            'process_paired_free_O2_vs_fixed':{m:grouped_ratio(part,m,'O2') for m in METHODS},
            'format_comparisons':{'int8_over_e3m4':grouped_ratio(part,'int8','e3m4'),
                                  'fp16_over_e3m4':grouped_ratio(part,'fp16','e3m4')}}
    dominance = {}
    for fmt, other in [('int8','e3m4'),('e3m4','int8')]:
        wins = [r for r in rows if all(b/a > 1.05 for a,b in zip(r['process_medians_ms'][fmt],r['process_medians_ms'][other]))]
        parents = sorted({r['parent_id'] for r in wins})
        all_parents = sorted({r['parent_id'] for r in rows})
        parent_level = [parent for parent in all_parents if all(
            gm([r['process_medians_ms'][other][p]/r['process_medians_ms'][fmt][p]
                for r in rows if r['parent_id'] == parent]) > 1.05 for p in range(4))]
        dominance[fmt] = {'config_ids':[r['config_id'] for r in wins], 'parents':parents,
                          'parent_aggregate_consistent_winners':parent_level,'pass':len(parents)>=2}
    gate = all(d['pass'] for d in dominance.values()) and split_summary['validation']['best_fixed_over_free_O2'] >= 1.10
    return {'G1_pass':gate, 'scope':'train/validation only; free observed oracle is not a measured deployable router',
            'source_sha256':runs[0]['source_sha256'], 'contract_sha256':runs[0]['contract_sha256'],
            'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'splits':split_summary,'dominance':dominance,'configs':rows,
            'test_timing_records':sum(r.get('split') == 'test' for run in runs for r in run['records']),
            'oracle_distinction':'Gate uses min of four-process GM labels per config. Process intervals use min of method medians within each process; they are separate optimistic summaries, not the same estimator.'}


def main():
    summary=compute(); out=ROOT/'results';(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'configurations.csv').open('w') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['config_id','parent_id','split','N','k','T_hex','label',
                'int8_ms','e3m4_ms','fp16_ms','O2_ms','O3_ms','int8_FP32','e3m4_FP32','fp16_FP32','output_count'])
        for r in summary['configs']:
            writer.writerow([r['config_id'],r['parent_id'],r['split'],r['n'],r['target_k'],r['threshold_hex'],r['label'],
                *[r['label_ms'][m] for m in METHODS],r['observed_free_O2_ms'],r['observed_free_O3_ms'],
                *[r['diagnostics'][m]['fp32_pairs'][0] for m in METHODS],r['diagnostics']['int8']['output_count']])
    print(json.dumps({'G1_pass':summary['G1_pass'],'dominance':summary['dominance'],
                      'validation':summary['splits']['validation'],'test_timing_records':summary['test_timing_records']},indent=2))


if __name__ == '__main__': main()
