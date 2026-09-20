"""CPU-only replay of the v2 evidence, not a replacement for the GPU gates."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from summarize import compute, METHODS
from src.cpu_reference import terminal_order_distances

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_summary(saved, computed, path=''):
    """Allow at most eight ULPs in derived statistics across Python/libm builds.

    Raw evidence, source hashes, predicates, integer counts and gate decisions
    are still exact. This bound is not used by any GPU correctness comparison.
    """
    assert type(saved) is type(computed), path
    if isinstance(saved, dict):
        assert saved.keys() == computed.keys(), path
        for key in saved:
            compare_summary(saved[key], computed[key], path+'/'+key)
    elif isinstance(saved, list):
        assert len(saved) == len(computed), path
        for i, (a, b) in enumerate(zip(saved, computed)):
            compare_summary(a, b, path+'/'+str(i))
    elif isinstance(saved, float):
        assert math.isfinite(saved) and math.isfinite(computed), path
        assert abs(saved-computed) <= 8*max(math.ulp(saved), math.ulp(computed)), path
    else:
        assert saved == computed, path


def main():
    out = ROOT / 'results'
    evidence = json.loads((out / 'EVIDENCE_MANIFEST.json').read_text())
    for name, expected in evidence['files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    old = ROOT.parent / 'lite_selector_20260920'
    assert (ROOT / 'workloads.jsonl').read_bytes() == (old / 'workloads.jsonl').read_bytes()
    for name in ('execution.py', 'workloads.py'):
        assert (ROOT/'src'/name).read_bytes() == (old/'src'/name).read_bytes()
    configs = [json.loads(s) for s in (ROOT/'workloads.jsonl').read_text().splitlines()]
    assert len(configs) == 96 and len({c['parent_id'] for c in configs}) == 24
    assert {s: sum(c['split'] == s for c in configs) for s in ('train','validation','test')} == {
        'train': 48, 'validation': 16, 'test': 32}
    assert all(float(c['threshold_d2']).hex() == c['threshold_hex'] for c in configs)
    allowed = [c for c in configs if c['split'] != 'test']
    counts = {'reference': 48, 'probe': 0, 'regression': 68, 'validate': 512,
              'memcheck': 16, 'racecheck': 16, 'initcheck': 16, 'synccheck': 16, 'stress': 132}
    labels = {f'g0_{name}_v2': n for name, n in counts.items()}
    labels.update({f'g1_bench_{p}': 1600 for p in range(4)})
    runs = {}
    first = None
    for label, n in labels.items():
        run = json.loads((out/(label+'.json')).read_text()); runs[label] = run
        first = first or run
        assert run['pass'] and 'error' not in run and len(run['records']) == n, label
        assert run['invalid_input_cases_rejected'] == 9
        assert run['source_sha256'] == first['source_sha256']
        assert run['environment'] == first['environment']
        assert run['contract_sha256'] == sha(ROOT/'CONTRACT.yaml')
        raw = [json.loads(s) for s in (out/(label+'.jsonl')).read_text().splitlines()]
        assert raw == run['records'], label
        compiled = json.loads((out/(label+'.compiled.json')).read_text())
        known = {v['cubin_sha256'] for v in compiled.values()}
        assert all(v['spills'] == 0 for v in compiled.values())
        for row in raw:
            assert row['correct'] and row['split'] != 'test'
            assert set(row['kernels'].values()) <= known
            assert row['threshold_hex'] == float(row['threshold_d2']).hex()
            assert row['feature_ms'] == row['inference_ms'] == 0
            assert row['selected_format'] == row['method']
            assert 0 <= row['output_count'] <= row['n']**2
            if row['method'] != 'fp64':
                assert row['stage1_accept'] + row['stage1_reject'] + row['fp32_pairs'] == row['n']*(row['n']+1)//2
    assert len(first['source_sha256']) == 11
    for name, expected in first['source_sha256'].items():
        assert sha(ROOT.parent/name) == expected, name
    probe = runs['g0_probe_v2']
    assert probe['native']['e3m4']['raw_0x10_self_dot_k64'] == 4
    assert probe['native']['e4m3']['raw_0x10_self_dot_k64'] == .0625
    assert all(v['metadata_and_intervals'] and v['pairs_checked'] == 97**2 for v in probe['numerical'].values())
    for name in ('memcheck','racecheck','initcheck','synccheck'):
        log = (out/f'g0_{name}_v2.sanitizer.log').read_text()
        assert ('0 hazards displayed (0 errors, 0 warnings)' if name == 'racecheck' else 'ERROR SUMMARY: 0 errors') in log
    stress = runs['g0_stress_v2']
    assert stress['distinct_input_addresses'] >= 16
    for case in {r['config_id'] for r in stress['records']}:
        rows = [r for r in stress['records'] if r['config_id'] == case]
        assert [r['method'] for r in rows] == ['fp64'] + ['int8','e3m4']*16
        assert len({r['output_sha256'] for r in rows}) == 1
    # Replay every recorded threshold-neighbor CPU qualification, including the
    # immutable v1 counterexample. The GPU full-array checks were done on device.
    saved = np.load(old/'results/g0_failure_subset_v1.npy', allow_pickle=False)
    random = np.random.Generator(np.random.PCG64(20260920)).uniform(-1,1,(33,512)).astype(np.float32)
    cases = [('saved_counterexample',saved,25,28), ('random_pair_1',random,1,2),
             ('random_pair_2',random,3,17), ('random_pair_3',random,14,32)]
    qual = runs['g0_reference_v2']
    for name, x, i, j in cases:
        dist = terminal_order_distances(x); center = float(dist[i,j])
        for suffix, t in [('below',float(np.nextafter(center,-np.inf))), ('equal',center),
                          ('above',float(np.nextafter(center,np.inf)))]:
            expected = np.flatnonzero(dist <= t).astype(np.uint64)
            rows = [r for r in qual['records'] if r['config_id'] == name+'_'+suffix]
            assert len(rows) == 4
            assert all(r['threshold_hex'] == t.hex() and r['output_count'] == len(expected)
                       and r['output_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest() for r in rows)
    assert len(qual['cpu_graph_audits']) == 12 and all(a['graph_pass'] for a in qual['cpu_graph_audits'])
    ptx_map = json.loads((out/'reference_ptx/manifest.json').read_text())['original_ptx_sha256_to_public_sha256']
    assert set(ptx_map) == {a['ptx_sha256'] for a in qual['cpu_graph_audits']}
    assert len(list((out/'reference_ptx').glob('*.ptx'))) == 12
    for ptx in (out/'reference_ptx').glob('*.ptx'):
        assert sha(ptx) == ptx_map[ptx.stem]
        text = ptx.read_text()
        masks = list(map(int,re.findall(r'shfl\.sync\.bfly\.b32\s+[^,]+,\s+[^,]+,\s+(\d+),\s+31',text)))
        assert masks == [16,16,8,8,4,4,2,2,1,1,2,2,1,1]*2
        assert '.reqntid 128' in text and 'fma.rn.f64' not in text
        assert [text.count(s) for s in ('sub.rn.f64','mul.rn.f64','add.rn.f64')] == [4,4,18]
    # Check full validation groups, then replay the exact randomized method order.
    valid = runs['g0_validate_v2']['records']
    for cfg in allowed:
        rows = [r for r in valid if r['config_id'] == cfg['config_id']]
        assert len(rows) == 8
        for scope in ('cpu_subset','full'):
            group = [r for r in rows if (r['role'] == 'cpu_subset') == (scope == 'cpu_subset')]
            assert len(group) == 4 and len({r['output_sha256'] for r in group}) == 1
    for p in range(4):
        run = runs[f'g1_bench_{p}']; assert run['process'] == p
        rng = np.random.Generator(np.random.PCG64(20260920+p))
        for index, cfg in enumerate(allowed):
            rows = run['records'][index*25:(index+1)*25]
            assert len(rows) == 25 and all(r['config_id'] == cfg['config_id'] for r in rows)
            assert all(r['input_sha256'] == cfg['input_sha256'] and r['threshold_hex'] == cfg['threshold_hex'] for r in rows)
            assert len({r['output_sha256'] for r in rows}) == 1
            assert [r['role'] for r in rows[:4]] == ['reference'] + ['compiler_preparation']*3
            for rep in range(7):
                order = list(rng.permutation(METHODS))
                group = rows[4+rep*3:7+rep*3]
                assert [r['method'] for r in group] == order
                for pos, row in enumerate(group):
                    assert row['order'] == order and row['position'] == pos and row['rep'] == rep
                    assert row['role'] == 'sample' and row['warmup'] == (rep < 2)
                    assert row['order_seed'] == 20260920+p
    summary = json.loads((out/'SUMMARY.json').read_text())
    compare_summary(summary, compute(ROOT))
    assert summary['test_timing_records'] == 0
    supervision = json.loads((out/'GPU_SUPERVISION.json').read_text())
    assert supervision['guard']['pass'] and supervision['guard']['exit_code'] == 0
    if not summary['G1_pass']:
        assert json.loads((ROOT/'model.json').read_text())['status'] == 'not_trained'
    print(json.dumps({'artifact_consistent': True, 'G0_fixed_paths': 'PASS', 'G1_pass': summary['G1_pass'],
                      'source_files': 11, 'benchmark_calls': 6400,
                      'held_out_timing_records': 0, 'scope': 'offline replay, not a new GPU run'}))


if __name__ == '__main__':
    main()
