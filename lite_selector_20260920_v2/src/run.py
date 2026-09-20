"""Gated G0/G1 execution. Held-out timing is deliberately not implemented here."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'precision_format_routing' / 'src'
sys.path.insert(0, str(BASE))
os.environ['TENSORJOIN_ARTIFACTS'] = str(ROOT / 'artifacts' / 'native')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / 'artifacts' / 'triton_cache')

import numpy as np
import torch
import triton
from triton.backends.nvidia.compiler import get_ptxas
import experiment as base
from execution import Engine, validate_input, numerical_test
from workloads import records, load
from cpu_reference import terminal_order_distances

METHODS = ['int8', 'e3m4', 'fp16']


def hashes():
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT / 'src', BASE) for p in sorted(folder.glob('*.py'))}


def environment():
    try:
        sklearn = importlib.metadata.version('scikit-learn')
    except importlib.metadata.PackageNotFoundError:
        sklearn = None
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
            'triton': triton.__version__, 'sklearn': sklearn, 'ptxas': get_ptxas(120).version,
            'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'gpu_identity_sha256': hashlib.sha256(os.environ['CUDA_VISIBLE_DEVICES'].encode()).hexdigest(),
            'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}


def cpu_state():
    stat = Path('/proc/stat').read_text().splitlines()
    return {'time_ns': time.time_ns(), 'cpu_ticks': list(map(int, stat[0].split()[1:])),
            'loadavg': Path('/proc/loadavg').read_text().strip(),
            'processes_running': int(next(s for s in stat if s.startswith('procs_running ')).split()[1])}


def check(ids, oracle, n):
    assert ids.dtype == np.uint64 and np.array_equal(ids, oracle), 'Full canonical ID mismatch'
    assert len(ids) <= n*n and (len(ids) == 0 or ids[-1] < n*n)
    assert np.all(ids[1:] > ids[:-1])
    assert np.all(np.isin(np.arange(n, dtype=np.uint64)*(n+1), ids)), 'Missing self pairs'


def cpu_oracle(x, threshold):
    return np.flatnonzero(terminal_order_distances(x) <= threshold).astype(np.uint64)


def extra_fixtures():
    x = np.zeros((33, 512), np.float32); x[1::2, 0] = 1; x[2::4, 0] = -1
    yield 'signed_abs1_duplicate33', x, 1.0
    yield 'zero33', np.zeros_like(x), 0.0
    z = np.zeros_like(x); z[1::2, 0] = np.finfo(np.float32).smallest_subnormal
    yield 'underflow33', z, 0.0


def invalid_checks():
    good = np.zeros((31, 512), np.float32)
    cases = [(good.astype(np.float64), 0.), (good[:, ::2], 0.), (good[::-1], 0.),
             (good, -1.), (good, float('inf')), (good, np.float32(0))]
    for value in (np.nan, np.inf, 1.01):
        bad = good.copy(); bad[0, 0] = value; cases.append((bad, 0.))
    for x, t in cases:
        try:
            validate_input(x, t)
        except ValueError:
            continue
        raise AssertionError('Invalid input accepted')
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference','probe','regression','validate','sanitizer','stress','bench'], required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--process', type=int, default=0)
    args = parser.parse_args()
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    path = out / (args.label + '.json')
    if path.exists(): raise FileExistsError(path.name)
    raw = (out / (args.label + '.jsonl')).open('x')
    init = time.perf_counter(); torch.cuda.init(); torch.cuda.synchronize()
    env = environment()
    assert env['capability'] == (12, 0) and env['ptxas'] == '13.1'
    assert (env['numpy'], env['torch'], env['triton']) == ('2.3.1','2.11.0+cu130','3.6.0')
    result = {'label': args.label, 'mode': args.mode, 'process': args.process,
              'environment': env, 'context_init_ms': (time.perf_counter()-init)*1e3,
              'source_sha256': hashes(), 'contract_sha256': hashlib.sha256((ROOT/'CONTRACT.yaml').read_bytes()).hexdigest(),
              'records': [], 'pass': False}
    engine = Engine(); engine.churn = args.mode == 'stress'
    rng = np.random.Generator(np.random.PCG64(20260920+args.process))
    def save(row):
        result['records'].append(row); raw.write(json.dumps(row)+'\n'); raw.flush()
        print(json.dumps(row), flush=True)
    def call(x, t, method, meta, role, oracle=None, **extra):
        result['current_call'] = dict(meta, method=method, role=role, n=len(x),
            threshold_hex=float(t).hex(), input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        before = cpu_state()
        ids, row = engine.run(x, t, method)
        after = cpu_state()
        if oracle is not None:
            try:
                check(ids, oracle, len(x))
            except AssertionError:
                row.update(correct=False, extra_ids=np.setdiff1d(ids,oracle).tolist(),
                           missing_ids=np.setdiff1d(oracle,ids).tolist())
                save(dict(meta, **row, method=method, role='failed_'+role,
                          threshold_hex=float(t).hex(), kernels=engine.record_kernels()))
                raise
        selected = engine.record_kernels()
        save(dict(meta, **row, threshold_hex=float(t).hex(), role=role, correct=True,
                  method=method, kernels=selected, cpu_before=before, cpu_after=after, **extra))
        return ids
    def subset_check(x, t, meta):
        idx = np.sort(np.random.Generator(np.random.PCG64(31337)).choice(len(x), min(32,len(x)), replace=False))
        small = x[idx].copy(); expected = cpu_oracle(small, t)
        for fmt in ['fp64'] + METHODS:
            call(small, t, fmt, meta, 'cpu_subset', expected)
    try:
        result['invalid_input_cases_rejected'] = invalid_checks()
        if args.mode == 'reference':
            import re
            old = ROOT.parent/'lite_selector_20260920'/'results'
            x = np.load(old/'g0_failure_subset_v1.npy', allow_pickle=False)
            d = json.loads((old/'g0_failure_diagnosis_v1.json').read_text())
            assert hashlib.sha256(x.tobytes()).hexdigest() == d['subset_input_sha256']
            random = np.random.Generator(np.random.PCG64(20260920)).uniform(-1,1,(33,512)).astype(np.float32)
            fixtures = [('saved_counterexample',x,25,28),
                        ('random_pair_1',random,1,2),('random_pair_2',random,3,17),
                        ('random_pair_3',random,14,32)]
            audits=[]
            for name, a, i, j in fixtures:
                center = float(terminal_order_distances(a)[i,j])
                for label, t in [('below',float(np.nextafter(center,-np.inf))),
                                 ('equal',center),('above',float(np.nextafter(center,np.inf)))]:
                    expected = cpu_oracle(a,t)
                    for method in ['fp64']+METHODS:
                        call(a,t,method,{'config_id':name+'_'+label,'split':'reference_qualification'},
                             'cpu_graph_qualification',expected)
                        if method == 'fp64':
                            kernel=engine.kernels['fp64']; ptx=kernel.asm['ptx']
                            masks=list(map(int,re.findall(r'shfl\.sync\.bfly\.b32\s+[^,]+,\s+[^,]+,\s+(\d+),\s+31',ptx)))
                            assert masks == [16,16,8,8,4,4,2,2,1,1,2,2,1,1]*2
                            assert '.reqntid 128' in ptx and ptx.count('mul.rn.f64')==4
                            assert ptx.count('sub.rn.f64')==4 and ptx.count('add.rn.f64')==18
                            assert re.search(r'and.b32\s+[^,]+,\s+[^,]+,\s+254;',ptx)
                            assert 'fma.rn.f64' not in ptx
                            audits.append({'case':name+'_'+label,'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                                           'cubin_sha256':hashlib.sha256(kernel.kernel).hexdigest(),'graph_pass':True})
            result['cpu_graph_audits']=audits
        elif args.mode == 'probe':
            result['native'] = base.native_probe(engine)
            assert result['native']['e3m4']['raw_0x10_self_dot_k64'] == 4.0
            assert result['native']['e4m3']['raw_0x10_self_dot_k64'] == .0625
            result['producer'] = base.producer_test(engine)
            result['numerical'] = numerical_test(engine)
            engine.record_kernels()
        elif args.mode in ('regression','sanitizer','stress'):
            for name, x, t in list(base.fixtures()) + list(extra_fixtures()):
                if args.mode in ('sanitizer','stress'):
                    if name not in ('cifar4096','boundary_zero32','signed_abs1_duplicate33','underflow33'): continue
                    x = x[:512].copy()
                meta = {'config_id': name, 'split': 'regression'}
                oracle = call(x, t, 'fp64', meta, 'reference')
                if len(x) <= 33: check(oracle, cpu_oracle(x, t), len(x))
                sequence = (['int8','e3m4']*16 if args.mode == 'stress' else METHODS)
                for rep, fmt in enumerate(sequence):
                    call(x, t, fmt, meta, 'stress' if args.mode == 'stress' else 'validation', oracle, rep=rep)
            if args.mode == 'stress':
                result['distinct_input_addresses'] = len(engine.addresses)
                assert len(engine.addresses) >= 16
        else:
            for meta in records():
                assert meta['split'] in ('train','validation'), 'Held-out performance is sealed'
                x = load(meta); t = meta['threshold_d2']
                identity = {k: meta[k] for k in ('config_id','parent_id','family','seed','split','target_k')}
                oracle = call(x, t, 'fp64', identity, 'reference')
                if args.mode == 'validate':
                    subset_check(x, t, identity)
                    for fmt in METHODS: call(x, t, fmt, identity, 'validation', oracle)
                else:
                    for fmt in METHODS: call(x, t, fmt, identity, 'compiler_preparation', oracle)
                    for rep in range(7):
                        order = list(rng.permutation(METHODS))
                        for position, fmt in enumerate(order):
                            call(x, t, fmt, identity, 'sample', oracle, rep=rep, warmup=rep<2,
                                 order=order, position=position, order_seed=20260920+args.process)
        result['pass'] = True
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        raw.close(); path.write_text(json.dumps(result, indent=2)+'\n')
        engine.capture(out / (args.label + '.compiled.json'))
    print(json.dumps({'pass': True, 'label': args.label}), flush=True)


if __name__ == '__main__': main()
