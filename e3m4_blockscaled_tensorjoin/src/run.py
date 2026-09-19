"""Run the bounded native block-scaled TensorJoin campaign; labels are immutable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'precision_format_routing' / 'src'
sys.path.insert(0, str(BASE))
os.environ['TENSORJOIN_ARTIFACTS'] = str(ROOT / 'artifacts' / 'plain')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / 'artifacts' / 'triton_cache')

import numpy as np
import torch
import triton
from triton.backends.nvidia.compiler import get_ptxas
import experiment as base
from backend import Engine
from checks import native_and_layout, producer_and_intervals

METHODS = ['int8', 'fp16', 'e4m3', 'e3m4', 'mx_e4m3', 'mx_e3m4']
BENCH_CASES = ['cifar4096', 'synthetic_clustered', 'synthetic_outlier']


def source_hashes():
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (ROOT / 'src', BASE) for p in sorted(directory.glob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['probe', 'validate', 'stress', 'bench'], required=True)
    parser.add_argument('--label', required=True); parser.add_argument('--order', type=int, default=0)
    parser.add_argument('--case', default='all'); parser.add_argument('--limit-n', type=int, default=0)
    args = parser.parse_args()
    result_dir = ROOT / 'results'; result_dir.mkdir(exist_ok=True)
    path = result_dir / (args.label + '.json')
    if path.exists(): raise FileExistsError(path.name)
    raw = (result_dir / (args.label + '.jsonl')).open('x')
    assert torch.cuda.get_device_capability() == (12, 0)
    compiler = get_ptxas(120); assert compiler.version == '13.1'
    result = {'mode': args.mode, 'label': args.label, 'order': args.order,
              'torch': torch.__version__, 'triton': triton.__version__,
              'device': torch.cuda.get_device_name(), 'ptxas_version': compiler.version,
              'source_sha256': source_hashes(), 'records': [], 'pass': False}
    engine = Engine(); engine.churn = args.mode == 'stress'
    def save(rec):
        result['records'].append(rec); raw.write(json.dumps(rec)+'\n'); raw.flush()
        print(json.dumps(rec), flush=True)
    try:
        if args.mode == 'probe':
            result['native_layout'] = native_and_layout(engine)
            result['producer_intervals_graph_streams'] = producer_and_intervals(engine)
        else:
            for name, x, threshold in base.fixtures():
                if args.case != 'all' and args.case != name: continue
                if args.mode == 'bench' and name not in BENCH_CASES: continue
                if args.limit_n and len(x) > args.limit_n:
                    x = x[:args.limit_n].copy(); name += f'_prefix{args.limit_n}'
                if args.mode == 'stress' and len(x) > 512:
                    x = x[:512].copy(); name += '_prefix512'
                oracle, rec = engine.run(x, threshold, 'fp64')
                save(dict(case=name, role='reference', **rec))
                if len(x) <= 32:
                    delta = x.astype(np.float64)[:, None, :]-x.astype(np.float64)[None, :, :]
                    expected = np.flatnonzero(np.sum(delta*delta, axis=2) <= threshold).astype(np.uint64)
                    assert np.array_equal(oracle, expected), (name, 'CPU oracle')
                methods = METHODS[::-1] if args.order % 2 else METHODS
                if args.mode == 'bench':
                    for fmt in methods:
                        got, rec = engine.run(x, threshold, fmt)
                        assert np.array_equal(got, oracle)
                        save(dict(case=name, role='compiler_preparation', **rec))
                for fmt in methods:
                    repeats = 16 if args.mode == 'stress' else (7 if args.mode == 'bench' else 1)
                    for rep in range(repeats):
                        got, rec = engine.run(x.copy(), threshold, fmt)
                        assert np.array_equal(got, oracle), (name, fmt, 'exact IDs')
                        save(dict(case=name, role='sample', rep=rep,
                                  warmup=args.mode == 'bench' and rep < 2, exact_ids=True, **rec))
                if args.mode == 'bench' and name == 'cifar4096':
                    for fmt in methods:
                        batch = []; start = time.perf_counter()
                        for rep in range(32):
                            got, rec = engine.run(x, threshold, fmt)
                            assert np.array_equal(got, oracle)
                            batch.append(dict(case=name, role='sustained', rep=rep, exact_ids=True, **rec))
                        elapsed = (time.perf_counter()-start)*1e3
                        for rec in batch: save(rec)
                        save({'case': name, 'format': fmt, 'role': 'batch_wall', 'calls': 32,
                              'wall_ms': elapsed, 'mean_call_wall_ms': elapsed/32,
                              'scope': 'includes outside-run identity registry and exact-ID assertion; excludes evidence serialization'})
        if args.mode == 'stress':
            result['distinct_gpu_input_addresses'] = len(engine.addresses)
            assert len(engine.addresses) >= 16
        result['pass'] = True
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        raw.close(); path.write_text(json.dumps(result, indent=2))
        engine.capture(result_dir / (args.label + '.compiled.json'))
    print(json.dumps({'pass': True, 'label': args.label}), flush=True)


if __name__ == '__main__': main()
