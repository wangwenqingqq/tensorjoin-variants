"""One immutable G15-A CPU preparation process; no CUDA initialization."""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from g2b_public_common import atomic_json, load_pageable_source, sha256_file
from preparation import original_prepare, chunked_prepare

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
EXPECTED_SOURCE = '95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
EXPECTED_QUANTIZER = 'c1d539458233fb3268149e74847d8e6a5c8f4e3a3400954e7a5ee44a74f36234'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--variant', choices=('original', 'chunked'), required=True)
    args = parser.parse_args()
    if not args.record_id.replace('_', '').isalnum():
        raise ValueError('Unsafe record ID')
    path = HERE / f'results/preparation_{args.record_id}.json'
    if path.exists():
        raise FileExistsError(path)
    source = PROJECT / 'data/g2b_cifar60000/vectors_f32.npy'
    assert sha256_file(source) == EXPECTED_SOURCE
    assert sha256_file(PROJECT / 'src/run_g4b_r1_public_opportunity.py') == EXPECTED_QUANTIZER
    vectors = load_pageable_source()
    function = {'original': original_prepare, 'chunked': chunked_prepare}[args.variant]
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    prepared = function(vectors)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    # The complete original construction follows the measured construction.
    reference = original_prepare(vectors)
    checks = []
    for name, actual, expected in zip(('codes', 'scales', 'norm2', 'residual', 'codes_T'), prepared, reference):
        exact = (actual.shape == expected.shape and actual.dtype == expected.dtype
                 and actual.tobytes() == expected.tobytes())
        checks.append({'name': name, 'shape': list(actual.shape), 'dtype': str(actual.dtype),
                       'bitwise_equal': exact,
                       'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    result = {'experiment_id': 'tensorjoin_20260905_g15a_host_preparation',
              'diagnostic_only': True, 'variant': args.variant, 'record_id': args.record_id,
              'host': platform.node(), 'python': sys.version, 'numpy': np.__version__,
              'affinity': sorted(os.sched_getaffinity(0)), 'loadavg': list(os.getloadavg()),
              'scope': 'pageable FP32 source -> four original code/metadata arrays plus contiguous transpose',
              'source_sha256': EXPECTED_SOURCE, 'preparation_seconds': elapsed,
              'cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
              'minor_faults': after.ru_minflt-before.ru_minflt,
              'major_faults': after.ru_majflt-before.ru_majflt,
              'peak_rss_kib_after_timing': after.ru_maxrss,
              'checks': checks, 'all_bitwise_equal': all(c['bitwise_equal'] for c in checks),
              'runner_sha256': sha256_file(Path(__file__)),
              'preparation_source_sha256': sha256_file(HERE / 'src/preparation.py'),
              'protocol_sha256': sha256_file(HERE / 'PLAN_AND_GATE0.md')}
    atomic_json(path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['all_bitwise_equal'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
