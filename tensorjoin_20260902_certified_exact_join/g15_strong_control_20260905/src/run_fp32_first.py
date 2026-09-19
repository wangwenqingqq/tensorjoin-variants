"""Validate, stress or measure one complete G15 FP32-first process."""

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from g2b_public_common import atomic_json, load_contract, load_pageable_source, sha256_file, validate_canonical
from fp32_first import (BoundCuBLAS, norm_metadata, check_input, warmup, run_operator,
                        classify, cache_identity, CAPACITY, GAMMA, ABS_DOT, DIMENSION)

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
EXPECTED = {
    'src/run_h2b_p1_pedantic_correctness.py': 'd747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1',
    'src/run_g2a_tensorjoin.py': '84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8',
}


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def oracle(vectors, threshold):
    n = len(vectors)
    v = vectors.astype(np.float64)
    distances = np.empty((n, n), np.float64)
    for row in range(0, n, 16):
        delta = v[row:row+16, None, :] - v[None, :, :]
        distances[row:row+16] = np.sum(delta*delta, axis=2, dtype=np.float64)
    ids = np.flatnonzero(distances.reshape(-1) <= threshold).astype(np.uint64)
    return ids, distances


def fixtures(source, threshold):
    rng = np.random.default_rng(15052026)
    yield 'real257', source[:257].copy(), threshold
    yield 'signed31', rng.normal(0, .02, (31, DIMENSION)).astype(np.float32), threshold
    yield 'zero1', np.zeros((1, DIMENSION), np.float32), 0.0
    yield 'ties129', np.zeros((129, DIMENSION), np.float32), 0.0
    cancellation = (np.float32(.5) + rng.integers(-2, 3, (31, DIMENSION)).astype(np.float32)*np.float32(2.0**-12))
    yield 'cancellation31', cancellation, 2.0**-10
    boundary = np.zeros((31, DIMENSION), np.float32)
    boundary[0, 0] = 1
    boundary[1, :2] = [1, 2.0**-27]
    boundary[2, 0] = -1
    for i in range(3, 31):
        boundary[i, i] = 1
    yield 'reference_not_exact_real31', boundary, 1.0
    tiny = np.nextafter(np.float32(0), np.float32(1))
    subnormal = rng.integers(-16, 17, (31, DIMENSION)).astype(np.float32) * tiny
    yield 'subnormal31', subnormal, 0.0


def interval_check(vectors, threshold, blas, distance):
    n = len(vectors)
    vg = torch.from_numpy(vectors).to('cuda')
    host = norm_metadata(vectors)
    metadata = tuple(torch.from_numpy(x).to('cuda') for x in host)
    dots = torch.empty((n, n), device='cuda', dtype=torch.float32)
    lower, upper = (torch.empty((n, n), device='cuda', dtype=torch.float64) for _ in range(2))
    result = torch.empty(CAPACITY, device='cuda', dtype=torch.int64)
    ambiguous = torch.empty_like(result)
    counter = torch.zeros(4, device='cuda', dtype=torch.int32)
    blas.gemm(vg, vg, dots)
    classify(dots, metadata, result, ambiguous, counter, 0, 0, threshold, n, lower, upper)
    lo, hi, pdot = lower.cpu().numpy(), upper.cpu().numpy(), dots.cpu().numpy()
    reference_dot = np.einsum('ik,jk->ij', vectors.astype(np.float64), vectors.astype(np.float64), optimize=False)
    dot_radius = GAMMA * host[2][:, None] * host[2][None, :] + ABS_DOT
    containment = int(np.count_nonzero((lo > distance) | (hi < distance)))
    unsafe = int(np.count_nonzero(((hi <= threshold) & (distance > threshold)) |
                                 ((lo > threshold) & (distance <= threshold))))
    dot_errors = int(np.count_nonzero(np.abs(pdot.astype(np.float64)-reference_dot) > dot_radius))
    # A true rectangle checks the row-major/column-major FFI mapping independently.
    qn, bn = min(13, n), min(19, n)
    rectangle = torch.empty((qn, bn), device='cuda', dtype=torch.float32)
    blas.gemm(vg[:qn], vg[n-bn:], rectangle)
    rdot = rectangle.cpu().numpy().astype(np.float64)
    rref = np.einsum('ik,jk->ij', vectors[:qn].astype(np.float64), vectors[n-bn:].astype(np.float64), optimize=False)
    rrad = GAMMA*host[2][:qn, None]*host[2][None, n-bn:] + ABS_DOT
    rectangular_errors = int(np.count_nonzero(np.abs(rdot-rref) > rrad))
    return {'pairs_checked': n*n, 'containment_violations': containment,
            'unsafe_direct_decisions': unsafe, 'dot_radius_violations': dot_errors,
            'rectangular_dot_violations': rectangular_errors,
            'max_dot_absolute_error': float(np.max(np.abs(pdot-reference_dot))),
            'all_pass': containment == unsafe == dot_errors == rectangular_errors == 0}


def validation(source, threshold, blas):
    records = []
    for name, vectors, radius in fixtures(source, threshold):
        check_input(vectors, radius)
        expected, distance = oracle(vectors, radius)
        bounds = interval_check(vectors, radius, blas, distance)
        warmup(len(vectors), radius, blas)
        actual, counts = run_operator(vectors, radius, blas)
        record = {'fixture': name, 'n': len(vectors), 'threshold': radius,
                  'source_sha256': sha_array(vectors), 'bounds': bounds,
                  'output_equal': bool(np.array_equal(actual, expected)),
                  'actual_hash': sha_array(actual), 'oracle_hash': sha_array(expected),
                  'pairs': int(actual.size), 'work_counts': counts,
                  'diagnostic_times_not_promoted': True}
        records.append(record)
        print('FIXTURE ' + json.dumps({k:v for k,v in record.items() if k != 'work_counts'}), flush=True)
        if not bounds['all_pass'] or not record['output_equal']:
            break
    passed = len(records) == 7 and all(r['output_equal'] and r['bounds']['all_pass'] for r in records)
    return {'fixtures': records, 'correctness': {'exact_contract_pass': passed},
            'scope': 'small real/adversarial reference equivalence, not full public performance'}


def stress(source, threshold, blas):
    inputs = (source[:129].copy(), source[257:386].copy())
    expected = [sha_array(oracle(v, threshold)[0]) for v in inputs]
    warmup(129, threshold, blas)
    pointers, count_signatures, failures = set(), [set(), set()], []
    started = time.perf_counter()
    for iteration in range(1000):
        churn = (torch.empty((129, DIMENSION), device='cuda', dtype=torch.float32)
                 if iteration % 2 else None)
        actual, record = run_operator(inputs[iteration % 2], threshold, blas)
        pointers.add(record['input_device_pointer'])
        signature = tuple(record[k] for k in ('ambiguous_upper_pairs', 'direct_accept_upper_pairs',
                                             'direct_reject_upper_pairs', 'accepted_upper_pairs', 'overflow_events'))
        count_signatures[iteration % 2].add(signature)
        if sha_array(actual) != expected[iteration % 2] or record['overflow_events']:
            failures.append({'iteration': iteration, 'record': record, 'output_sha256': sha_array(actual)})
            break
        del churn
        if (iteration+1) % 100 == 0:
            print('STRESS_PROGRESS ' + str(iteration+1), flush=True)
    passed = (iteration == 999 and not failures and len(pointers) >= 2
              and all(len(s) == 1 for s in count_signatures))
    return {'iterations_completed': iteration+1, 'distinct_input_device_pointers': sorted(pointers),
            'per_input_count_signatures': [sorted(s) for s in count_signatures],
            'expected_output_hashes': expected, 'failures': failures,
            'stress_wall_seconds_not_performance': time.perf_counter()-started,
            'scope': '1000 complete N129 calls, alternating input/pointer churn; default stream only',
            'correctness': {'exact_contract_pass': passed}}


def trace(source, blas, shape_index):
    shapes = ((4096, 4096), (4096, 2656), (2656, 2656))
    rows, columns = shapes[shape_index]
    q = torch.from_numpy(source[:rows]).to('cuda')
    b = torch.from_numpy(source[-columns:]).to('cuda')
    out = torch.empty((rows, columns), device='cuda', dtype=torch.float32)
    label = f'G15_PEDANTIC_{rows}_{columns}_512'
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    blas.gemm(q, b, out)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    # A probe is precision/runtime-binding evidence, not a full-output oracle.
    observed = out[:8, :8].cpu().numpy().astype(np.float64)
    v, w = source[:8].astype(np.float64), source[-columns:][:8].astype(np.float64)
    expected = np.einsum('ik,jk->ij', v, w, optimize=False)
    bound = GAMMA*np.linalg.norm(v, axis=1)[:, None]*np.linalg.norm(w, axis=1)[None, :] + ABS_DOT
    ok = bool(np.all(np.abs(observed-expected) <= bound))
    return {'shape': [rows, columns, DIMENSION], 'nvtx_label': label,
            'scope': 'selected-code trace with a 64-dot numerical probe; not a public exact-output run',
            'correctness': {'exact_contract_pass': ok, 'meaning': '64-dot probe only'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('validation', 'compatibility', 'screen', 'stress', 'trace'), required=True)
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--shape-index', type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if not args.record_id.replace('_', '').isalnum():
        raise ValueError('Unsafe record ID')
    result_path = HERE/f'results/fp32_{args.phase}_{args.record_id}.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('GPU 2 and explicit TF32 override are mandatory')
    for relative, digest in EXPECTED.items():
        if sha256_file(PROJECT/relative) != digest:
            raise RuntimeError('Original dependency drift: ' + relative)
    if platform.node() != 'gpu-host-8':
        raise RuntimeError('Wrong host')
    cache = Path(os.environ['TRITON_CACHE_DIR'])
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()):
        raise RuntimeError('A fresh compiler cache is required')
    contract = load_contract()
    if contract['source_npy_sha256'] != '95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c':
        raise RuntimeError('Source changed')
    source = load_pageable_source()
    threshold = float(contract['threshold_d2'])
    check_input(source, threshold)
    torch.cuda.init()
    blas = BoundCuBLAS()
    common = {'experiment_id': 'tensorjoin_20260905_g15_fp32_first', 'phase': args.phase,
              'record_id': args.record_id, 'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
              'physical_gpu': 2, 'python': sys.version, 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'numpy': np.__version__,
              'library': blas.record(), 'input_contract': contract,
              'runner_sha256': sha256_file(Path(__file__)),
              'operator_sha256': sha256_file(HERE/'src/fp32_first.py'),
              'kernel_sha256': sha256_file(HERE/'src/fp32_first_kernels.py'),
              'plan_sha256': sha256_file(HERE/'PLAN_AND_GATE0.md'),
              'design_sha256': sha256_file(HERE/'BASELINE_DESIGN.md')}
    if args.phase == 'validation':
        result = validation(source, threshold, blas)
    elif args.phase == 'stress':
        result = stress(source, threshold, blas)
    elif args.phase == 'trace':
        result = trace(source, blas, args.shape_index)
    else:
        warmup(len(source), threshold, blas)
        cache_before = cache_identity(cache)
        torch.cuda.reset_peak_memory_stats()
        canonical, result = run_operator(source, threshold, blas)
        cache_after = cache_identity(cache)
        correct = validate_canonical(canonical, contract)
        correct['overflow_events'] = result['overflow_events']
        correct['exact_contract_pass'] = bool(correct['exact_contract_pass'] and not result['overflow_events']
                                              and cache_before == cache_after)
        result.update(correctness=correct, timing_scope='pageable FP32 host -> sorted canonical uint64 host',
                      diagnostic_only=True, cache_before=cache_before, cache_after=cache_after,
                      no_timed_jit=cache_before == cache_after,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    result = {**common, **result, 'compiled_cache': cache_identity(cache)}
    atomic_json(result_path, result)
    print('G15_FP32_COMPLETE ' + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['correctness']['exact_contract_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
