"""Independent CPU enclosure and bitwise attribution of G16 GPU metadata."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from g2b_public_common import atomic_json, load_pageable_source, sha256_file
from run_g4b_r1_public_opportunity import quantize_with_analytic_residual
from gpu_preparation import prepare_device

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent


def data_cases(source):
    rng = np.random.default_rng(16052026)
    yield 'cifar60000', source
    yield 'signed257', rng.uniform(-1, 1, (257, 512)).astype(np.float32)
    yield 'zeros31', np.zeros((31, 512), np.float32)
    one_hot = np.zeros((129, 512), np.float32)
    one_hot[np.arange(129), np.arange(129)] = 1
    yield 'onehot129', one_hot
    yield 'cancellation31', (np.float32(.5) + rng.integers(-2, 3, (31, 512)).astype(np.float32)*np.float32(2.0**-12))
    tiny = np.nextafter(np.float32(0), np.float32(1))
    yield 'subnormal31', rng.integers(-16384, 16385, (31, 512)).astype(np.float32)*tiny
    exponent = rng.integers(-149, 0, (257, 512))
    yield 'mixed_exponents257', np.ldexp(rng.uniform(-1, 1, (257, 512)), exponent).astype(np.float32)


def check(vectors):
    device = torch.from_numpy(vectors).to('cuda')
    outputs = prepare_device(device)
    gpu = tuple(x.cpu().numpy() for x in outputs)
    reference = quantize_with_analytic_residual(vectors)
    q, s, norm, errors, transposed = gpu
    bound_violations = norm_mismatches = 0
    for start in range(0, len(vectors), 4096):
        stop = min(start+4096, len(vectors))
        q64, s64 = q[start:stop].astype(np.int64), s[start:stop].astype(np.float64)
        reconstructed = q64.astype(np.float64)*s64[:, None]
        residual = vectors[start:stop].astype(np.float64) - reconstructed
        norm_reference = (np.sum(q64*q64, axis=1).astype(np.float64)*s64**2).astype(np.float32)
        residual_reference = np.sqrt(np.sum(residual*residual, axis=1, dtype=np.float64))
        norm_mismatches += int(np.count_nonzero(norm_reference.view(np.uint32) != norm[start:stop].view(np.uint32)))
        bound_violations += int(np.count_nonzero(errors[start:stop].astype(np.float64) < residual_reference))
    differences = {name: int(np.count_nonzero(actual.view(np.uint8) != expected.view(np.uint8)))
                   for name, actual, expected in zip(('codes', 'scales', 'norms', 'errors'), gpu, reference)}
    transpose_equal = bool(np.array_equal(transposed, q.T))
    return {'n': len(vectors), 'cpu_original_byte_differences': differences,
            'reconstruction_norm_bitwise_mismatches': norm_mismatches,
            'residual_enclosure_violations': bound_violations,
            'transpose_exact': transpose_equal,
            'output_hashes': [hashlib.sha256(a.tobytes()).hexdigest() for a in gpu],
            'input_device_pointer': int(device.data_ptr()),
            'pass': norm_mismatches == 0 and bound_violations == 0 and transpose_equal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()
    if not args.record_id.replace('_', '').isalnum():
        raise ValueError('Unsafe record ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('Expected GPU 2')
    gate = json.loads((PROJECT/'g15_strong_control_20260905/results/public_screen.json').read_text())
    assert gate['complete'], 'G15 public screen must complete before G16 GPU work'
    source = load_pageable_source()
    assert sha256_file(PROJECT/'data/g2b_cifar60000/vectors_f32.npy') == '95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c'
    records = []
    if args.stress:
        vectors = (source[:129].copy(), source[257:386].copy())
        expected = [check(x)['output_hashes'] for x in vectors]
        pointers = set()
        for i in range(1000):
            churn = torch.empty((129, 512), dtype=torch.float32, device='cuda') if i % 2 else None
            result = check(vectors[i % 2])
            pointers.add(result['input_device_pointer'])
            if not result['pass'] or result['output_hashes'] != expected[i % 2]:
                records.append({'iteration': i, **result})
                break
            del churn
            if (i+1) % 100 == 0:
                print('METADATA_STRESS ' + str(i+1), flush=True)
        passed = i == 999 and not records and len(pointers) >= 2
        details = {'iterations': i+1, 'pointers': sorted(pointers), 'failures': records}
    else:
        for name, vectors in data_cases(source):
            record = {'name': name, **check(vectors)}
            records.append(record)
            print('METADATA_CASE ' + json.dumps(record), flush=True)
            if not record['pass']:
                break
        passed = len(records) == 7 and all(r['pass'] for r in records)
        details = {'cases': records}
    result = {'experiment_id': 'tensorjoin_20260905_g16_gpu_metadata', 'details': details,
              'correctness': {'exact_contract_pass': passed},
              'scope': 'metadata/enclosure only; not a full join or timing claim',
              'runner_sha256': sha256_file(Path(__file__)),
              'kernel_sha256': sha256_file(HERE/'src/gpu_preparation.py')}
    atomic_json(HERE/f'results/metadata_{args.record_id}.json', result)
    return 0 if passed else 2


if __name__ == '__main__':
    main()
