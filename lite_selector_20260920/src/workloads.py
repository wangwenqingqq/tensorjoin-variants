"""Frozen CPU input/threshold construction; no measured costs or test labels."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['clustered', 'outlier', 'logrange', 'mixed']
SPLITS = {'train': [101, 102, 103], 'validation': [201], 'test': [301, 302]}


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def clustered(rng, n, outlier=False):
    # Arithmetic and draw order match the retained synthetic generator.
    x = rng.normal(size=(n, 512)).astype(np.float32) * np.float32(.01)
    x[:, 0] = .5 if outlier else .02
    x += rng.normal(size=(16, 512)).astype(np.float32)[np.arange(n) % 16] * np.float32(.01)
    return x


def generate(family, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    if family in ('clustered', 'outlier'):
        return clustered(rng, 4096, family == 'outlier')
    if family == 'logrange':
        magnitude = np.exp2(rng.uniform(-10, 0, (4096, 512)))
        sign = 2 * rng.integers(0, 2, magnitude.shape) - 1
        return (magnitude * sign).astype(np.float32)
    assert family == 'mixed'
    streams = [np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(3)]
    x = np.concatenate([clustered(streams[0], 2048), clustered(streams[1], 2048, True)])
    return x[streams[2].permutation(4096)].copy()


def sample_pairs(n, count=65536):
    ranks = np.random.Generator(np.random.PCG64(9001)).choice(n * (n - 1) // 2, count, replace=False)
    ends = np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64))
    i = np.searchsorted(ends, ranks, side='right')
    starts = np.concatenate([[0], ends[:-1]])
    j = i + 1 + ranks - starts[i]
    assert np.all(i < j) and np.all(j < n) and len(np.unique(i*n+j)) == count
    return i, j


def thresholds(x):
    i, j = sample_pairs(len(x)); distances = np.empty(len(i), np.float64)
    for start in range(0, len(i), 2048):
        delta = x[i[start:start+2048]].astype(np.float64) - x[j[start:start+2048]].astype(np.float64)
        distances[start:start+2048] = np.sum(delta * delta, axis=1, dtype=np.float64)
    ordered = np.sort(distances)
    return {k: float(ordered[math.ceil(len(i)*k/(len(x)-1))-1]) for k in (8, 128)}, digest(np.stack([i, j])), digest(distances)


def prepare():
    (ROOT / 'data').mkdir(exist_ok=True)
    path = ROOT / 'workloads.jsonl'
    if path.exists():
        raise FileExistsError(path.name)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with path.open('x') as out:
        for family in FAMILIES:
            for split, seeds in SPLITS.items():
                for seed in seeds:
                    parent = f'{family}_{seed}'; x = generate(family, seed)
                    assert x.dtype == np.float32 and x.flags.c_contiguous and np.isfinite(x).all() and np.abs(x).max() <= 1
                    np.save(ROOT / 'data' / (parent + '.npy'), x, allow_pickle=False)
                    for n in (1024, 4096):
                        t, pair_hash, distance_hash = thresholds(x[:n])
                        for k, threshold in t.items():
                            rec = {'config_id': f'{parent}_n{n}_k{k}', 'parent_id': parent, 'family': family,
                                   'seed': seed, 'split': split, 'n': n, 'd': 512, 'target_k': k,
                                   'input_sha256': digest(x[:n]), 'parent_sha256': digest(x),
                                   'threshold_d2': threshold, 'threshold_hex': threshold.hex(),
                                   'sample_pair_sha256': pair_hash, 'sample_distance_sha256': distance_hash,
                                   'generator_sha256': source_hash, 'numpy_version': np.__version__}
                            out.write(json.dumps(rec)+'\n'); out.flush()
    print(json.dumps({'workloads': 96, 'parents': 24, 'gpu_execution': False}))


def records(splits=('train', 'validation')):
    return [r for line in (ROOT / 'workloads.jsonl').read_text().splitlines()
            if (r := json.loads(line))['split'] in splits]


def load(rec):
    x = np.load(ROOT / 'data' / (rec['parent_id'] + '.npy'), allow_pickle=False)[:rec['n']].copy()
    assert digest(x) == rec['input_sha256']
    assert float(rec['threshold_d2']).hex() == rec['threshold_hex']
    return x


if __name__ == '__main__':
    prepare()
