#!/usr/bin/env python3
"""Validate GDS-Join against the frozen G2A canonical FP64 oracle."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_gds_cifar4096_g2a"
N = 4_096
D = 512
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2a_cifar4096"
LIBRARY = PROJECT / "adapters/gds_g2a/libgpuselfjoin.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True, choices=range(2))
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<u8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"G2A GDS run requires CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    result_path = PROJECT / f"results/g2a_gds_process_{args.run_id}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    for path in (DATA_DIR / "vectors_f32.npy", DATA_DIR / "oracle_pairs_u64.npy", DATA_DIR / "metadata.json", LIBRARY):
        if not path.is_file():
            raise FileNotFoundError(path)

    with (DATA_DIR / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    epsilon = float(metadata["radius"]["epsilon"])
    vectors_f32 = np.load(DATA_DIR / "vectors_f32.npy", allow_pickle=False)
    vectors = np.ascontiguousarray(vectors_f32, dtype=np.float64)
    oracle = np.asarray(
        np.load(DATA_DIR / "oracle_pairs_u64.npy", allow_pickle=False), dtype=np.uint64
    )
    if vectors.shape != (N, D) or oracle.ndim != 1:
        raise ValueError((vectors.shape, oracle.shape))

    run = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "epsilon": epsilon,
        "input_sha256": sha256_file(DATA_DIR / "vectors_f32.npy"),
        "oracle_file_sha256": sha256_file(DATA_DIR / "oracle_pairs_u64.npy"),
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "library_sha256": sha256_file(LIBRARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    library = ctypes.CDLL(str(LIBRARY))
    f64_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    u32_array = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
    library.GDSJoinPy.argtypes = [
        f64_array,
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_int,
        u32_array,
    ]
    library.GDSJoinPy.restype = None
    library.copyResultIntoPythonArray.argtypes = [u32_array, ctypes.c_uint]
    library.copyResultIntoPythonArray.restype = None

    counts = np.zeros(N, dtype=np.uint32)
    flattened = np.ascontiguousarray(vectors.reshape(-1), dtype=np.float64)
    call_started = time.perf_counter()
    library.GDSJoinPy(flattened, N, epsilon, D, 0, counts)
    total = int(counts.sum(dtype=np.uint64))
    neighbors = np.empty(total, dtype=np.uint32)
    library.copyResultIntoPythonArray(neighbors, total)
    diagnostic_wall_s = time.perf_counter() - call_started

    invalid_neighbor_ids = int(np.count_nonzero(neighbors >= N))
    queries = np.repeat(np.arange(N, dtype=np.uint64), counts.astype(np.int64))
    if queries.size != neighbors.size:
        raise AssertionError("GDS count vector does not reconstruct result length")
    canonical = queries * np.uint64(N) + neighbors.astype(np.uint64)
    canonical.sort()
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    unique = np.unique(canonical)
    missing = np.setdiff1d(oracle, unique, assume_unique=True)
    extra = np.setdiff1d(unique, oracle, assume_unique=True)
    exact = (
        invalid_neighbor_ids == 0
        and duplicate_pairs == 0
        and missing.size == 0
        and extra.size == 0
        and canonical.size == oracle.size
    )

    result = {
        **run,
        "measurement_status": "correctness_admission_only_no_performance_claim",
        "diagnostic_wall_s": diagnostic_wall_s,
        "counts_sum": total,
        "counts_min": int(counts.min()),
        "counts_median": float(np.median(counts)),
        "counts_max": int(counts.max()),
        "canonical_pair_count": int(canonical.size),
        "canonical_unique_pair_count": int(unique.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "invalid_neighbor_ids": invalid_neighbor_ids,
        "duplicate_pairs": duplicate_pairs,
        "missing_pairs": int(missing.size),
        "extra_pairs": int(extra.size),
        "first_missing_pairs": missing[:16].astype(np.uint64).tolist(),
        "first_extra_pairs": extra[:16].astype(np.uint64).tolist(),
        "exact_match": exact,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
