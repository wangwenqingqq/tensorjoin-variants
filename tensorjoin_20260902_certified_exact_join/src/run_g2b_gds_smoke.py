#!/usr/bin/env python3
"""Run one non-performance GDS-Join FP64 full-Cifar G2B smoke."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_gds_cifar60000_g2b_smoke"
N = 60_000
D = 512
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2b_cifar60000"
LIBRARY = PROJECT / "adapters/gds_g2a/libgpuselfjoin.so"
RESULT = PROJECT / "results/g2b_gds_smoke.json"
PAIR_FILE = PROJECT / "artifacts/g2b/gds_pairs_u64_le.bin"


def sha256_file(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u8").tobytes(order="C")
    ).hexdigest()


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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"G2B GDS smoke requires CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    temporary_pair = PAIR_FILE.with_name(f".{PAIR_FILE.name}.tmp.{os.getpid()}")
    for path in (RESULT, PAIR_FILE, temporary_pair):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    required = (DATA_DIR / "vectors_f32.npy", DATA_DIR / "metadata.json", LIBRARY)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    epsilon = float(metadata["epsilon"])
    expected_count = int(metadata["diagnostic_expected_exact_count"]["value"])
    vectors_f32 = np.load(DATA_DIR / "vectors_f32.npy", allow_pickle=False)
    if vectors_f32.shape != (N, D) or vectors_f32.dtype != np.float32:
        raise ValueError((vectors_f32.shape, vectors_f32.dtype))
    vectors = np.ascontiguousarray(vectors_f32, dtype=np.float64)
    flattened = np.ascontiguousarray(vectors.reshape(-1), dtype=np.float64)

    run = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "shape": [N, D],
        "source_dtype": "float32",
        "exact_method_dtype": "float64 exact widening",
        "epsilon": epsilon,
        "effective_epsilon_d2": float(metadata["effective_epsilon_d2"]),
        "input_sha256": sha256_file(DATA_DIR / "vectors_f32.npy"),
        "metadata_sha256": sha256_file(DATA_DIR / "metadata.json"),
        "library_sha256": sha256_file(LIBRARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "diagnostic_expected_count": expected_count,
        "diagnostic_count_source_sha256": metadata["diagnostic_expected_exact_count"][
            "source_file_sha256"
        ],
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
    call_started = time.perf_counter()
    library.GDSJoinPy(flattened, N, epsilon, D, 0, counts)
    total = int(counts.sum(dtype=np.uint64))
    neighbors = np.empty(total, dtype=np.uint32)
    library.copyResultIntoPythonArray(neighbors, total)
    diagnostic_call_wall_s = time.perf_counter() - call_started

    invalid_neighbor_ids = int(np.count_nonzero(neighbors >= N))
    queries = np.repeat(np.arange(N, dtype=np.uint64), counts.astype(np.int64))
    if queries.size != neighbors.size:
        raise AssertionError("GDS count vector does not reconstruct result length")
    canonical = queries * np.uint64(N) + neighbors.astype(np.uint64)
    canonical.sort()
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    self_pairs = int(np.count_nonzero(canonical // np.uint64(N) == canonical % np.uint64(N)))
    reverse = (canonical % np.uint64(N)) * np.uint64(N) + canonical // np.uint64(N)
    symmetry_missing = int(np.setdiff1d(reverse, canonical, assume_unique=True).size)
    count_match = total == expected_count
    structural_pass = (
        invalid_neighbor_ids == 0
        and duplicate_pairs == 0
        and self_pairs == N
        and symmetry_missing == 0
        and count_match
    )

    PAIR_FILE.parent.mkdir(parents=True, exist_ok=True)
    canonical.astype("<u8", copy=False).tofile(temporary_pair)
    os.replace(temporary_pair, PAIR_FILE)
    result = {
        **run,
        "measurement_status": "full_correctness_resource_smoke_no_performance_claim",
        "oracle_status": "count_and_structure_only_until_independent_full_hash_agreement",
        "diagnostic_call_wall_s": diagnostic_call_wall_s,
        "counts_sum": total,
        "counts_min": int(counts.min()),
        "counts_median": float(np.median(counts)),
        "counts_max": int(counts.max()),
        "canonical_pair_count": int(canonical.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "pair_file": str(PAIR_FILE.relative_to(PROJECT)),
        "pair_file_bytes": PAIR_FILE.stat().st_size,
        "pair_file_sha256": sha256_file(PAIR_FILE),
        "invalid_neighbor_ids": invalid_neighbor_ids,
        "duplicate_pairs": duplicate_pairs,
        "self_pairs": self_pairs,
        "symmetry_missing_pairs": symmetry_missing,
        "diagnostic_count_match": count_match,
        "structural_smoke_pass": structural_pass,
    }
    atomic_json(RESULT, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
