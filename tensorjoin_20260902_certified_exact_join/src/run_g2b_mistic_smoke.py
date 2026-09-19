#!/usr/bin/env python3
"""Run one non-performance MiSTIC FP64 full-Cifar G2B smoke."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_mistic_cifar60000_g2b_smoke"
N = 60_000
D = 512
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2b_cifar60000"
BINARY = PROJECT / "adapters/mistic_g2a/build_g2a/main_d512"
RESULT = PROJECT / "results/g2b_mistic_smoke.json"
PAIR_FILE = PROJECT / "artifacts/g2b/mistic_pairs_u64_le.bin"
GDS_PAIR_FILE = PROJECT / "artifacts/g2b/gds_pairs_u64_le.bin"
TENSORJOIN_PAIR_FILE = PROJECT / "artifacts/g2b/tensorjoin_pairs_u64_le.bin"


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


def compare_sorted(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    missing = np.setdiff1d(reference, candidate, assume_unique=True)
    extra = np.setdiff1d(candidate, reference, assume_unique=True)
    return {
        "reference_count": int(reference.size),
        "reference_raw_u64_sha256": sha256_u64(reference),
        "missing_pairs": int(missing.size),
        "extra_pairs": int(extra.size),
        "first_missing_pairs": missing[:16].astype(np.uint64).tolist(),
        "first_extra_pairs": extra[:16].astype(np.uint64).tolist(),
        "exact_match": missing.size == 0 and extra.size == 0,
    }


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            f"G2B MiSTIC smoke requires CUDA_VISIBLE_DEVICES=0, got {visible!r}"
        )
    temporary_pair = PAIR_FILE.with_name(f".{PAIR_FILE.name}.tmp.{os.getpid()}")
    for path in (RESULT, PAIR_FILE, temporary_pair):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    required = (
        DATA_DIR / "vectors_f64_exact_widening.raw",
        DATA_DIR / "metadata.json",
        BINARY,
        GDS_PAIR_FILE,
        TENSORJOIN_PAIR_FILE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    epsilon = float(metadata["epsilon"])
    expected_count = int(metadata["diagnostic_expected_exact_count"]["value"])
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
        "input_sha256": sha256_file(DATA_DIR / "vectors_f64_exact_widening.raw"),
        "metadata_sha256": sha256_file(DATA_DIR / "metadata.json"),
        "binary_sha256": sha256_file(BINARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "diagnostic_historical_count": expected_count,
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)
    env = os.environ.copy()
    env["MISTIC_G2A_OUTPUT"] = str(temporary_pair)
    command = [
        str(BINARY),
        str(DATA_DIR / "vectors_f64_exact_widening.raw"),
        str(D),
        "0",
        format(epsilon, ".17g"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=False)
    diagnostic_wall_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"MiSTIC exited with {completed.returncode}")
    if not temporary_pair.is_file() or temporary_pair.stat().st_size % 8:
        raise RuntimeError("MiSTIC did not produce a valid uint64 pair file")
    os.replace(temporary_pair, PAIR_FILE)

    canonical = np.fromfile(PAIR_FILE, dtype="<u8")
    if canonical.size and np.any(canonical[1:] < canonical[:-1]):
        raise AssertionError("MiSTIC adapter output is not sorted")
    invalid_pairs = int(np.count_nonzero(canonical >= np.uint64(N) * np.uint64(N)))
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    rows = canonical // np.uint64(N)
    columns = canonical - rows * np.uint64(N)
    self_pairs = int(np.count_nonzero(rows == columns))
    reverse = columns * np.uint64(N) + rows
    symmetry_missing = int(np.setdiff1d(reverse, canonical, assume_unique=True).size)
    gds = np.fromfile(GDS_PAIR_FILE, dtype="<u8")
    tensorjoin = np.fromfile(TENSORJOIN_PAIR_FILE, dtype="<u8")
    gds_comparison = compare_sorted(gds, canonical)
    tensorjoin_comparison = compare_sorted(tensorjoin, canonical)
    structural_pass = (
        invalid_pairs == 0
        and duplicate_pairs == 0
        and self_pairs == N
        and symmetry_missing == 0
    )
    three_way_exact = (
        structural_pass
        and bool(gds_comparison["exact_match"])
        and bool(tensorjoin_comparison["exact_match"])
    )
    result = {
        **run,
        "command": command,
        "measurement_status": "full_correctness_resource_smoke_no_performance_claim",
        "diagnostic_wall_s": diagnostic_wall_s,
        "pair_file": str(PAIR_FILE.relative_to(PROJECT)),
        "pair_file_bytes": PAIR_FILE.stat().st_size,
        "pair_file_sha256": sha256_file(PAIR_FILE),
        "canonical_pair_count": int(canonical.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "invalid_pairs": invalid_pairs,
        "duplicate_pairs": duplicate_pairs,
        "self_pairs": self_pairs,
        "symmetry_missing_pairs": symmetry_missing,
        "diagnostic_historical_count_match": canonical.size == expected_count,
        "gds_comparison": gds_comparison,
        "tensorjoin_comparison": tensorjoin_comparison,
        "structural_smoke_pass": structural_pass,
        "three_way_full_hash_exact_match": three_way_exact,
    }
    atomic_json(RESULT, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if three_way_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
