#!/usr/bin/env python3
"""Validate the MiSTIC adapter against the frozen G2A FP64 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_mistic_cifar4096_g2a"
N = 4_096
D = 512
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2a_cifar4096"
BINARY = PROJECT / "adapters/mistic_g2a/build_g2a/main_d512"


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
    return hashlib.sha256(np.asarray(values, dtype="<u8").tobytes(order="C")).hexdigest()


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
        raise RuntimeError(f"G2A MiSTIC run requires CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    result_path = PROJECT / f"results/g2a_mistic_process_{args.run_id}.json"
    pair_path = PROJECT / f"results/g2a_mistic_process_{args.run_id}_pairs_u64_le.bin"
    temporary_pair_path = pair_path.with_name(f".{pair_path.name}.tmp.{os.getpid()}")
    for path in (result_path, pair_path, temporary_pair_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    for path in (DATA_DIR / "vectors_f64_le.bin", DATA_DIR / "oracle_pairs_u64.npy", DATA_DIR / "metadata.json", BINARY):
        if not path.is_file():
            raise FileNotFoundError(path)

    with (DATA_DIR / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    epsilon = float(metadata["radius"]["epsilon"])
    oracle = np.asarray(
        np.load(DATA_DIR / "oracle_pairs_u64.npy", allow_pickle=False), dtype=np.uint64
    )
    run = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "epsilon": epsilon,
        "input_sha256": sha256_file(DATA_DIR / "vectors_f64_le.bin"),
        "oracle_file_sha256": sha256_file(DATA_DIR / "oracle_pairs_u64.npy"),
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "binary_sha256": sha256_file(BINARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)
    env = os.environ.copy()
    env["MISTIC_G2A_OUTPUT"] = str(temporary_pair_path)
    command = [
        str(BINARY),
        str(DATA_DIR / "vectors_f64_le.bin"),
        str(D),
        "0",
        format(epsilon, ".17g"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=False)
    diagnostic_wall_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"MiSTIC exited with {completed.returncode}")
    if not temporary_pair_path.is_file() or temporary_pair_path.stat().st_size % 8:
        raise RuntimeError("MiSTIC did not produce a valid uint64 pair file")
    os.replace(temporary_pair_path, pair_path)

    canonical = np.fromfile(pair_path, dtype="<u8")
    invalid_pairs = int(np.count_nonzero(canonical >= N * N))
    if canonical.size and np.any(canonical[1:] < canonical[:-1]):
        raise AssertionError("MiSTIC adapter output is not sorted")
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    unique = np.unique(canonical)
    missing = np.setdiff1d(oracle, unique, assume_unique=True)
    extra = np.setdiff1d(unique, oracle, assume_unique=True)
    exact = (
        invalid_pairs == 0
        and duplicate_pairs == 0
        and missing.size == 0
        and extra.size == 0
        and canonical.size == oracle.size
    )
    result = {
        **run,
        "command": command,
        "measurement_status": "correctness_admission_only_no_performance_claim",
        "diagnostic_wall_s": diagnostic_wall_s,
        "pair_file": str(pair_path),
        "pair_file_sha256": sha256_file(pair_path),
        "canonical_pair_count": int(canonical.size),
        "canonical_unique_pair_count": int(unique.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "invalid_pairs": invalid_pairs,
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
