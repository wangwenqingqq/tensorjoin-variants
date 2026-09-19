#!/usr/bin/env python3
"""Shared immutable-contract helpers for G2B public-denominator adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
from pathlib import Path

import numpy as np


N = 60_000
D = 512
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2b_cifar60000"
SMOKE_SUMMARY = PROJECT / "results/g2b_smoke_summary.json"
RECORD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")


def parse_record_id(description: str) -> str:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--record-id", required=True)
    args = parser.parse_args()
    if not RECORD_ID_PATTERN.fullmatch(args.record_id):
        raise ValueError(f"Unsafe record id: {args.record_id!r}")
    return args.record_id


def parse_public_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--phase", choices=("screen", "formal"), default="screen")
    args = parser.parse_args()
    if not RECORD_ID_PATTERN.fullmatch(args.record_id):
        raise ValueError(f"Unsafe record id: {args.record_id!r}")
    return args


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
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {path} or {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_contract() -> dict[str, object]:
    required = (
        DATA_DIR / "vectors_f32.npy",
        DATA_DIR / "vectors_f32.raw",
        DATA_DIR / "metadata.json",
        SMOKE_SUMMARY,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_SUMMARY.read_text(encoding="utf-8"))
    if not smoke.get("exact_correctness_resource_gate_pass"):
        raise RuntimeError("G2B exact smoke gate is not accepted")
    return {
        "epsilon": float(metadata["epsilon"]),
        "threshold_d2": float(metadata["effective_epsilon_d2"]),
        "expected_count": int(smoke["canonical_pair_count"]),
        "expected_hash": str(smoke["canonical_raw_u64_sha256"]),
        "source_npy_sha256": sha256_file(DATA_DIR / "vectors_f32.npy"),
        "source_raw_sha256": sha256_file(DATA_DIR / "vectors_f32.raw"),
        "metadata_sha256": sha256_file(DATA_DIR / "metadata.json"),
        "smoke_summary_sha256": sha256_file(SMOKE_SUMMARY),
    }


def load_pageable_source() -> np.ndarray:
    vectors = np.load(DATA_DIR / "vectors_f32.npy", allow_pickle=False)
    if vectors.shape != (N, D) or vectors.dtype != np.float32:
        raise ValueError((vectors.shape, vectors.dtype))
    if not vectors.flags.c_contiguous:
        raise ValueError("Frozen source must be C-contiguous")
    return vectors


def validate_canonical(
    canonical: np.ndarray, contract: dict[str, object]
) -> dict[str, object]:
    canonical = np.asarray(canonical, dtype=np.uint64)
    sorted_output = bool(
        canonical.size < 2 or np.all(canonical[1:] >= canonical[:-1])
    )
    invalid_pairs = int(np.count_nonzero(canonical >= np.uint64(N) * np.uint64(N)))
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    rows = canonical // np.uint64(N)
    columns = canonical - rows * np.uint64(N)
    self_pairs = int(np.count_nonzero(rows == columns))
    reverse = columns * np.uint64(N) + rows
    symmetry_missing = int(
        np.setdiff1d(reverse, canonical, assume_unique=sorted_output and duplicate_pairs == 0).size
    )
    output_hash = sha256_u64(canonical)
    exact_pass = bool(
        sorted_output
        and invalid_pairs == 0
        and duplicate_pairs == 0
        and self_pairs == N
        and symmetry_missing == 0
        and canonical.size == int(contract["expected_count"])
        and output_hash == contract["expected_hash"]
    )
    return {
        "canonical_pair_count": int(canonical.size),
        "canonical_raw_u64_sha256": output_hash,
        "expected_pair_count": int(contract["expected_count"]),
        "expected_raw_u64_sha256": contract["expected_hash"],
        "sorted_output": sorted_output,
        "invalid_pairs": invalid_pairs,
        "duplicate_pairs": duplicate_pairs,
        "self_pairs": self_pairs,
        "symmetry_missing_pairs": symmetry_missing,
        "exact_contract_pass": exact_pass,
    }


def max_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
