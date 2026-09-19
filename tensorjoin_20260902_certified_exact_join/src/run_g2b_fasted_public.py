#!/usr/bin/env python3
"""Run pinned FaSTED as a lower-quality G2B public-denominator context method."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

from g2b_public_common import (
    D,
    N,
    PROJECT,
    atomic_json,
    load_contract,
    max_rss_kib,
    sha256_file,
    sha256_u64,
)


EXPERIMENT_ID = "tensorjoin_20260903_g2b_fasted_public_context"
BINARY = PROJECT / "adapters/fasted_g2b_public/build/main"
SOURCE_RAW = PROJECT / "data/g2b_cifar60000/vectors_f32.raw"
EXACT_PAIR_FILE = PROJECT / "artifacts/g2b/tensorjoin_pairs_u64_le.bin"
PATTERNS = {
    "public_seconds": re.compile(r"^G2B_PUBLIC_SECONDS=([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE),
    "output_pairs": re.compile(r"^G2B_PUBLIC_OUTPUT_PAIRS=([0-9]+)$", re.MULTILINE),
    "pairs_found": re.compile(r"^FASTED_G2B_PAIRS_FOUND=([0-9]+)$", re.MULTILINE),
    "pairs_stored": re.compile(r"^FASTED_G2B_PAIRS_STORED=([0-9]+)$", re.MULTILINE),
    "materialized_pairs": re.compile(r"^FASTED_G2B_MATERIALIZED_PAIRS=([0-9]+)$", re.MULTILINE),
    "invalid_pairs": re.compile(r"^FASTED_G2B_INVALID_PAIRS=([0-9]+)$", re.MULTILINE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--phase", choices=("validation", "formal"), required=True)
    args = parser.parse_args()
    if not args.record_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe record id: {args.record_id!r}")
    return args


def one(pattern_name: str, text: str, cast):
    matches = PATTERNS[pattern_name].findall(text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern_name} line, got {matches}")
    return cast(matches[0])


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    result_path = PROJECT / f"results/g2b_public_{args.phase}_fasted_{args.record_id}.json"
    pair_path = PROJECT / f"artifacts/g2b_public_{args.phase}/fasted_{args.record_id}.u64.bin"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (result_path, pair_path):
        if path.exists():
            raise FileExistsError(path)
    for path in (BINARY, SOURCE_RAW, EXACT_PAIR_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = load_contract()
    if sha256_file(SOURCE_RAW) != contract["source_raw_sha256"]:
        raise RuntimeError("FaSTED raw input differs from the frozen source")
    exact = np.fromfile(EXACT_PAIR_FILE, dtype="<u8")
    if sha256_u64(exact) != contract["expected_hash"]:
        raise RuntimeError("Exact comparison file differs from the frozen contract")

    run = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "record_id": args.record_id,
        "method": "fasted_fp16_input_fp32_accumulation_context_only",
        "quality_contract": "approximate_relative_to_frozen_fp64_exact_output",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "shape": [N, D],
        "source_dtype": "float32",
        "method_input_dtype": "fp16",
        "method_accumulation_dtype": "fp32",
        "epsilon": contract["epsilon"],
        "threshold_d2": contract["threshold_d2"],
        "source_raw_sha256": contract["source_raw_sha256"],
        "exact_pair_file_sha256": sha256_file(EXACT_PAIR_FILE),
        "binary_sha256": sha256_file(BINARY),
        "build_receipt_sha256": sha256_file(
            PROJECT / "receipts/g2b_fasted_public_build.json"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "timing_scope": "f32 pageable host -> sorted canonical u64 host",
        "excluded_before_timer": [
            "file_io",
            "process_startup",
            "cuda_context_initialization",
        ],
        "excluded_after_timer": [
            "disk_serialization",
            "pair_file_reload",
            "quality_comparison",
            "hashing",
            "json_write",
        ],
        "redistribution_status": "private evaluation only; pinned source had no license file",
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)
    env = os.environ.copy()
    env["FASTED_G2B_OUTPUT"] = str(pair_path)
    command = [
        str(BINARY),
        str(SOURCE_RAW),
        str(D),
        format(float(contract["epsilon"]), ".17g"),
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FaSTED exited with {completed.returncode}")
    parsed = {
        "public_seconds": one("public_seconds", completed.stdout, float),
        "output_pairs": one("output_pairs", completed.stdout, int),
        "pairs_found": one("pairs_found", completed.stdout, int),
        "pairs_stored": one("pairs_stored", completed.stdout, int),
        "materialized_pairs": one("materialized_pairs", completed.stdout, int),
        "invalid_pairs_emitted": one("invalid_pairs", completed.stdout, int),
    }
    approximate = np.fromfile(pair_path, dtype="<u8")
    sorted_output = bool(
        approximate.size < 2 or np.all(approximate[1:] >= approximate[:-1])
    )
    duplicate_pairs = int(np.count_nonzero(approximate[1:] == approximate[:-1]))
    invalid_pairs = int(np.count_nonzero(approximate >= np.uint64(N) * np.uint64(N)))
    # Only enable NumPy's unique-input shortcut after we have proved that the
    # materialized FaSTED output is sorted and duplicate-free.  Validation must
    # remain correct even if an adapter regression emits repeated pair IDs.
    approximate_is_unique = bool(sorted_output and duplicate_pairs == 0)
    intersection = np.intersect1d(
        exact, approximate, assume_unique=approximate_is_unique
    )
    exact_only = np.setdiff1d(
        exact, approximate, assume_unique=approximate_is_unique
    )
    approximate_only = np.setdiff1d(
        approximate, exact, assume_unique=approximate_is_unique
    )
    precision = float(intersection.size / approximate.size) if approximate.size else 0.0
    recall = float(intersection.size / exact.size) if exact.size else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    structural_pass = bool(
        sorted_output
        and duplicate_pairs == 0
        and invalid_pairs == 0
        and parsed["invalid_pairs_emitted"] == 0
        and parsed["output_pairs"] == approximate.size
        and parsed["pairs_stored"] == approximate.size
        and parsed["materialized_pairs"] == approximate.size
    )
    result = {
        **run,
        "command": command,
        "measurement_status": (
            "formal_context_same_public_denominator_not_an_exact_keeper"
            if args.phase == "formal"
            else "adapter_validation_context_only_no_performance_claim"
        ),
        **parsed,
        "pair_file": str(pair_path.relative_to(PROJECT)),
        "pair_file_bytes": pair_path.stat().st_size,
        "canonical_raw_u64_sha256": sha256_u64(approximate),
        "pair_file_sha256": sha256_file(pair_path),
        "sorted_output": sorted_output,
        "duplicate_pairs": duplicate_pairs,
        "invalid_pairs": invalid_pairs,
        "exact_intersection_pairs": int(intersection.size),
        "exact_only_pairs": int(exact_only.size),
        "approximate_only_pairs": int(approximate_only.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "first_exact_only_pairs": exact_only[:16].astype(np.uint64).tolist(),
        "first_approximate_only_pairs": approximate_only[:16].astype(np.uint64).tolist(),
        "structural_context_pass": structural_pass,
        "max_rss_kib_after_run_wrapper_only": max_rss_kib(),
        "max_rss_kib_fasted_child": int(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        ),
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
