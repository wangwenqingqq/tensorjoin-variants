#!/usr/bin/env python3
"""Run one MiSTIC FP64 process under the frozen G2B public denominator."""

from __future__ import annotations

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
    parse_public_args,
    sha256_file,
    validate_canonical,
)


EXPERIMENT_ID = "tensorjoin_20260903_g2b_public_screen"
BINARY = PROJECT / "adapters/mistic_g2b_public/build/main_d512"
SOURCE_RAW = PROJECT / "data/g2b_cifar60000/vectors_f32.raw"
PUBLIC_TIME = re.compile(r"^G2B_PUBLIC_SECONDS=([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)
PUBLIC_PAIRS = re.compile(r"^G2B_PUBLIC_OUTPUT_PAIRS=([0-9]+)$", re.MULTILINE)


def main() -> int:
    args = parse_public_args(__doc__ or "MiSTIC public adapter")
    record_id = args.record_id
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    result_path = PROJECT / f"results/g2b_public_{args.phase}_mistic_{record_id}.json"
    pair_path = PROJECT / f"artifacts/g2b_public_{args.phase}/mistic_{record_id}.u64.bin"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (result_path, pair_path):
        if path.exists():
            raise FileExistsError(path)
    if not BINARY.is_file():
        raise FileNotFoundError(BINARY)

    contract = load_contract()
    if sha256_file(SOURCE_RAW) != contract["source_raw_sha256"]:
        raise RuntimeError("MiSTIC float32 raw input differs from the frozen source")
    run = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "record_id": record_id,
        "method": "mistic_fp64",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "shape": [N, D],
        "source_dtype": "float32",
        "method_dtype": "float64 exact widening",
        "epsilon": contract["epsilon"],
        "threshold_d2": contract["threshold_d2"],
        "source_raw_sha256": contract["source_raw_sha256"],
        "metadata_sha256": contract["metadata_sha256"],
        "smoke_summary_sha256": contract["smoke_summary_sha256"],
        "binary_sha256": sha256_file(BINARY),
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
            "hashing",
            "correctness_checks",
            "json_write",
        ],
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)
    env = os.environ.copy()
    env["MISTIC_G2B_OUTPUT"] = str(pair_path)
    command = [
        str(BINARY),
        str(SOURCE_RAW),
        str(D),
        "0",
        format(float(contract["epsilon"]), ".17g"),
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"MiSTIC exited with {completed.returncode}")
    matches = PUBLIC_TIME.findall(completed.stdout)
    pair_matches = PUBLIC_PAIRS.findall(completed.stdout)
    if len(matches) != 1 or len(pair_matches) != 1:
        raise RuntimeError("MiSTIC did not emit exactly one public timing/count record")
    public_seconds = float(matches[0])
    emitted_count = int(pair_matches[0])
    if not pair_path.is_file() or pair_path.stat().st_size % 8:
        raise RuntimeError("MiSTIC did not produce a valid uint64 pair file")
    canonical = np.fromfile(pair_path, dtype="<u8")
    correctness = validate_canonical(canonical, contract)
    correctness["emitted_count"] = emitted_count
    correctness["emitted_count_matches_file"] = emitted_count == canonical.size
    correctness["exact_contract_pass"] = bool(
        correctness["exact_contract_pass"]
        and correctness["emitted_count_matches_file"]
    )
    result = {
        **run,
        "command": command,
        "measurement_status": (
            "formal_public_denominator_exact_keeper"
            if args.phase == "formal"
            else "diagnostic_public_denominator_cheap_screen"
        ),
        "public_seconds": public_seconds,
        "pair_file": str(pair_path.relative_to(PROJECT)),
        "pair_file_bytes": pair_path.stat().st_size,
        "pair_file_sha256": sha256_file(pair_path),
        "max_rss_kib_after_run_wrapper_only": max_rss_kib(),
        "max_rss_kib_mistic_child": int(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        ),
        "correctness": correctness,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if correctness["exact_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
