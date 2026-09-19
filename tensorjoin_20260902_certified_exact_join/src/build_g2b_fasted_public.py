#!/usr/bin/env python3
"""Build the private pinned FaSTED G2B public-denominator adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


ADAPTER = PROJECT / "adapters/fasted_g2b_public"
BUILD = ADAPTER / "build"
NVCC = Path("/usr/local/cuda-13.1/bin/nvcc")
BINARY = BUILD / "main"
RECEIPT = PROJECT / "receipts/g2b_fasted_public_build.json"
FLAGS = (
    "-std=c++17",
    "-O3",
    "-Xptxas=-v",
    "-Xcompiler",
    "-fopenmp",
    "-lcuda",
    "-lineinfo",
    "-D_MWAITXINTRIN_H_INCLUDED",
    "-D_FORCE_INLINES",
    "-arch=sm_120",
)


def main() -> int:
    for path in (NVCC, ADAPTER / "main.cu", ADAPTER / "findPairs.cuh"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if BINARY.exists() or RECEIPT.exists():
        raise FileExistsError(f"Refusing to overwrite {BINARY} or {RECEIPT}")
    BUILD.mkdir(parents=True, exist_ok=True)
    command = [str(NVCC), *FLAGS, "main.cu", "-o", "build/main"]
    print("COMMAND " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=ADAPTER, check=True)
    source_files = sorted(
        path
        for path in ADAPTER.rglob("*")
        if path.is_file() and "build" not in path.parts and path.name != "main_upstream_9af85ed8.cu"
    )
    receipt = {
        "name": "FaSTED G2B private public-denominator adapter",
        "redistribution_status": "private evaluation only; pinned source had no license file",
        "upstream_revision": "9af85ed8edc818d5aa7cc0473187631adab3e6ba",
        "upstream_tree": "cb1b71b2b56ff59865c18211957d7fcc9781e3da",
        "upstream_archive": "artifacts/upstream/fasted_9af85ed8_source.tar.gz",
        "upstream_archive_sha256": sha256_file(
            PROJECT / "artifacts/upstream/fasted_9af85ed8_source.tar.gz"
        ),
        "configuration": {
            "architecture": "sm_120",
            "language_standard": "c++17",
            "optimization": "O3",
        },
        "changes": [
            "Read frozen raw float32 input before public timer",
            "Include FP16 conversion/padding through sorted canonical host output",
            "Retain upstream D2H pair vector instead of discarding it",
            "Serialize canonical IDs only after timer",
        ],
        "kernel_or_distance_predicate_change": False,
        "command": command,
        "nvcc": subprocess.run(
            [str(NVCC), "--version"], check=True, text=True, capture_output=True
        ).stdout,
        "source_hashes": {
            str(path.relative_to(ADAPTER)): sha256_file(path) for path in source_files
        },
        "upstream_main_snapshot_sha256": sha256_file(
            ADAPTER / "main_upstream_9af85ed8.cu"
        ),
        "binary_sha256": sha256_file(BINARY),
        "binary_bytes": BINARY.stat().st_size,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(RECEIPT, receipt)
    print("BUILD_COMPLETE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

