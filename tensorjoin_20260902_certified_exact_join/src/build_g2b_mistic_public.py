#!/usr/bin/env python3
"""Build the isolated MiSTIC G2B public-denominator adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


ADAPTER = PROJECT / "adapters/mistic_g2b_public"
BUILD = ADAPTER / "build"
NVCC = Path("/usr/local/cuda-13.1/bin/nvcc")
SOURCES = ("main", "launcher", "kernel", "nodes", "tree", "utils")
FLAGS = (
    "-std=c++17",
    "-O3",
    "-Xcompiler=-fopenmp",
    "-lineinfo",
    "-arch=sm_120",
    "-I.",
    "-DDIM=512",
    "-DBS=256",
    "-DKB=1024",
)
RECEIPT = PROJECT / "receipts/g2b_mistic_public_build.json"


def main() -> int:
    if not NVCC.is_file():
        raise FileNotFoundError(NVCC)
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    BUILD.mkdir(parents=True, exist_ok=True)
    binary = BUILD / "main_d512"
    targets = [BUILD / f"{name}.o" for name in SOURCES] + [binary]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite build targets: {existing}")

    commands: list[list[str]] = []
    for name in SOURCES:
        command = [
            str(NVCC),
            *FLAGS,
            "-c",
            f"src/{name}.cu",
            "-o",
            f"build/{name}.o",
        ]
        print("COMMAND " + json.dumps(command), flush=True)
        subprocess.run(command, cwd=ADAPTER, check=True)
        commands.append(command)
    link = [
        str(NVCC),
        *FLAGS,
        *(f"build/{name}.o" for name in SOURCES),
        "-o",
        "build/main_d512",
    ]
    print("COMMAND " + json.dumps(link), flush=True)
    subprocess.run(link, cwd=ADAPTER, check=True)
    commands.append(link)

    nvcc_version = subprocess.run(
        [str(NVCC), "--version"], check=True, text=True, capture_output=True
    ).stdout
    receipt = {
        "name": "MiSTIC G2B public-denominator adapter",
        "adapter_path": str(ADAPTER),
        "parent_adapter": "adapters/mistic_g2a",
        "configuration": {
            "architecture": "sm_120",
            "language_standard": "c++17",
            "DIM": 512,
            "BS": 256,
            "KB": 1024,
            "CUDA_DEVICE": 0,
            "KT": 5,
        },
        "changes": [
            "Read the frozen float32 source before the public timer",
            "Include exact float32-to-float64 widening in the public timer",
            "Stop after canonical host sort and before disk serialization",
        ],
        "kernel_or_distance_predicate_change": False,
        "commands": commands,
        "nvcc": nvcc_version,
        "source_hashes": {
            f"src/{name}.cu": sha256_file(ADAPTER / f"src/{name}.cu")
            for name in SOURCES
        },
        "binary_sha256": sha256_file(binary),
        "binary_bytes": binary.stat().st_size,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "parent_build_receipt_sha256": sha256_file(
            PROJECT / "receipts/g2a_mistic_adapter_build.json"
        ),
    }
    atomic_json(RECEIPT, receipt)
    print("BUILD_COMPLETE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

