#!/usr/bin/env python3
"""Run one GDS-Join FP64 process under the frozen G2B public denominator."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from g2b_public_common import (
    D,
    N,
    PROJECT,
    atomic_json,
    load_contract,
    load_pageable_source,
    max_rss_kib,
    parse_public_args,
    sha256_file,
    validate_canonical,
)


EXPERIMENT_ID = "tensorjoin_20260903_g2b_public_screen"
LIBRARY = PROJECT / "adapters/gds_g2a/libgpuselfjoin.so"


def initialize_cuda_runtime() -> None:
    candidates = (
        "/usr/local/cuda-13.1/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "libcudart.so",
    )
    runtime = None
    errors: list[str] = []
    for candidate in candidates:
        try:
            runtime = ctypes.CDLL(candidate)
            break
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    if runtime is None:
        raise RuntimeError("Unable to load libcudart: " + "; ".join(errors))
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFree.restype = ctypes.c_int
    runtime.cudaDeviceSynchronize.argtypes = []
    runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    for name, status in (
        ("cudaSetDevice", runtime.cudaSetDevice(0)),
        ("cudaFree(nullptr)", runtime.cudaFree(None)),
        ("cudaDeviceSynchronize", runtime.cudaDeviceSynchronize()),
    ):
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}")


def main() -> int:
    args = parse_public_args(__doc__ or "GDS public adapter")
    record_id = args.record_id
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not LIBRARY.is_file():
        raise FileNotFoundError(LIBRARY)
    result_path = PROJECT / f"results/g2b_public_{args.phase}_gds_{record_id}.json"
    if result_path.exists():
        raise FileExistsError(result_path)

    contract = load_contract()
    vectors_f32 = load_pageable_source()
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
    initialize_cuda_runtime()

    run = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "record_id": record_id,
        "method": "gds_fp64",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": visible,
        "shape": [N, D],
        "source_dtype": "float32",
        "method_dtype": "float64 exact widening",
        "epsilon": contract["epsilon"],
        "threshold_d2": contract["threshold_d2"],
        "source_npy_sha256": contract["source_npy_sha256"],
        "metadata_sha256": contract["metadata_sha256"],
        "smoke_summary_sha256": contract["smoke_summary_sha256"],
        "library_sha256": sha256_file(LIBRARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "timing_scope": "f32 pageable host -> sorted canonical u64 host",
        "excluded_before_timer": [
            "file_io",
            "process_startup",
            "dynamic_library_load",
            "cuda_context_initialization",
        ],
        "excluded_after_timer": ["hashing", "correctness_checks", "json_write"],
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    public_started = time.perf_counter()
    vectors_f64 = np.ascontiguousarray(vectors_f32, dtype=np.float64)
    flattened = vectors_f64.reshape(-1)
    counts = np.zeros(N, dtype=np.uint32)
    library.GDSJoinPy(flattened, N, float(contract["epsilon"]), D, 0, counts)
    total = int(counts.sum(dtype=np.uint64))
    neighbors = np.empty(total, dtype=np.uint32)
    library.copyResultIntoPythonArray(neighbors, total)
    queries = np.repeat(np.arange(N, dtype=np.uint64), counts.astype(np.int64))
    if queries.size != neighbors.size:
        raise AssertionError("GDS count vector does not reconstruct result length")
    canonical = queries * np.uint64(N) + neighbors.astype(np.uint64)
    canonical.sort()
    public_seconds = time.perf_counter() - public_started

    correctness = validate_canonical(canonical, contract)
    result = {
        **run,
        "measurement_status": (
            "formal_public_denominator_exact_keeper"
            if args.phase == "formal"
            else "diagnostic_public_denominator_cheap_screen"
        ),
        "public_seconds": public_seconds,
        "counts_min": int(counts.min()),
        "counts_median": float(np.median(counts)),
        "counts_max": int(counts.max()),
        "max_rss_kib_after_run": max_rss_kib(),
        "correctness": correctness,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if correctness["exact_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
