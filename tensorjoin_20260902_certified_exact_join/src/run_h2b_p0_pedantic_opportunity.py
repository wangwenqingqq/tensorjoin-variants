#!/usr/bin/env python3
"""Run the frozen CPU-only H2B-P0 pedantic-SGEMM opportunity gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import run_h1_multivector_gpu_screen as h1


EXPERIMENT_ID = "tensorjoin_20260903_h2b_p0_pedantic_opportunity"
PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results/h2b_p0_pedantic_opportunity.json"
H1_RUNNER_SHA256 = "11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e"
H1_KERNEL_SHA256 = "c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db"
H2A_CORRECTNESS_SHA256 = "d773c3cf08578af1b4845d5dc3bdddced4138f167659e24c7bf25eb3a3b5c9be"
H2A_SAFETY_SHA256 = "d10f7dab881175cb742325104d1f93479a0c76aacf5f107a5768505b42e39da4"
H2A_TIMING_SHA256 = "a97efa258375b3bc87949fc6a0efaf15fb863958dc8d099aa6f1344952cfdcb4"
MAX_AMBIGUOUS_FRACTION = 0.05
MAX_TARGET8_AMBIGUOUS_FRACTION = 0.02
FP64_RECONSTRUCTION_FACTOR = 32.0
ABSOLUTE_RECONSTRUCTION_GUARD = 1e-12


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_dependencies() -> None:
    expected = {
        PROJECT / "src/run_h1_multivector_gpu_screen.py": H1_RUNNER_SHA256,
        PROJECT / "src/h1_multivector_kernels.py": H1_KERNEL_SHA256,
        PROJECT / "results/h2a_fp32_strong_correctness.json": H2A_CORRECTNESS_SHA256,
        PROJECT / "results/h2a_fp32_strong_memcheck_summary.json": H2A_SAFETY_SHA256,
        PROJECT / "results/h2a_fp32_strong_timing_screen.json": H2A_TIMING_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"immutable dependency mismatch: {path}")


def lift_object_bounds(
    data: h1.PreparedDataset,
    lower_tokens: np.ndarray,
    upper_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(data.query.slices), len(data.base.slices))
    lower_objects = np.empty(shape, dtype=np.float64)
    upper_objects = np.empty(shape, dtype=np.float64)
    for qi, query_slice in enumerate(data.query.slices):
        for bi, base_slice in enumerate(data.base.slices):
            lower_objects[qi, bi] = h1.symmetric_chamfer(
                lower_tokens[query_slice, base_slice]
            )
            upper_objects[qi, bi] = h1.symmetric_chamfer(
                upper_tokens[query_slice, base_slice]
            )
    return lower_objects, upper_objects


def evaluate_dataset(data: h1.PreparedDataset) -> dict[str, object]:
    started = time.time()
    exact_tokens, exact_objects = h1.cpu_exact(data)
    query64 = data.query.vectors.astype(np.float64)
    base64 = data.base.vectors.astype(np.float64)
    query_norm2 = np.einsum("ij,ij->i", query64, query64)
    base_norm2 = np.einsum("ij,ij->i", base64, base64)

    unit_roundoff = 2.0**-24
    path_operations = 2 * data.dimension + 2
    gamma = (path_operations * unit_roundoff) / (
        1.0 - path_operations * unit_roundoff
    )
    product_sum_bound = np.sqrt(query_norm2[:, None] * base_norm2[None, :])
    dot_radius = (
        gamma * product_sum_bound
        + 4.0 * data.dimension * float(np.finfo(np.float32).tiny)
    )
    fp64_rounding = FP64_RECONSTRUCTION_FACTOR * np.finfo(np.float64).eps * (
        query_norm2[:, None]
        + base_norm2[None, :]
        + 2.0 * (product_sum_bound + dot_radius)
    )
    distance_radius = (
        2.0 * dot_radius + fp64_rounding + ABSOLUTE_RECONSTRUCTION_GUARD
    )

    # P0 has no measured SGEMM midpoint.  The interval is therefore expanded
    # around the exact value by two radii: one for the allowed midpoint shift
    # and one for the interval around that midpoint.
    lower_tokens = np.maximum(exact_tokens - 2.0 * distance_radius, 0.0)
    upper_tokens = exact_tokens + 2.0 * distance_radius
    lower_objects, upper_objects = lift_object_bounds(
        data, lower_tokens, upper_tokens
    )

    token_lower_violations = int(np.count_nonzero(lower_tokens > exact_tokens))
    token_upper_violations = int(np.count_nonzero(upper_tokens < exact_tokens))
    object_lower_violations = int(np.count_nonzero(lower_objects > exact_objects))
    object_upper_violations = int(np.count_nonzero(upper_objects < exact_objects))
    cells = []
    for target, threshold in sorted(data.thresholds.items()):
        direct_accept = upper_objects <= threshold
        direct_reject = lower_objects > threshold
        ambiguous = ~(direct_accept | direct_reject)
        exact_inside = exact_objects <= threshold
        ambiguous_count = int(np.count_nonzero(ambiguous))
        ambiguous_fraction = float(ambiguous_count / exact_objects.size)
        fraction_gate = (
            MAX_TARGET8_AMBIGUOUS_FRACTION
            if target == 8
            else MAX_AMBIGUOUS_FRACTION
        )
        cell_pass = bool(
            token_lower_violations == 0
            and token_upper_violations == 0
            and object_lower_violations == 0
            and object_upper_violations == 0
            and np.count_nonzero(direct_accept & ~exact_inside) == 0
            and np.count_nonzero(direct_reject & exact_inside) == 0
            and ambiguous_fraction <= fraction_gate
        )
        cells.append(
            {
                "target_results_per_query": int(target),
                "threshold": float(threshold),
                "direct_accepts": int(np.count_nonzero(direct_accept)),
                "direct_rejects": int(np.count_nonzero(direct_reject)),
                "ambiguous_objects": ambiguous_count,
                "ambiguous_fraction": ambiguous_fraction,
                "maximum_ambiguous_fraction": fraction_gate,
                "unsafe_direct_accepts": int(
                    np.count_nonzero(direct_accept & ~exact_inside)
                ),
                "unsafe_direct_rejects": int(
                    np.count_nonzero(direct_reject & exact_inside)
                ),
                "opportunity_pass": cell_pass,
            }
        )
    return {
        "dataset": data.name,
        "dimension": data.dimension,
        "query_objects": len(data.query.slices),
        "base_objects": len(data.base.slices),
        "query_tokens": len(data.query.vectors),
        "base_tokens": len(data.base.vectors),
        "cache": str(data.cache),
        "cache_sha256": data.cache_sha256,
        "gamma_path_operations": path_operations,
        "gamma_value": gamma,
        "maximum_distance_radius": float(distance_radius.max()),
        "maximum_worst_envelope_half_width": float(
            (2.0 * distance_radius).max()
        ),
        "token_lower_containment_violations": token_lower_violations,
        "token_upper_containment_violations": token_upper_violations,
        "object_lower_containment_violations": object_lower_violations,
        "object_upper_containment_violations": object_upper_violations,
        "cells": cells,
        "opportunity_pass": all(cell["opportunity_pass"] for cell in cells),
        "elapsed_s": time.time() - started,
    }


def main() -> int:
    require_dependencies()
    datasets = [evaluate_dataset(h1.prepare_dataset(name)) for name in h1.DATASETS]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "cpu_only_pessimistic_opportunity",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "real_vs_synthetic": "real public feature caches",
        "gpu_work_executed": False,
        "constants": {
            "unit_roundoff": 2.0**-24,
            "dot_path_operations": "2D+2",
            "underflow_floor": "4D*tiny32",
            "fp64_reconstruction_factor": FP64_RECONSTRUCTION_FACTOR,
            "absolute_reconstruction_guard": ABSOLUTE_RECONSTRUCTION_GUARD,
            "p0_worst_placement_multiplier": 2.0,
            "maximum_ambiguous_fraction": MAX_AMBIGUOUS_FRACTION,
            "maximum_target8_ambiguous_fraction": MAX_TARGET8_AMBIGUOUS_FRACTION,
        },
        "sources": {
            "runner_sha256": sha256_file(Path(__file__)),
            "h1_runner_sha256": H1_RUNNER_SHA256,
            "h1_kernel_sha256": H1_KERNEL_SHA256,
            "h2a_correctness_sha256": H2A_CORRECTNESS_SHA256,
            "h2a_safety_sha256": H2A_SAFETY_SHA256,
            "h2a_timing_sha256": H2A_TIMING_SHA256,
        },
        "datasets": datasets,
        "opportunity_pass": all(row["opportunity_pass"] for row in datasets),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["opportunity_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
