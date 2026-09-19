#!/usr/bin/env python3
"""Evaluate the frozen G3C-A certified-FP32 opportunity gate.

The audited G3B-R1 GPU kernel is used only to enumerate its exact ambiguous
upper-triangle pairs.  The proposed second-stage interval is then evaluated on
the host.  This is neither generated-code proof nor performance evidence.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    D,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    N,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    analytic_certificate_compact_i64,
    outward_float32,
    quantize_with_analytic_residual,
)


EXPERIMENT_ID = "tensorjoin_20260903_g3c_a_host_fp32_interval"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/g2a_cifar4096"
RESULT = PROJECT / "results/g3c_a_host_fp32_interval.json"
AMBIGUOUS_ARTIFACT = PROJECT / "results/g3c_a_g3b_r1_ambiguous_u64_le.bin"
PROTOCOL = PROJECT / "PROTOCOL_G3C_A.md"
EXPECTED_VECTOR_SHA256 = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"
EXPECTED_ORACLE_FILE_SHA256 = "ac93d96ac39ea5a2bc2d1e81baa179a533a6344aae97f058b3b33a8155e6c0c9"
EXPECTED_ORACLE_RAW_SHA256 = "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d"
EXPECTED_G3B_R1_AMBIGUOUS_COUNT = 102_079
EXPECTED_G3B_R1_AMBIGUOUS_SHA256 = (
    "6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42"
)
MAX_FP64_REFINED = 10_207  # floor(10% * 102,079)
PAIR_BLOCK = 2_048
FP32_UNIT_ROUNDOFF = 2.0**-24
FP32_TINY_NORMAL = float(np.finfo(np.float32).tiny)
FIXED_GUARD_DIAGNOSTIC = 1.0e-3
MAX_REPORTED_VIOLATIONS = 16


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u8").tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_u64(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(np.asarray(values, dtype="<u8").tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def first_records(
    destination: list[dict[str, object]],
    mask: np.ndarray,
    pair_ids: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    if len(destination) >= MAX_REPORTED_VIOLATIONS or not np.any(mask):
        return
    remaining = MAX_REPORTED_VIOLATIONS - len(destination)
    for offset in np.flatnonzero(mask)[:remaining]:
        record: dict[str, object] = {"pair_id": int(pair_ids[offset])}
        for name, values in fields.items():
            value = values[offset]
            record[name] = bool(value) if np.issubdtype(values.dtype, np.bool_) else float(value)
        destination.append(record)


def enumerate_g3b_r1_ambiguity(
    vectors: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, object]]:
    codes, scales, reconstructed_norm2, residual_error = (
        quantize_with_analytic_residual(vectors)
    )
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)
    capacity = scheduled_tiles * BLOCK_M * BLOCK_N
    device = torch.device("cuda:0")
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2).to(device)
    errors_gpu = torch.from_numpy(residual_error).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    direct_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    ambiguous_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    counters = torch.zeros(5, dtype=torch.int32, device=device)
    analytic_certificate_compact_i64[(scheduled_tiles,)](
        codes_gpu,
        codes_t_gpu,
        scales_gpu,
        norms_gpu,
        errors_gpu,
        tile_rows_gpu,
        tile_columns_gpu,
        direct_ids,
        ambiguous_ids,
        counters,
        float(epsilon_lower),
        float(epsilon_upper),
        EXPRESSION_RELATIVE_RADIUS,
        EXPRESSION_ABSOLUTE_RADIUS,
        SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
        SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
        M=N,
        N_=N,
        K=D,
        CAPACITY=capacity,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    torch.cuda.synchronize()
    counts = counters.cpu().numpy().astype(np.int64)
    direct_count = int(counts[0])
    ambiguous_count = int(counts[1])
    overflow = int(counts[2])
    if max(direct_count, ambiguous_count) > capacity:
        overflow += 1
    ambiguous = np.sort(
        ambiguous_ids[: min(ambiguous_count, capacity)]
        .cpu()
        .numpy()
        .astype(np.uint64, copy=False)
    )
    gpu_metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
        "direct_accept_upper_pairs": direct_count,
        "ambiguous_upper_pairs": ambiguous_count,
        "overflow_events": overflow,
        "epsilon_lower_float32": float(epsilon_lower),
        "epsilon_upper_float32": float(epsilon_upper),
    }
    del (
        codes_gpu,
        codes_t_gpu,
        scales_gpu,
        norms_gpu,
        errors_gpu,
        tile_rows_gpu,
        tile_columns_gpu,
        direct_ids,
        ambiguous_ids,
        counters,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return ambiguous, gpu_metadata


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    if RESULT.exists() or AMBIGUOUS_ARTIFACT.exists():
        raise FileExistsError("G3C-A evidence path already exists")

    started = time.perf_counter()
    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    for path in (vector_path, oracle_path, metadata_path, PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    vector_sha256 = sha256_file(vector_path)
    oracle_file_sha256 = sha256_file(oracle_path)
    if vector_sha256 != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Frozen vector hash mismatch")
    if oracle_file_sha256 != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Frozen oracle file hash mismatch")

    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon = float(metadata["radius"]["epsilon"])
    if not math.isclose(epsilon * epsilon, threshold_d2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Frozen epsilon and squared threshold disagree")

    print("G3B_R1_AMBIGUITY_ENUMERATION_START", flush=True)
    ambiguous, gpu_metadata = enumerate_g3b_r1_ambiguity(vectors, epsilon)
    ambiguous_sha256 = sha256_u64(ambiguous)
    source_ambiguity_pass = bool(
        ambiguous.size == EXPECTED_G3B_R1_AMBIGUOUS_COUNT
        and ambiguous_sha256 == EXPECTED_G3B_R1_AMBIGUOUS_SHA256
        and gpu_metadata["overflow_events"] == 0
    )
    if not source_ambiguity_pass:
        raise RuntimeError(
            "Live G3B-R1 ambiguity does not reproduce the frozen count/hash"
        )
    atomic_u64(AMBIGUOUS_ARTIFACT, ambiguous)
    print("G3B_R1_AMBIGUITY_ENUMERATION_COMPLETE", flush=True)

    oracle_truth = np.zeros(N * N, dtype=np.bool_)
    oracle_truth[oracle.astype(np.intp)] = True
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    reduction_operations = D - 1
    reduction_gamma = (
        reduction_operations * FP32_UNIT_ROUNDOFF
        / (1.0 - reduction_operations * FP32_UNIT_ROUNDOFF)
    )
    counts = {
        "source_ambiguous_upper_pairs": int(ambiguous.size),
        "fp32_direct_accept_upper_pairs": 0,
        "fp32_direct_reject_upper_pairs": 0,
        "fp64_refine_upper_pairs": 0,
        "interval_violations": 0,
        "unsafe_fp32_accepts": 0,
        "unsafe_fp32_rejects": 0,
        "fixed_guard_fp64_refine_upper_pairs_diagnostic": 0,
    }
    violation_examples: list[dict[str, object]] = []
    unsafe_examples: list[dict[str, object]] = []
    maximum_actual_error = 0.0
    maximum_radius = 0.0
    maximum_bound_fraction = 0.0
    minimum_bound_slack = math.inf
    radius_values: list[np.ndarray] = []

    for start in range(0, ambiguous.size, PAIR_BLOCK):
        stop = min(start + PAIR_BLOCK, ambiguous.size)
        pair_ids = ambiguous[start:stop]
        rows = (pair_ids // np.uint64(N)).astype(np.intp)
        columns = (pair_ids % np.uint64(N)).astype(np.intp)
        x = vectors[rows]
        y = vectors[columns]
        x64 = x.astype(np.float64)
        y64 = y.astype(np.float64)
        exact_delta = x64 - y64
        exact_d2 = np.sum(exact_delta * exact_delta, axis=1, dtype=np.float64)

        # Model an explicit float32 subtract, square, and arbitrary-order sum.
        q = np.subtract(x, y, dtype=np.float32)
        q64 = q.astype(np.float64)
        q2_exact = q64 * q64
        terms = np.multiply(q, q, dtype=np.float32)
        terms64 = terms.astype(np.float64)
        d2_hat = np.sum(terms, axis=1, dtype=np.float32).astype(np.float64)

        # Conservative FTZ-aware source-to-q error, rounded-square error, and
        # serial D-term reduction error.  Serial gamma covers every binary
        # reduction order with at most D-1 roundings for nonnegative terms.
        difference_error = (
            FP32_UNIT_ROUNDOFF * (np.abs(x64) + np.abs(y64))
            + FP32_TINY_NORMAL
        )
        source_to_q2 = np.sum(
            2.0 * np.abs(q64) * difference_error + difference_error**2,
            axis=1,
            dtype=np.float64,
        )
        square_error = (
            FP32_UNIT_ROUNDOFF * np.sum(q2_exact, axis=1, dtype=np.float64)
            + D * FP32_TINY_NORMAL
        )
        reduction_error = (
            reduction_gamma * np.sum(np.abs(terms64), axis=1, dtype=np.float64)
            + reduction_operations * FP32_TINY_NORMAL
        )
        radius = np.nextafter(
            source_to_q2 + square_error + reduction_error, np.inf
        )
        lower = np.nextafter(np.maximum(d2_hat - radius, 0.0), -np.inf)
        upper = np.nextafter(d2_hat + radius, np.inf)
        interval_violation = (exact_d2 < lower) | (exact_d2 > upper)
        accept = upper <= float(threshold_lower)
        reject = lower > float(threshold_upper)
        refine = ~(accept | reject)
        truth = oracle_truth[pair_ids.astype(np.intp)]
        unsafe_accept = accept & ~truth
        unsafe_reject = reject & truth

        counts["fp32_direct_accept_upper_pairs"] += int(np.count_nonzero(accept))
        counts["fp32_direct_reject_upper_pairs"] += int(np.count_nonzero(reject))
        counts["fp64_refine_upper_pairs"] += int(np.count_nonzero(refine))
        counts["interval_violations"] += int(np.count_nonzero(interval_violation))
        counts["unsafe_fp32_accepts"] += int(np.count_nonzero(unsafe_accept))
        counts["unsafe_fp32_rejects"] += int(np.count_nonzero(unsafe_reject))
        fixed_refine = ~(
            (d2_hat <= threshold_d2 - FIXED_GUARD_DIAGNOSTIC)
            | (d2_hat > threshold_d2 + FIXED_GUARD_DIAGNOSTIC)
        )
        counts["fixed_guard_fp64_refine_upper_pairs_diagnostic"] += int(
            np.count_nonzero(fixed_refine)
        )

        actual_error = np.abs(d2_hat - exact_d2)
        maximum_actual_error = max(maximum_actual_error, float(np.max(actual_error)))
        maximum_radius = max(maximum_radius, float(np.max(radius)))
        maximum_bound_fraction = max(
            maximum_bound_fraction, float(np.max(actual_error / radius))
        )
        minimum_bound_slack = min(
            minimum_bound_slack, float(np.min(radius - actual_error))
        )
        radius_values.append(radius)
        first_records(
            violation_examples,
            interval_violation,
            pair_ids,
            {
                "exact_d2": exact_d2,
                "d2_hat": d2_hat,
                "radius": radius,
                "lower": lower,
                "upper": upper,
            },
        )
        first_records(
            unsafe_examples,
            unsafe_accept | unsafe_reject,
            pair_ids,
            {
                "truth": truth,
                "accept": accept,
                "reject": reject,
                "exact_d2": exact_d2,
                "lower": lower,
                "upper": upper,
            },
        )
        if start % (16 * PAIR_BLOCK) == 0:
            print(
                "PROGRESS "
                + json.dumps(
                    {
                        "pairs_complete": stop,
                        "fp64_refine_upper_pairs": counts[
                            "fp64_refine_upper_pairs"
                        ],
                        "interval_violations": counts["interval_violations"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    all_radii = np.concatenate(radius_values)
    stage_partition_pass = bool(
        counts["fp32_direct_accept_upper_pairs"]
        + counts["fp32_direct_reject_upper_pairs"]
        + counts["fp64_refine_upper_pairs"]
        == ambiguous.size
    )
    fp64_fraction = counts["fp64_refine_upper_pairs"] / int(ambiguous.size)
    gate_pass = bool(
        source_ambiguity_pass
        and stage_partition_pass
        and counts["interval_violations"] == 0
        and counts["unsafe_fp32_accepts"] == 0
        and counts["unsafe_fp32_rejects"] == 0
        and counts["fp64_refine_upper_pairs"] <= MAX_FP64_REFINED
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": (
            "gpu_enumerated_host_analytic_opportunity_gate_not_gpu_proof_or_performance"
        ),
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_visible_devices": visible,
        "shape": [N, D],
        "threshold_d2_exact_float64": threshold_d2,
        "threshold_d2_lower_float32": float(threshold_lower),
        "threshold_d2_upper_float32": float(threshold_upper),
        "source_sha256": vector_sha256,
        "oracle_file_sha256": oracle_file_sha256,
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "source_ambiguity": {
            **gpu_metadata,
            "raw_u64_sha256": ambiguous_sha256,
            "expected_count": EXPECTED_G3B_R1_AMBIGUOUS_COUNT,
            "expected_raw_u64_sha256": EXPECTED_G3B_R1_AMBIGUOUS_SHA256,
            "contract_pass": source_ambiguity_pass,
            "artifact": str(AMBIGUOUS_ARTIFACT.relative_to(PROJECT)),
            "artifact_sha256": sha256_file(AMBIGUOUS_ARTIFACT),
        },
        "analytic_contract": {
            "float32_unit_roundoff": FP32_UNIT_ROUNDOFF,
            "float32_tiny_normal": FP32_TINY_NORMAL,
            "difference_error": "u*(abs(x)+abs(y))+tiny",
            "square_error": "u*sum(q^2)+D*tiny",
            "reduction_operations": reduction_operations,
            "reduction_gamma": reduction_gamma,
            "reduction_error": "gamma_(D-1)*sum(abs(fl(q^2)))+(D-1)*tiny",
            "threshold_comparison": "outward-rounded float32 threshold endpoints",
            "pair_block": PAIR_BLOCK,
        },
        "stage_counts": counts,
        "stage_partition_pass": stage_partition_pass,
        "interval_audit": {
            "maximum_actual_absolute_error": maximum_actual_error,
            "maximum_radius": maximum_radius,
            "maximum_bound_fraction": maximum_bound_fraction,
            "minimum_bound_slack": minimum_bound_slack,
            "radius_minimum": float(np.min(all_radii)),
            "radius_median": float(np.median(all_radii)),
            "radius_p99": float(np.quantile(all_radii, 0.99)),
            "radius_maximum": float(np.max(all_radii)),
            "violations": counts["interval_violations"],
            "first_violations": violation_examples,
        },
        "first_unsafe_decisions": unsafe_examples,
        "fp64_refine_fraction": fp64_fraction,
        "maximum_fp64_refine_upper_pairs": MAX_FP64_REFINED,
        "maximum_fp64_refine_fraction": 0.10,
        "g3c_a_gate_pass": gate_pass,
        "g3c_b_gpu_candidate_admitted": gate_pass,
        "performance_claim_allowed": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROTOCOL),
        "g3b_r1_runner_sha256": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(RESULT, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
