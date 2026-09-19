#!/usr/bin/env python3
"""Evaluate the frozen G3A host analytic certificate opportunity gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_analytic_certificate_g3a"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/g2a_cifar4096"
RESULT = PROJECT / "results/g3a_analytic_certificate.json"
N = 4_096
D = 512
BLOCK_ROWS = 4
REFERENCE_AMBIGUOUS = 102_277
MAX_AMBIGUOUS = 127_846
F32_UNIT_ROUNDOFF = 2.0**-24
EXPRESSION_RELATIVE_RADIUS = 2.0**-17
EXPRESSION_ABSOLUTE_RADIUS = 8.0 * float(np.finfo(np.float32).tiny)
MAX_REPORTED_VIOLATIONS = 16
EXPECTED_VECTOR_SHA256 = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"
EXPECTED_ORACLE_FILE_SHA256 = "ac93d96ac39ea5a2bc2d1e81baa179a533a6344aae97f058b3b33a8155e6c0c9"
EXPECTED_ORACLE_RAW_SHA256 = "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d"
EXPECTED_ORACLE_COUNT = 262_144


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


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def append_examples(
    destination: list[dict[str, object]],
    mask: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    if len(destination) >= MAX_REPORTED_VIOLATIONS or not np.any(mask):
        return
    local_rows, local_columns = np.nonzero(mask)
    remaining = MAX_REPORTED_VIOLATIONS - len(destination)
    for local_row, local_column in zip(
        local_rows[:remaining], local_columns[:remaining], strict=True
    ):
        record: dict[str, object] = {
            "row": int(rows[local_row]),
            "column": int(columns[local_column]),
        }
        for name, values in fields.items():
            record[name] = float(values[local_row, local_column])
        destination.append(record)


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    started = time.perf_counter()
    metadata_path = DATA / "metadata.json"
    vectors_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    for path in (metadata_path, vectors_path, oracle_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = np.ascontiguousarray(
        np.load(vectors_path, allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(
        np.load(oracle_path, allow_pickle=False), dtype=np.uint64
    )
    if vectors.shape != (N, D):
        raise ValueError(vectors.shape)
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon = float(metadata["radius"]["epsilon"])
    if not math.isclose(epsilon * epsilon, threshold_d2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Frozen epsilon and squared threshold disagree")
    oracle_rows = oracle // np.uint64(N)
    oracle_columns = oracle - oracle_rows * np.uint64(N)
    oracle_upper_mask = oracle_rows <= oracle_columns
    oracle_upper_rows = oracle_rows[oracle_upper_mask].astype(np.intp)
    oracle_upper_columns = oracle_columns[oracle_upper_mask].astype(np.intp)
    oracle_truth = np.zeros((N, N), dtype=np.bool_)
    oracle_truth[oracle_upper_rows, oracle_upper_columns] = True
    vector_sha256 = sha256_file(vectors_path)
    oracle_file_sha256 = sha256_file(oracle_path)
    oracle_raw_sha256 = sha256_u64(oracle)
    if vector_sha256 != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Frozen vector file hash mismatch")
    if oracle_file_sha256 != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Frozen oracle file hash mismatch")
    if oracle_raw_sha256 != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen oracle hash mismatch")

    codes, scales = quantize(vectors)
    max_integer_accumulator = D * 127 * 127
    int32_safe = max_integer_accumulator <= np.iinfo(np.int32).max
    float32_exact_integer_conversion = max_integer_accumulator <= 2**24
    code_i64 = codes.astype(np.int64)
    code_norm_i64 = np.sum(code_i64 * code_i64, axis=1, dtype=np.int64)
    if int(np.max(code_norm_i64)) > max_integer_accumulator:
        raise AssertionError("Code norm exceeds the declared accumulator bound")
    code_f64 = codes.astype(np.float64)
    print("COMPUTE_CODE_DOT_START", flush=True)
    code_dot_f64 = code_f64 @ code_f64.T
    if not np.all(code_dot_f64 == np.rint(code_dot_f64)):
        raise AssertionError("Integer-valued code dot lost exactness in float64 BLAS")
    if float(np.max(np.abs(code_dot_f64))) > max_integer_accumulator:
        raise AssertionError("Observed code dot exceeds the declared accumulator bound")
    code_dot_i64 = code_dot_f64.astype(np.int64)
    del code_dot_f64, code_f64
    print("COMPUTE_CODE_DOT_COMPLETE", flush=True)

    scale_f64 = scales.astype(np.float64)
    scale_ld = scales.astype(np.longdouble)
    code_norm_ld = code_norm_i64.astype(np.longdouble)
    reconstructed_norm_exact_ld = code_norm_ld * scale_ld * scale_ld
    reconstructed_norm_f32 = (
        code_norm_i64.astype(np.float64) * scale_f64 * scale_f64
    ).astype(np.float32)

    reconstruction_f64 = code_i64.astype(np.float64) * scale_f64[:, None]
    residual_f64 = vectors.astype(np.float64) - reconstruction_f64
    residual_sum_sq_f64 = np.sum(
        residual_f64 * residual_f64, axis=1, dtype=np.float64
    )
    residual_operation_count = 2 * D + 2
    residual_gamma = (
        residual_operation_count * np.finfo(np.float64).eps / 2.0
    ) / (1.0 - residual_operation_count * np.finfo(np.float64).eps / 2.0)
    residual_error_upper = np.nextafter(
        np.sqrt(residual_sum_sq_f64 / (1.0 - residual_gamma)), np.inf
    )
    del reconstruction_f64, residual_f64, residual_sum_sq_f64, code_i64

    counters = {
        "analytical_upper_comparisons": 0,
        "direct_accept_upper_pairs": 0,
        "direct_reject_upper_pairs": 0,
        "ambiguous_upper_pairs": 0,
        "unsafe_direct_accepts": 0,
        "unsafe_direct_rejects": 0,
        "expression_bound_violations": 0,
        "source_distance_interval_violations": 0,
    }
    expression_examples: list[dict[str, object]] = []
    interval_examples: list[dict[str, object]] = []
    unsafe_examples: list[dict[str, object]] = []
    maximum_expression_absolute_error = 0.0
    maximum_expression_bound_fraction = 0.0
    minimum_expression_bound_slack = math.inf
    maximum_interval_width = 0.0

    vectors_f64 = vectors.astype(np.float64)
    all_columns = np.arange(N, dtype=np.int64)
    for start in range(0, N, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, N)
        rows = np.arange(start, stop, dtype=np.int64)
        c_i64 = code_dot_i64[start:stop]
        c_f32 = c_i64.astype(np.float32)
        dot_f32 = (c_f32 * scales[start:stop, None]).astype(np.float32)
        dot_f32 = (dot_f32 * scales[None, :]).astype(np.float32)
        sum_norm_f32 = (
            reconstructed_norm_f32[start:stop, None]
            + reconstructed_norm_f32[None, :]
        ).astype(np.float32)
        twice_dot_f32 = (np.float32(2.0) * dot_f32).astype(np.float32)
        reconstructed_d2_f32 = (sum_norm_f32 - twice_dot_f32).astype(np.float32)

        c_ld = c_i64.astype(np.longdouble)
        exact_dot_ld = c_ld * scale_ld[start:stop, None] * scale_ld[None, :]
        exact_reconstructed_d2_ld = (
            reconstructed_norm_exact_ld[start:stop, None]
            + reconstructed_norm_exact_ld[None, :]
            - np.longdouble(2.0) * exact_dot_ld
        )
        expression_sum_abs_ld = (
            np.abs(reconstructed_norm_exact_ld[start:stop, None])
            + np.abs(reconstructed_norm_exact_ld[None, :])
            + np.longdouble(2.0) * np.abs(exact_dot_ld)
        )
        expression_radius = (
            EXPRESSION_RELATIVE_RADIUS
            * np.asarray(expression_sum_abs_ld, dtype=np.float64)
            + EXPRESSION_ABSOLUTE_RADIUS
        )
        expression_error = np.abs(
            reconstructed_d2_f32.astype(np.longdouble)
            - exact_reconstructed_d2_ld
        )
        expression_error_f64 = np.asarray(expression_error, dtype=np.float64)
        expression_violation = expression_error > expression_radius.astype(np.longdouble)

        # Direct float64 differences are the independent source-distance audit.
        difference = (
            vectors_f64[start:stop, None, :] - vectors_f64[None, :, :]
        )
        source_d2 = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )
        source_distance = np.sqrt(source_d2)
        del difference

        lower_reconstructed = np.sqrt(
            np.maximum(reconstructed_d2_f32.astype(np.float64) - expression_radius, 0.0)
        )
        upper_reconstructed = np.sqrt(
            np.maximum(reconstructed_d2_f32.astype(np.float64) + expression_radius, 0.0)
        )
        residual_radius = (
            residual_error_upper[start:stop, None] + residual_error_upper[None, :]
        )
        lower = np.maximum(lower_reconstructed - residual_radius, 0.0)
        upper = upper_reconstructed + residual_radius
        interval_violation = (source_distance < lower) | (source_distance > upper)

        valid_upper = rows[:, None] <= all_columns[None, :]
        accept = valid_upper & (upper <= epsilon)
        reject = valid_upper & (lower > epsilon)
        ambiguous = valid_upper & ~(accept | reject)
        truth = oracle_truth[start:stop]
        unsafe_accept = accept & ~truth
        unsafe_reject = reject & truth

        counters["analytical_upper_comparisons"] += int(np.count_nonzero(valid_upper))
        counters["direct_accept_upper_pairs"] += int(np.count_nonzero(accept))
        counters["direct_reject_upper_pairs"] += int(np.count_nonzero(reject))
        counters["ambiguous_upper_pairs"] += int(np.count_nonzero(ambiguous))
        counters["unsafe_direct_accepts"] += int(np.count_nonzero(unsafe_accept))
        counters["unsafe_direct_rejects"] += int(np.count_nonzero(unsafe_reject))
        counters["expression_bound_violations"] += int(
            np.count_nonzero(expression_violation & valid_upper)
        )
        counters["source_distance_interval_violations"] += int(
            np.count_nonzero(interval_violation & valid_upper)
        )
        valid_error = expression_error_f64[valid_upper]
        valid_radius = expression_radius[valid_upper]
        maximum_expression_absolute_error = max(
            maximum_expression_absolute_error, float(np.max(valid_error))
        )
        maximum_expression_bound_fraction = max(
            maximum_expression_bound_fraction,
            float(np.max(valid_error / valid_radius)),
        )
        minimum_expression_bound_slack = min(
            minimum_expression_bound_slack,
            float(np.min(valid_radius - valid_error)),
        )
        maximum_interval_width = max(
            maximum_interval_width, float(np.max((upper - lower)[valid_upper]))
        )

        append_examples(
            expression_examples,
            expression_violation & valid_upper,
            rows,
            all_columns,
            {
                "expression_error": expression_error_f64,
                "expression_radius": expression_radius,
                "reconstructed_d2_f32": reconstructed_d2_f32,
            },
        )
        append_examples(
            interval_examples,
            interval_violation & valid_upper,
            rows,
            all_columns,
            {
                "source_distance": source_distance,
                "lower": lower,
                "upper": upper,
            },
        )
        append_examples(
            unsafe_examples,
            unsafe_accept | unsafe_reject,
            rows,
            all_columns,
            {
                "source_distance": source_distance,
                "lower": lower,
                "upper": upper,
            },
        )
        if start % 512 == 0:
            print(
                "PROGRESS "
                + json.dumps(
                    {
                        "rows_complete": stop,
                        "ambiguous_upper_pairs": counters["ambiguous_upper_pairs"],
                        "expression_bound_violations": counters[
                            "expression_bound_violations"
                        ],
                        "interval_violations": counters[
                            "source_distance_interval_violations"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_upper_comparisons = N * (N + 1) // 2
    observed_oracle_upper_pairs = int(np.count_nonzero(oracle_truth))
    oracle_contract_pass = bool(
        oracle.size == EXPECTED_ORACLE_COUNT
        and vector_sha256 == EXPECTED_VECTOR_SHA256
        and oracle_file_sha256 == EXPECTED_ORACLE_FILE_SHA256
        and oracle_raw_sha256 == EXPECTED_ORACLE_RAW_SHA256
        and observed_oracle_upper_pairs == int(np.count_nonzero(oracle_upper_mask))
    )
    ambiguity_ratio = counters["ambiguous_upper_pairs"] / REFERENCE_AMBIGUOUS
    gate_pass = bool(
        oracle_contract_pass
        and int32_safe
        and float32_exact_integer_conversion
        and counters["analytical_upper_comparisons"] == expected_upper_comparisons
        and counters["expression_bound_violations"] == 0
        and counters["source_distance_interval_violations"] == 0
        and counters["unsafe_direct_accepts"] == 0
        and counters["unsafe_direct_rejects"] == 0
        and counters["ambiguous_upper_pairs"] <= MAX_AMBIGUOUS
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "host_analytic_opportunity_gate_not_gpu_proof_or_performance",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "shape": [N, D],
        "threshold_d2": threshold_d2,
        "epsilon_from_threshold": epsilon,
        "source_sha256": vector_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "oracle_file_sha256": oracle_file_sha256,
        "oracle_raw_u64_sha256": oracle_raw_sha256,
        "oracle_directed_pair_count": int(oracle.size),
        "oracle_upper_pair_count": observed_oracle_upper_pairs,
        "oracle_contract_pass": oracle_contract_pass,
        "integer_preconditions": {
            "code_range": [-127, 127],
            "dimension": D,
            "maximum_absolute_accumulator": max_integer_accumulator,
            "int32_safe": int32_safe,
            "float32_exact_integer_conversion": float32_exact_integer_conversion,
            "observed_maximum_absolute_dot": int(np.max(np.abs(code_dot_i64))),
        },
        "analytic_constants": {
            "float32_unit_roundoff": F32_UNIT_ROUNDOFF,
            "expression_relative_radius": EXPRESSION_RELATIVE_RADIUS,
            "expression_absolute_radius": EXPRESSION_ABSOLUTE_RADIUS,
            "residual_operation_count": residual_operation_count,
            "residual_gamma_float64": residual_gamma,
        },
        "residual_error_upper": {
            "minimum": float(np.min(residual_error_upper)),
            "median": float(np.median(residual_error_upper)),
            "maximum": float(np.max(residual_error_upper)),
        },
        "expression_audit": {
            "maximum_absolute_error": maximum_expression_absolute_error,
            "maximum_bound_fraction": maximum_expression_bound_fraction,
            "minimum_bound_slack": minimum_expression_bound_slack,
            "violations": counters["expression_bound_violations"],
            "first_violations": expression_examples,
        },
        "interval_audit": {
            "maximum_width": maximum_interval_width,
            "violations": counters["source_distance_interval_violations"],
            "first_violations": interval_examples,
        },
        "stage_counts": counters,
        "first_unsafe_decisions": unsafe_examples,
        "reference_ambiguous_upper_pairs": REFERENCE_AMBIGUOUS,
        "maximum_allowed_ambiguous_upper_pairs": MAX_AMBIGUOUS,
        "analytic_ambiguity_ratio_vs_g2a2": ambiguity_ratio,
        "g3a_gate_pass": gate_pass,
        "gpu_proof_candidate_admitted": gate_pass,
        "performance_claim_allowed": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3A.md"),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(RESULT, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
