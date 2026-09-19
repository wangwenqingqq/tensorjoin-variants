#!/usr/bin/env python3
"""Run one G3B-R1 validation with normal-or-zero residual-bound storage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from run_g2a_tensorjoin import refine_ambiguous_fp64_i64


EXPERIMENT_ID = "tensorjoin_20260903_gpu_analytic_certificate_g3b_r1"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/g2a_cifar4096"
N = 4_096
D = 512
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
MAX_AMBIGUOUS = 127_846
EXPRESSION_RELATIVE_RADIUS = 2.0**-16
EXPRESSION_ABSOLUTE_RADIUS = 16.0 * float(np.finfo(np.float32).tiny)
SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS = 2.0**-20
SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS = 16.0 * float(np.finfo(np.float32).tiny)
EXPECTED_VECTOR_SHA256 = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"
EXPECTED_ORACLE_FILE_SHA256 = "ac93d96ac39ea5a2bc2d1e81baa179a533a6344aae97f058b3b33a8155e6c0c9"
EXPECTED_ORACLE_RAW_SHA256 = "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d"


@triton.jit
def analytic_certificate_compact_i64(
    codes,
    codes_transposed,
    scales,
    reconstructed_norm2,
    residual_error_upper,
    tile_rows,
    tile_columns,
    result_ids,
    ambiguous_ids,
    counters,
    epsilon_lower,
    epsilon_upper,
    expression_relative_radius,
    expression_absolute_radius,
    final_relative_radius,
    final_absolute_radius,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(axis=0)
    program_m = tl.load(tile_rows + program)
    program_n = tl.load(tile_columns + program)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = codes + offsets_m[:, None] * K + offsets_k[None, :]
    b_ptrs = codes_transposed + offsets_k[:, None] * N_ + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N_

    scale_i = tl.load(scales + offsets_m, mask=offsets_m < M, other=0.0)
    scale_j = tl.load(scales + offsets_n, mask=offsets_n < N_, other=0.0)
    norm_i = tl.load(
        reconstructed_norm2 + offsets_m, mask=offsets_m < M, other=0.0
    )
    norm_j = tl.load(
        reconstructed_norm2 + offsets_n, mask=offsets_n < N_, other=0.0
    )
    error_i = tl.load(
        residual_error_upper + offsets_m, mask=offsets_m < M, other=0.0
    )
    error_j = tl.load(
        residual_error_upper + offsets_n, mask=offsets_n < N_, other=0.0
    )
    scaled_dot = accumulator.to(tl.float32) * scale_i[:, None] * scale_j[None, :]
    reconstructed_d2 = norm_i[:, None] + norm_j[None, :] - 2.0 * scaled_dot
    term_abs_sum = (
        tl.abs(norm_i[:, None])
        + tl.abs(norm_j[None, :])
        + 2.0 * tl.abs(scaled_dot)
    )
    expression_radius = (
        expression_relative_radius * term_abs_sum + expression_absolute_radius
    )
    reconstructed_lower = tl.sqrt(
        tl.maximum(reconstructed_d2 - expression_radius, 0.0)
    )
    reconstructed_upper = tl.sqrt(
        tl.maximum(reconstructed_d2 + expression_radius, 0.0)
    )
    residual_radius = error_i[:, None] + error_j[None, :]
    final_magnitude = tl.maximum(reconstructed_upper + residual_radius, 1.0)
    final_radius = (
        final_relative_radius * final_magnitude + final_absolute_radius
    )
    lower = tl.maximum(reconstructed_lower - residual_radius - final_radius, 0.0)
    upper = reconstructed_upper + residual_radius + final_radius

    valid = (
        (offsets_m[:, None] < M)
        & (offsets_n[None, :] < N_)
        & (offsets_m[:, None] <= offsets_n[None, :])
    )
    accept = valid & (upper <= epsilon_lower)
    reject = valid & (lower > epsilon_upper)
    ambiguous = valid & ~(accept | reject)
    pair_ids = (
        offsets_m[:, None].to(tl.int64) * N_ + offsets_n[None, :].to(tl.int64)
    )
    counter_lanes = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accept_positions = tl.atomic_add(
        counters + counter_lanes, 1, mask=accept, sem="relaxed"
    )
    accept_store = accept & (accept_positions < CAPACITY)
    tl.store(result_ids + accept_positions, pair_ids, mask=accept_store)
    tl.atomic_add(
        counters + 2 + counter_lanes,
        1,
        mask=accept & ~accept_store,
        sem="relaxed",
    )
    ambiguous_positions = tl.atomic_add(
        counters + 1 + counter_lanes, 1, mask=ambiguous, sem="relaxed"
    )
    ambiguous_store = ambiguous & (ambiguous_positions < CAPACITY)
    tl.store(ambiguous_ids + ambiguous_positions, pair_ids, mask=ambiguous_store)
    tl.atomic_add(
        counters + 2 + counter_lanes,
        1,
        mask=ambiguous & ~ambiguous_store,
        sem="relaxed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True, choices=range(2))
    return parser.parse_args()


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


def upward_float32(values: np.ndarray) -> np.ndarray:
    rounded = values.astype(np.float32)
    increment = rounded.astype(np.float64) < values.astype(np.float64)
    rounded[increment] = np.nextafter(rounded[increment], np.float32(np.inf))
    return rounded


def quantize_with_analytic_residual(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    code_i64 = codes.astype(np.int64)
    code_norm_i64 = np.sum(code_i64 * code_i64, axis=1, dtype=np.int64)
    reconstructed_norm2 = (
        code_norm_i64.astype(np.float64) * scales.astype(np.float64) ** 2
    ).astype(np.float32)
    reconstruction = code_i64.astype(np.float64) * scales.astype(np.float64)[:, None]
    residual = vectors.astype(np.float64) - reconstruction
    residual_sum_sq = np.sum(residual * residual, axis=1, dtype=np.float64)
    operations = 2 * D + 2
    unit_f64 = np.finfo(np.float64).eps / 2.0
    gamma = operations * unit_f64 / (1.0 - operations * unit_f64)
    residual_root = np.sqrt(residual_sum_sq / (1.0 - gamma))
    residual_upper_f64 = np.nextafter(residual_root, np.inf)
    residual_upper_f64[residual_root == 0.0] = 0.0
    residual_error = upward_float32(residual_upper_f64)
    positive_subnormal = (residual_error > 0.0) & (
        residual_error < np.finfo(np.float32).tiny
    )
    residual_error[positive_subnormal] = np.finfo(np.float32).tiny
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(reconstructed_norm2),
        np.ascontiguousarray(residual_error),
    )


def outward_float32(value: float) -> tuple[np.float32, np.float32]:
    rounded = np.float32(value)
    lower = rounded
    upper = rounded
    if float(rounded) > value:
        lower = np.nextafter(rounded, np.float32(-np.inf))
    if float(rounded) < value:
        upper = np.nextafter(rounded, np.float32(np.inf))
    return lower, upper


def sorted_ids(tensor: torch.Tensor, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    return np.sort(tensor[:count].cpu().numpy().astype(np.uint64, copy=False))


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g3b_r1_gpu_analytic_process_{args.run_id}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    for path in (vector_path, oracle_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(vector_path) != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Vector hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")

    started = time.perf_counter()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(
        np.load(oracle_path, allow_pickle=False), dtype=np.uint64
    )
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon = float(metadata["radius"]["epsilon"])
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    codes, scales, reconstructed_norm2, residual_error = (
        quantize_with_analytic_residual(vectors)
    )
    accumulator_limit = D * 127 * 127
    if accumulator_limit > min(2**24, np.iinfo(np.int32).max):
        raise RuntimeError("Exact integer accumulator/conversion precondition failed")

    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)
    capacity = scheduled_tiles * BLOCK_M * BLOCK_N

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2).to(device)
    errors_gpu = torch.from_numpy(residual_error).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    result_ids = torch.empty(capacity, dtype=torch.int64, device=device)
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
        result_ids,
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
    scan_counts = counters.cpu().numpy().astype(np.int64)
    direct_count = int(scan_counts[0])
    ambiguous_count = int(scan_counts[1])
    direct = sorted_ids(result_ids, min(direct_count, capacity))
    ambiguous = sorted_ids(ambiguous_ids, min(ambiguous_count, capacity))

    if ambiguous_count:
        refine_ambiguous_fp64_i64[(ambiguous_count,)](
            vectors_gpu,
            vectors_gpu,
            ambiguous_ids,
            result_ids,
            counters,
            threshold_d2,
            N_=N,
            K=D,
            CAPACITY=capacity,
            BLOCK_K=REFINE_BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
    final_counts = counters.cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    overflow = int(final_counts[2])
    if max(direct_count, ambiguous_count) > capacity:
        overflow += 1
    if final_count > capacity:
        overflow += 1
    accepted_upper = sorted_ids(result_ids, min(final_count, capacity))

    oracle_rows = oracle // np.uint64(N)
    oracle_columns = oracle - oracle_rows * np.uint64(N)
    oracle_upper = np.sort(oracle[oracle_rows <= oracle_columns])
    direct_duplicates = duplicate_count(direct)
    ambiguous_duplicates = duplicate_count(ambiguous)
    final_duplicates = duplicate_count(accepted_upper)
    direct_unique = direct_duplicates == 0
    ambiguous_unique = ambiguous_duplicates == 0
    final_unique = final_duplicates == 0
    direct_ambiguous_overlap = int(
        np.intersect1d(
            direct,
            ambiguous,
            assume_unique=direct_unique and ambiguous_unique,
        ).size
    )
    false_accepts = np.setdiff1d(direct, oracle_upper, assume_unique=direct_unique)
    covered = np.union1d(direct, ambiguous)
    false_rejects = np.setdiff1d(oracle_upper, covered, assume_unique=True)
    missing_upper = np.setdiff1d(
        oracle_upper, accepted_upper, assume_unique=final_unique
    )
    extra_upper = np.setdiff1d(
        accepted_upper, oracle_upper, assume_unique=final_unique
    )
    upper_rows = accepted_upper // np.uint64(N)
    upper_columns = accepted_upper - upper_rows * np.uint64(N)
    invalid_upper = int(
        np.count_nonzero(
            (accepted_upper >= np.uint64(N) * np.uint64(N))
            | (upper_rows > upper_columns)
        )
    )
    nonself = upper_rows != upper_columns
    reverse = upper_columns[nonself] * np.uint64(N) + upper_rows[nonself]
    canonical = np.sort(np.concatenate((accepted_upper, reverse)))
    exact_match = bool(
        canonical.size == oracle.size
        and sha256_u64(canonical) == EXPECTED_ORACLE_RAW_SHA256
        and np.array_equal(canonical, oracle)
    )
    gate_pass = bool(
        direct_duplicates == 0
        and ambiguous_duplicates == 0
        and final_duplicates == 0
        and direct_ambiguous_overlap == 0
        and false_accepts.size == 0
        and false_rejects.size == 0
        and missing_upper.size == 0
        and extra_upper.size == 0
        and invalid_upper == 0
        and overflow == 0
        and ambiguous_count <= MAX_AMBIGUOUS
        and exact_match
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "g3b_gpu_correctness_process_not_performance",
        "run_id": args.run_id,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [N, D],
        "threshold_d2": threshold_d2,
        "epsilon_exact_float64": epsilon,
        "epsilon_lower_float32": float(epsilon_lower),
        "epsilon_upper_float32": float(epsilon_upper),
        "source_sha256": sha256_file(vector_path),
        "oracle_file_sha256": sha256_file(oracle_path),
        "oracle_raw_u64_sha256": EXPECTED_ORACLE_RAW_SHA256,
        "analytic_constants": {
            "expression_relative_radius": EXPRESSION_RELATIVE_RADIUS,
            "expression_absolute_radius": EXPRESSION_ABSOLUTE_RADIUS,
            "sqrt_residual_final_relative_radius": SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
            "sqrt_residual_final_absolute_radius": SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
            "accumulator_limit": accumulator_limit,
        },
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
        "direct_accept_upper_pairs": direct_count,
        "ambiguous_upper_pairs": ambiguous_count,
        "fp64_refined_upper_pairs": ambiguous_count,
        "final_accepted_upper_pairs": final_count,
        "direct_raw_u64_sha256": sha256_u64(direct),
        "ambiguous_raw_u64_sha256": sha256_u64(ambiguous),
        "accepted_upper_raw_u64_sha256": sha256_u64(accepted_upper),
        "canonical_pair_count": int(canonical.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "direct_duplicates": direct_duplicates,
        "ambiguous_duplicates": ambiguous_duplicates,
        "final_duplicates": final_duplicates,
        "direct_ambiguous_overlap": direct_ambiguous_overlap,
        "unsafe_direct_accepts": int(false_accepts.size),
        "unsafe_direct_rejects": int(false_rejects.size),
        "missing_upper_pairs": int(missing_upper.size),
        "extra_upper_pairs": int(extra_upper.size),
        "invalid_or_lower_triangle_pairs": invalid_upper,
        "overflow_events": overflow,
        "first_unsafe_direct_accepts": false_accepts[:16].tolist(),
        "first_unsafe_direct_rejects": false_rejects[:16].tolist(),
        "first_missing_upper_pairs": missing_upper[:16].tolist(),
        "first_extra_upper_pairs": extra_upper[:16].tolist(),
        "exact_match": exact_match,
        "process_gate_pass": gate_pass,
        "performance_claim_allowed": False,
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "diagnostic_wall_seconds": time.perf_counter() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "design_sha256": sha256_file(PROJECT / "DESIGN_G3B.md"),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3B_R1.md"),
        "fp64_refinement_source_sha256": sha256_file(
            PROJECT / "src/run_g2a_tensorjoin.py"
        ),
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)

    del (
        vectors_gpu,
        codes_gpu,
        codes_t_gpu,
        scales_gpu,
        norms_gpu,
        errors_gpu,
        tile_rows_gpu,
        tile_columns_gpu,
        result_ids,
        ambiguous_ids,
        counters,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
