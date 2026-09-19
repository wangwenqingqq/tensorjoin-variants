#!/usr/bin/env python3
"""Run one G3C-B three-stage exact self-range-join validation."""

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


EXPERIMENT_ID = "tensorjoin_20260903_g3c_b_gpu_cascade"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/g2a_cifar4096"
FP32_BLOCK_K = 256
FP64_BLOCK_K = 256
FP32_DISTANCE_RELATIVE_RADIUS = 2.0**-14
FP32_INPUT_MAGNITUDE_RADIUS = 2.0**-22
FP32_FINAL_RELATIVE_RADIUS = 2.0**-22
FP32_ABSOLUTE_RADIUS = 4096.0 * float(np.finfo(np.float32).tiny)
MAX_FP64_REFINED = 10_207
EXPECTED_VECTOR_SHA256 = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"
EXPECTED_ORACLE_FILE_SHA256 = "ac93d96ac39ea5a2bc2d1e81baa179a533a6344aae97f058b3b33a8155e6c0c9"
EXPECTED_ORACLE_RAW_SHA256 = "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d"
EXPECTED_G3B_R1_AMBIGUOUS_COUNT = 102_079
EXPECTED_G3B_R1_AMBIGUOUS_SHA256 = (
    "6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42"
)


@triton.jit
def certified_fp32_filter_i64(
    vectors,
    ambiguous_ids,
    result_ids,
    fp64_ids,
    counters,
    threshold_lower,
    threshold_upper,
    distance_relative_radius,
    input_magnitude_radius,
    final_relative_radius,
    absolute_radius,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(ambiguous_ids + pair_index)
    row = pair_id // N_
    column = pair_id - row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float32)
    input_magnitude_d2 = tl.zeros((1,), dtype=tl.float32)
    different_coordinates = tl.zeros((1,), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        x = tl.load(vectors + row * K + k, mask=k < K, other=0.0)
        y = tl.load(vectors + column * K + k, mask=k < K, other=0.0)
        delta = x - y
        input_magnitude = tl.abs(x) + tl.abs(y)
        distance_d2 += tl.sum(delta * delta, axis=0)
        input_magnitude_d2 += tl.sum(input_magnitude * input_magnitude, axis=0)
        different_coordinates += tl.sum((x != y).to(tl.int32), axis=0)

    radius = (
        distance_relative_radius * tl.abs(distance_d2)
        + input_magnitude_radius * tl.abs(input_magnitude_d2)
        + absolute_radius
    )
    final_magnitude = tl.maximum(tl.abs(distance_d2) + radius, 1.0)
    radius += final_relative_radius * final_magnitude + absolute_radius
    lower = tl.maximum(distance_d2 - radius, 0.0)
    upper = distance_d2 + radius
    bitwise_equal = different_coordinates == 0
    accept = bitwise_equal | (upper <= threshold_lower)
    reject = (~bitwise_equal) & (lower > threshold_upper)
    needs_fp64 = ~(accept | reject)
    counter_lane = tl.zeros((1,), dtype=tl.int32)

    position = tl.atomic_add(counters + counter_lane, 1, mask=accept, sem="relaxed")
    can_store = accept & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=accept & ~can_store,
        sem="relaxed",
    )
    fp64_position = tl.atomic_add(
        counters + 3 + counter_lane, 1, mask=needs_fp64, sem="relaxed"
    )
    fp64_can_store = needs_fp64 & (fp64_position < CAPACITY)
    tl.store(fp64_ids + fp64_position, pair_id, mask=fp64_can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=needs_fp64 & ~fp64_can_store,
        sem="relaxed",
    )
    tl.atomic_add(counters + 4 + counter_lane, 1, mask=reject, sem="relaxed")
    tl.atomic_add(counters + 5 + counter_lane, 1, mask=bitwise_equal, sem="relaxed")


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
    result_path = PROJECT / f"results/g3c_b_gpu_cascade_process_{args.run_id}.json"
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
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon = float(metadata["radius"]["epsilon"])
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
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
    fp64_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    counters = torch.zeros(6, dtype=torch.int32, device=device)

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
    g3b_counts = counters.cpu().numpy().astype(np.int64)
    g3b_direct_count = int(g3b_counts[0])
    g3b_ambiguous_count = int(g3b_counts[1])
    g3b_direct = sorted_ids(result_ids, min(g3b_direct_count, capacity))
    g3b_ambiguous = sorted_ids(ambiguous_ids, min(g3b_ambiguous_count, capacity))

    if g3b_ambiguous_count:
        certified_fp32_filter_i64[(g3b_ambiguous_count,)](
            vectors_gpu,
            ambiguous_ids,
            result_ids,
            fp64_ids,
            counters,
            float(threshold_lower),
            float(threshold_upper),
            FP32_DISTANCE_RELATIVE_RADIUS,
            FP32_INPUT_MAGNITUDE_RADIUS,
            FP32_FINAL_RELATIVE_RADIUS,
            FP32_ABSOLUTE_RADIUS,
            N_=N,
            K=D,
            CAPACITY=capacity,
            BLOCK_K=FP32_BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
    fp32_counts = counters.cpu().numpy().astype(np.int64)
    accepted_before_fp64_count = int(fp32_counts[0])
    fp32_accept_count = accepted_before_fp64_count - g3b_direct_count
    fp64_count = int(fp32_counts[3])
    fp32_reject_count = int(fp32_counts[4])
    bitwise_equal_count = int(fp32_counts[5])
    accepted_before_fp64 = sorted_ids(
        result_ids, min(accepted_before_fp64_count, capacity)
    )
    fp32_accept = np.setdiff1d(
        accepted_before_fp64, g3b_direct, assume_unique=True
    )
    fp64_pairs = sorted_ids(fp64_ids, min(fp64_count, capacity))

    if fp64_count:
        refine_ambiguous_fp64_i64[(fp64_count,)](
            vectors_gpu,
            vectors_gpu,
            fp64_ids,
            result_ids,
            counters,
            threshold_d2,
            N_=N,
            K=D,
            CAPACITY=capacity,
            BLOCK_K=FP64_BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
    final_counts = counters.cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    overflow = int(final_counts[2])
    if max(g3b_direct_count, g3b_ambiguous_count, fp64_count, final_count) > capacity:
        overflow += 1
    accepted_upper = sorted_ids(result_ids, min(final_count, capacity))

    oracle_rows = oracle // np.uint64(N)
    oracle_columns = oracle - oracle_rows * np.uint64(N)
    oracle_upper = np.sort(oracle[oracle_rows <= oracle_columns])
    rejected = np.setdiff1d(
        g3b_ambiguous,
        np.union1d(fp32_accept, fp64_pairs),
        assume_unique=True,
    )
    g3b_false_accepts = np.setdiff1d(g3b_direct, oracle_upper, assume_unique=True)
    fp32_false_accepts = np.setdiff1d(fp32_accept, oracle_upper, assume_unique=True)
    fp32_false_rejects = np.intersect1d(rejected, oracle_upper, assume_unique=True)
    missing_upper = np.setdiff1d(oracle_upper, accepted_upper, assume_unique=True)
    extra_upper = np.setdiff1d(accepted_upper, oracle_upper, assume_unique=True)
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
    g3b_ambiguity_contract_pass = bool(
        g3b_ambiguous_count == EXPECTED_G3B_R1_AMBIGUOUS_COUNT
        and sha256_u64(g3b_ambiguous) == EXPECTED_G3B_R1_AMBIGUOUS_SHA256
    )
    stage_partition_pass = bool(
        fp32_accept_count + fp32_reject_count + fp64_count == g3b_ambiguous_count
        and fp32_accept_count == int(fp32_accept.size)
        and fp32_reject_count == int(rejected.size)
    )
    duplicate_counts = {
        "g3b_direct": duplicate_count(g3b_direct),
        "g3b_ambiguous": duplicate_count(g3b_ambiguous),
        "fp32_accept": duplicate_count(fp32_accept),
        "fp64": duplicate_count(fp64_pairs),
        "final": duplicate_count(accepted_upper),
    }
    gate_pass = bool(
        g3b_ambiguity_contract_pass
        and stage_partition_pass
        and all(value == 0 for value in duplicate_counts.values())
        and g3b_false_accepts.size == 0
        and fp32_false_accepts.size == 0
        and fp32_false_rejects.size == 0
        and missing_upper.size == 0
        and extra_upper.size == 0
        and invalid_upper == 0
        and overflow == 0
        and fp64_count <= MAX_FP64_REFINED
        and exact_match
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "g3c_b_gpu_correctness_process_not_performance",
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
        "threshold_d2_exact_float64": threshold_d2,
        "threshold_d2_lower_float32": float(threshold_lower),
        "threshold_d2_upper_float32": float(threshold_upper),
        "epsilon_exact_float64": epsilon,
        "epsilon_lower_float32": float(epsilon_lower),
        "epsilon_upper_float32": float(epsilon_upper),
        "source_sha256": sha256_file(vector_path),
        "oracle_file_sha256": sha256_file(oracle_path),
        "oracle_raw_u64_sha256": EXPECTED_ORACLE_RAW_SHA256,
        "g3b_analytic_constants": {
            "expression_relative_radius": EXPRESSION_RELATIVE_RADIUS,
            "expression_absolute_radius": EXPRESSION_ABSOLUTE_RADIUS,
            "sqrt_residual_final_relative_radius": SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
            "sqrt_residual_final_absolute_radius": SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
            "accumulator_limit": accumulator_limit,
        },
        "fp32_analytic_constants": {
            "distance_relative_radius": FP32_DISTANCE_RELATIVE_RADIUS,
            "input_magnitude_radius": FP32_INPUT_MAGNITUDE_RADIUS,
            "final_relative_radius": FP32_FINAL_RELATIVE_RADIUS,
            "absolute_radius": FP32_ABSOLUTE_RADIUS,
            "block_k": FP32_BLOCK_K,
        },
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
        "g3b_direct_accept_upper_pairs": g3b_direct_count,
        "g3b_ambiguous_upper_pairs": g3b_ambiguous_count,
        "fp32_direct_accept_upper_pairs": fp32_accept_count,
        "fp32_direct_reject_upper_pairs": fp32_reject_count,
        "fp32_bitwise_equal_upper_pairs": bitwise_equal_count,
        "fp64_refined_upper_pairs": fp64_count,
        "fp64_refine_fraction_of_g3b_ambiguity": (
            fp64_count / g3b_ambiguous_count if g3b_ambiguous_count else 0.0
        ),
        "maximum_fp64_refined_upper_pairs": MAX_FP64_REFINED,
        "final_accepted_upper_pairs": final_count,
        "canonical_pair_count": int(canonical.size),
        "g3b_direct_raw_u64_sha256": sha256_u64(g3b_direct),
        "g3b_ambiguous_raw_u64_sha256": sha256_u64(g3b_ambiguous),
        "fp32_accept_raw_u64_sha256": sha256_u64(fp32_accept),
        "fp32_reject_raw_u64_sha256": sha256_u64(rejected),
        "fp64_raw_u64_sha256": sha256_u64(fp64_pairs),
        "accepted_upper_raw_u64_sha256": sha256_u64(accepted_upper),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "duplicate_counts": duplicate_counts,
        "g3b_ambiguity_contract_pass": g3b_ambiguity_contract_pass,
        "stage_partition_pass": stage_partition_pass,
        "unsafe_g3b_direct_accepts": int(g3b_false_accepts.size),
        "unsafe_fp32_direct_accepts": int(fp32_false_accepts.size),
        "unsafe_fp32_direct_rejects": int(fp32_false_rejects.size),
        "missing_upper_pairs": int(missing_upper.size),
        "extra_upper_pairs": int(extra_upper.size),
        "invalid_or_lower_triangle_pairs": invalid_upper,
        "overflow_events": overflow,
        "first_unsafe_g3b_direct_accepts": g3b_false_accepts[:16].tolist(),
        "first_unsafe_fp32_direct_accepts": fp32_false_accepts[:16].tolist(),
        "first_unsafe_fp32_direct_rejects": fp32_false_rejects[:16].tolist(),
        "first_missing_upper_pairs": missing_upper[:16].tolist(),
        "first_extra_upper_pairs": extra_upper[:16].tolist(),
        "exact_match": exact_match,
        "process_gate_pass": gate_pass,
        "performance_claim_allowed": False,
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "diagnostic_wall_seconds": time.perf_counter() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3C_B.md"),
        "g3b_r1_runner_sha256": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
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
        fp64_ids,
        counters,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
