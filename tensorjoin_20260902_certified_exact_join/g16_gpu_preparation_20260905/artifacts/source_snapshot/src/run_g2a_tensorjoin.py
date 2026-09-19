#!/usr/bin/env python3
"""Run the G2A scalable TensorJoin canonical-output correctness gate."""

from __future__ import annotations

import argparse
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

from run_d1_triton import quantize_per_vector, upward_float32


EXPERIMENT_ID = "tensorjoin_20260903_tensorjoin_cifar4096_g2a"
N = 4_096
D = 512
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
INITIAL_RESULTS_PER_POINT = 8
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2a_cifar4096"


@triton.jit
def fused_int8_certificate_compact_i64(
    query_codes,
    base_codes_transposed,
    query_scales,
    base_scales,
    query_reconstructed_norm2,
    base_reconstructed_norm2,
    query_errors,
    base_errors,
    result_ids,
    ambiguous_ids,
    counters,
    threshold_d2,
    bound_pad,
    M: tl.constexpr,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_m = tl.program_id(axis=0)
    program_n = tl.program_id(axis=1)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    query_ptrs = query_codes + offsets_m[:, None] * K + offsets_k[None, :]
    base_ptrs = base_codes_transposed + offsets_k[:, None] * N_ + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        query_block = tl.load(
            query_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        base_block = tl.load(
            base_ptrs,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(query_block, base_block, accumulator, out_dtype=tl.int32)
        query_ptrs += BLOCK_K
        base_ptrs += BLOCK_K * N_

    query_scale = tl.load(query_scales + offsets_m, mask=offsets_m < M, other=0.0)
    base_scale = tl.load(base_scales + offsets_n, mask=offsets_n < N_, other=0.0)
    query_norm2 = tl.load(
        query_reconstructed_norm2 + offsets_m, mask=offsets_m < M, other=0.0
    )
    base_norm2 = tl.load(
        base_reconstructed_norm2 + offsets_n, mask=offsets_n < N_, other=0.0
    )
    query_error = tl.load(query_errors + offsets_m, mask=offsets_m < M, other=0.0)
    base_error = tl.load(base_errors + offsets_n, mask=offsets_n < N_, other=0.0)
    dot = accumulator.to(tl.float32) * query_scale[:, None] * base_scale[None, :]
    reconstructed_d2 = query_norm2[:, None] + base_norm2[None, :] - 2.0 * dot
    reconstructed_d = tl.sqrt(tl.maximum(reconstructed_d2, 0.0))
    radius = query_error[:, None] + base_error[None, :] + bound_pad
    lower = tl.maximum(reconstructed_d - radius, 0.0)
    upper = reconstructed_d + radius
    lower_d2 = lower * lower
    upper_d2 = upper * upper
    valid = (offsets_m[:, None] < M) & (offsets_n[None, :] < N_)
    accept = valid & (upper_d2 <= threshold_d2)
    reject = valid & (lower_d2 > threshold_d2)
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


@triton.jit
def filter_ambiguous_fp32_i64(
    query,
    base,
    ambiguous_ids,
    result_ids,
    fp64_ids,
    counters,
    threshold_d2,
    fp32_guard,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(ambiguous_ids + pair_index)
    query_row = pair_id // N_
    base_row = pair_id - query_row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float32)
    different_coordinates = tl.zeros((1,), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_values = tl.load(query + query_row * K + k, mask=k < K, other=0.0)
        base_values = tl.load(base + base_row * K + k, mask=k < K, other=0.0)
        delta = query_values - base_values
        distance_d2 += tl.sum(delta * delta, axis=0)
        different_coordinates += tl.sum((query_values != base_values).to(tl.int32), axis=0)

    bitwise_equal = different_coordinates == 0
    accept = bitwise_equal | (distance_d2 <= threshold_d2 - fp32_guard)
    reject = (~bitwise_equal) & (distance_d2 > threshold_d2 + fp32_guard)
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
    fp64_store = needs_fp64 & (fp64_position < CAPACITY)
    tl.store(fp64_ids + fp64_position, pair_id, mask=fp64_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=needs_fp64 & ~fp64_store,
        sem="relaxed",
    )
    tl.atomic_add(counters + 4 + counter_lane, 1, mask=bitwise_equal, sem="relaxed")


@triton.jit
def refine_ambiguous_fp64_i64(
    query,
    base,
    fp64_ids,
    result_ids,
    counters,
    threshold_d2,
    N_: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(fp64_ids + pair_index)
    query_row = pair_id // N_
    base_row = pair_id - query_row * N_
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_values = tl.load(query + query_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        base_values = tl.load(base + base_row * K + k, mask=k < K, other=0.0).to(
            tl.float64
        )
        delta = query_values - base_values
        distance_d2 += tl.sum(delta * delta, axis=0)
    inside = distance_d2 <= threshold_d2
    counter_lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(counters + counter_lane, 1, mask=inside, sem="relaxed")
    can_store = inside & (position < CAPACITY)
    tl.store(result_ids + position, pair_id, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True, choices=range(2))
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u8").tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sorted_device_ids(values: torch.Tensor, count: int) -> np.ndarray:
    return np.sort(values[:count].cpu().numpy().astype(np.uint64, copy=False))


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"G2A TensorJoin requires CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g2a_tensorjoin_process_{args.run_id}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    for path in (DATA_DIR / "vectors_f32.npy", DATA_DIR / "oracle_pairs_u64.npy", DATA_DIR / "metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    with (DATA_DIR / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    vectors = np.ascontiguousarray(
        np.load(DATA_DIR / "vectors_f32.npy", allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(
        np.load(DATA_DIR / "oracle_pairs_u64.npy", allow_pickle=False), dtype=np.uint64
    )
    if vectors.shape != (N, D):
        raise ValueError(vectors.shape)

    codes, scales, errors = quantize_per_vector(vectors)
    reconstructed_norm2 = (
        np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * scales.astype(np.float64) ** 2
    )
    device = torch.device("cuda:0")
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2.astype(np.float32)).to(device)
    errors_gpu = torch.from_numpy(upward_float32(errors)).to(device)
    scan_grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))

    run = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.run_id,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "threshold_d2": threshold_d2,
        "input_sha256": sha256_file(DATA_DIR / "vectors_f32.npy"),
        "oracle_file_sha256": sha256_file(DATA_DIR / "oracle_pairs_u64.npy"),
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "quantization_source_sha256": sha256_file(PROJECT / "src/run_d1_triton.py"),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    capacity = N * INITIAL_RESULTS_PER_POINT
    attempts: list[dict[str, int]] = []
    while True:
        result_ids = torch.empty(capacity, dtype=torch.int64, device=device)
        ambiguous_ids = torch.empty(capacity, dtype=torch.int64, device=device)
        fp64_ids = torch.empty(capacity, dtype=torch.int64, device=device)
        counters = torch.zeros(5, dtype=torch.int32, device=device)
        fused_int8_certificate_compact_i64[scan_grid](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            scales_gpu,
            norms_gpu,
            norms_gpu,
            errors_gpu,
            errors_gpu,
            result_ids,
            ambiguous_ids,
            counters,
            threshold_d2,
            BOUND_PAD,
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
        attempt = {
            "capacity": capacity,
            "direct_accept_pairs": int(scan_counts[0]),
            "ambiguous_pairs": int(scan_counts[1]),
            "overflow_events": int(scan_counts[2]),
        }
        attempts.append(attempt)
        print("CAPACITY_ATTEMPT " + json.dumps(attempt, sort_keys=True), flush=True)
        if (
            scan_counts[2] == 0
            and scan_counts[0] <= capacity
            and scan_counts[1] <= capacity
        ):
            break
        if capacity >= N * N:
            raise RuntimeError("Capacity retry exceeded the full pair space")
        capacity = min(capacity * 2, N * N)

    direct_count = int(scan_counts[0])
    ambiguous_count = int(scan_counts[1])
    direct_ids = sorted_device_ids(result_ids, direct_count)
    ambiguous_cpu = sorted_device_ids(ambiguous_ids, ambiguous_count)
    direct_duplicates = int(np.count_nonzero(direct_ids[1:] == direct_ids[:-1]))
    ambiguous_duplicates = int(
        np.count_nonzero(ambiguous_cpu[1:] == ambiguous_cpu[:-1])
    )
    cross_stage_duplicates = int(np.intersect1d(direct_ids, ambiguous_cpu).size)
    int8_false_accepts = int(np.setdiff1d(direct_ids, oracle, assume_unique=True).size)
    int8_false_rejects = int(
        np.setdiff1d(
            oracle,
            np.union1d(direct_ids, ambiguous_cpu),
            assume_unique=True,
        ).size
    )

    filter_ambiguous_fp32_i64[(ambiguous_count,)](
        vectors_gpu,
        vectors_gpu,
        ambiguous_ids,
        result_ids,
        fp64_ids,
        counters,
        threshold_d2,
        FP32_DISTANCE_GUARD,
        N_=N,
        K=D,
        CAPACITY=capacity,
        BLOCK_K=REFINE_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    after_fp32 = counters.cpu().numpy().astype(np.int64)
    after_fp32_count = int(after_fp32[0])
    fp64_count = int(after_fp32[3])
    equality_count = int(after_fp32[4])
    fp32_accept_ids = sorted_device_ids(
        result_ids[direct_count:after_fp32_count], after_fp32_count - direct_count
    )
    fp64_cpu = sorted_device_ids(fp64_ids, fp64_count)
    fp32_reject_ids = np.setdiff1d(
        ambiguous_cpu, np.union1d(fp32_accept_ids, fp64_cpu), assume_unique=True
    )
    fp32_false_accepts = int(
        np.setdiff1d(fp32_accept_ids, oracle, assume_unique=True).size
    )
    fp32_false_rejects = int(
        np.intersect1d(fp32_reject_ids, oracle, assume_unique=True).size
    )

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
        BLOCK_K=REFINE_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    final_counts = counters.cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    final_ids = sorted_device_ids(result_ids, min(final_count, capacity))
    invalid_pairs = int(np.count_nonzero(final_ids >= N * N))
    final_duplicates = int(np.count_nonzero(final_ids[1:] == final_ids[:-1]))
    unique_final = np.unique(final_ids)
    missing = np.setdiff1d(oracle, unique_final, assume_unique=True)
    extra = np.setdiff1d(unique_final, oracle, assume_unique=True)
    later_overflow = int(final_counts[2])
    exact = (
        direct_duplicates == 0
        and ambiguous_duplicates == 0
        and cross_stage_duplicates == 0
        and int8_false_accepts == 0
        and int8_false_rejects == 0
        and fp32_false_accepts == 0
        and fp32_false_rejects == 0
        and later_overflow == 0
        and invalid_pairs == 0
        and final_duplicates == 0
        and missing.size == 0
        and extra.size == 0
        and final_ids.size == oracle.size
    )

    result = {
        **run,
        "measurement_status": "correctness_and_capacity_admission_only_no_performance_claim",
        "bound_pad": BOUND_PAD,
        "fp32_distance_guard": FP32_DISTANCE_GUARD,
        "initial_capacity": N * INITIAL_RESULTS_PER_POINT,
        "accepted_capacity": capacity,
        "capacity_attempts": attempts,
        "direct_accept_pairs": direct_count,
        "ambiguous_pairs": ambiguous_count,
        "fp32_accept_pairs": int(fp32_accept_ids.size),
        "fp32_reject_pairs": int(fp32_reject_ids.size),
        "fp64_refine_pairs": fp64_count,
        "bitwise_equal_ambiguous_pairs": equality_count,
        "direct_duplicates": direct_duplicates,
        "ambiguous_duplicates": ambiguous_duplicates,
        "cross_stage_duplicates": cross_stage_duplicates,
        "int8_false_accepts": int8_false_accepts,
        "int8_false_rejects": int8_false_rejects,
        "fp32_false_accepts": fp32_false_accepts,
        "fp32_false_rejects": fp32_false_rejects,
        "later_overflow_events": later_overflow,
        "canonical_pair_count": int(final_ids.size),
        "canonical_unique_pair_count": int(unique_final.size),
        "canonical_raw_u64_sha256": sha256_u64(final_ids),
        "invalid_pairs": invalid_pairs,
        "duplicate_pairs": final_duplicates,
        "missing_pairs": int(missing.size),
        "extra_pairs": int(extra.size),
        "first_missing_pairs": missing[:16].astype(np.uint64).tolist(),
        "first_extra_pairs": extra[:16].astype(np.uint64).tolist(),
        "exact_match": exact,
        "diagnostic_total_wall_s": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
