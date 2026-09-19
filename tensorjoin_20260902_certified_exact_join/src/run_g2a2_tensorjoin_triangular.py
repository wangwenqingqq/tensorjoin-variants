#!/usr/bin/env python3
"""Validate the proof-bounded triangular TensorJoin path on the G2A oracle."""

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
from run_g2a_tensorjoin import (
    filter_ambiguous_fp32_i64,
    refine_ambiguous_fp64_i64,
)


EXPERIMENT_ID = "tensorjoin_20260903_tensorjoin_cifar4096_triangular_g2a2"
N = 4_096
D = 512
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
TILES_PER_BATCH = 4_096
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2a_cifar4096"


@triton.jit
def triangular_int8_certificate_compact_i64(
    query_codes,
    base_codes_transposed,
    query_scales,
    base_scales,
    query_reconstructed_norm2,
    base_reconstructed_norm2,
    query_errors,
    base_errors,
    tile_rows,
    tile_columns,
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
    program = tl.program_id(axis=0)
    program_m = tl.load(tile_rows + program)
    program_n = tl.load(tile_columns + program)
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
    valid = (
        (offsets_m[:, None] < M)
        & (offsets_n[None, :] < N_)
        & (offsets_m[:, None] <= offsets_n[None, :])
    )
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
    canonical = np.asarray(values, dtype="<u8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def device_ids(values: torch.Tensor, start: int, stop: int) -> np.ndarray:
    if stop <= start:
        return np.empty(0, dtype=np.uint64)
    return values[start:stop].cpu().numpy().astype(np.uint64, copy=False)


def concatenate_sorted(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.sort(np.concatenate(parts).astype(np.uint64, copy=False))


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            f"G2A2 TensorJoin requires CUDA_VISIBLE_DEVICES=0, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g2a2_tensorjoin_process_{args.run_id}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    required = (
        DATA_DIR / "vectors_f32.npy",
        DATA_DIR / "oracle_pairs_u64.npy",
        DATA_DIR / "metadata.json",
    )
    for path in required:
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
    oracle_rows = oracle // np.uint64(N)
    oracle_columns = oracle - oracle_rows * np.uint64(N)
    oracle_upper = oracle[oracle_rows <= oracle_columns]

    codes, scales, errors = quantize_per_vector(vectors)
    reconstructed_norm2 = (
        np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * scales.astype(np.float64) ** 2
    )
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    if tile_rows.size != tile_extent * (tile_extent + 1) // 2:
        raise AssertionError("Triangular tile schedule size mismatch")

    device = torch.device("cuda:0")
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2.astype(np.float32)).to(device)
    errors_gpu = torch.from_numpy(upward_float32(errors)).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)

    max_batch_tiles = min(TILES_PER_BATCH, int(tile_rows.size))
    capacity = max_batch_tiles * BLOCK_M * BLOCK_N
    result_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    ambiguous_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    fp64_ids = torch.empty(capacity, dtype=torch.int64, device=device)

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
        "oracle_upper_raw_u64_sha256": sha256_u64(oracle_upper),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "imported_full_scan_runner_sha256": sha256_file(
            PROJECT / "src/run_g2a_tensorjoin.py"
        ),
        "quantization_source_sha256": sha256_file(PROJECT / "src/run_d1_triton.py"),
        "tile_extent": tile_extent,
        "scheduled_tiles": int(tile_rows.size),
        "tiles_per_batch": TILES_PER_BATCH,
        "buffer_capacity_per_stage": capacity,
        "capacity_rule": "scheduled_tiles_in_largest_batch * 64 * 64",
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    direct_parts: list[np.ndarray] = []
    ambiguous_parts: list[np.ndarray] = []
    fp32_accept_parts: list[np.ndarray] = []
    fp64_parts: list[np.ndarray] = []
    accepted_parts: list[np.ndarray] = []
    batch_records: list[dict[str, int]] = []
    total_equality = 0
    total_overflow = 0

    for batch_index, tile_start in enumerate(range(0, tile_rows.size, TILES_PER_BATCH)):
        tile_stop = min(tile_start + TILES_PER_BATCH, int(tile_rows.size))
        batch_tiles = tile_stop - tile_start
        counters = torch.zeros(5, dtype=torch.int32, device=device)
        triangular_int8_certificate_compact_i64[(batch_tiles,)](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            scales_gpu,
            norms_gpu,
            norms_gpu,
            errors_gpu,
            errors_gpu,
            tile_rows_gpu[tile_start:tile_stop],
            tile_columns_gpu[tile_start:tile_stop],
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
        direct_count = int(scan_counts[0])
        ambiguous_count = int(scan_counts[1])
        direct_parts.append(device_ids(result_ids, 0, direct_count))
        ambiguous_parts.append(device_ids(ambiguous_ids, 0, ambiguous_count))

        if ambiguous_count:
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
        fp32_accept_parts.append(
            device_ids(result_ids, direct_count, after_fp32_count)
        )
        fp64_parts.append(device_ids(fp64_ids, 0, fp64_count))

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
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
            )
            torch.cuda.synchronize()
        final_counts = counters.cpu().numpy().astype(np.int64)
        final_count = int(final_counts[0])
        overflow = int(final_counts[2])
        if max(direct_count, ambiguous_count, after_fp32_count, fp64_count, final_count) > capacity:
            overflow += 1
        accepted_parts.append(device_ids(result_ids, 0, min(final_count, capacity)))
        total_equality += equality_count
        total_overflow += overflow
        record = {
            "batch_index": batch_index,
            "tile_start": tile_start,
            "tile_stop": tile_stop,
            "scheduled_tiles": batch_tiles,
            "worst_case_valid_pairs": batch_tiles * BLOCK_M * BLOCK_N,
            "direct_accept_pairs": direct_count,
            "ambiguous_pairs": ambiguous_count,
            "fp32_accept_pairs": after_fp32_count - direct_count,
            "fp64_refine_pairs": fp64_count,
            "final_accept_pairs": final_count,
            "overflow_events": overflow,
        }
        batch_records.append(record)
        print("BATCH_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)

    direct_ids = concatenate_sorted(direct_parts)
    ambiguous = concatenate_sorted(ambiguous_parts)
    fp32_accept = concatenate_sorted(fp32_accept_parts)
    fp64_candidates = concatenate_sorted(fp64_parts)
    accepted_upper = concatenate_sorted(accepted_parts)
    fp32_reject = np.setdiff1d(
        ambiguous, np.union1d(fp32_accept, fp64_candidates), assume_unique=True
    )

    upper_rows = accepted_upper // np.uint64(N)
    upper_columns = accepted_upper - upper_rows * np.uint64(N)
    nonself = upper_rows < upper_columns
    reverse = upper_columns[nonself] * np.uint64(N) + upper_rows[nonself]
    directed = np.sort(np.concatenate((accepted_upper, reverse)).astype(np.uint64))

    direct_duplicates = duplicate_count(direct_ids)
    ambiguous_duplicates = duplicate_count(ambiguous)
    cross_stage_duplicates = int(np.intersect1d(direct_ids, ambiguous).size)
    accepted_upper_duplicates = duplicate_count(accepted_upper)
    directed_duplicates = duplicate_count(directed)
    invalid_upper = int(
        np.count_nonzero(
            (accepted_upper >= np.uint64(N) * np.uint64(N))
            | (upper_rows > upper_columns)
        )
    )
    int8_false_accepts = int(
        np.setdiff1d(direct_ids, oracle_upper, assume_unique=True).size
    )
    int8_false_rejects = int(
        np.setdiff1d(
            oracle_upper,
            np.union1d(direct_ids, ambiguous),
            assume_unique=True,
        ).size
    )
    fp32_false_accepts = int(
        np.setdiff1d(fp32_accept, oracle_upper, assume_unique=True).size
    )
    fp32_false_rejects = int(
        np.intersect1d(fp32_reject, oracle_upper, assume_unique=True).size
    )
    missing = np.setdiff1d(oracle, directed, assume_unique=True)
    extra = np.setdiff1d(directed, oracle, assume_unique=True)
    exact = (
        direct_duplicates == 0
        and ambiguous_duplicates == 0
        and cross_stage_duplicates == 0
        and accepted_upper_duplicates == 0
        and directed_duplicates == 0
        and invalid_upper == 0
        and int8_false_accepts == 0
        and int8_false_rejects == 0
        and fp32_false_accepts == 0
        and fp32_false_rejects == 0
        and total_overflow == 0
        and missing.size == 0
        and extra.size == 0
        and directed.size == oracle.size
        and accepted_upper.size == oracle_upper.size
    )

    result = {
        **run,
        "measurement_status": "g2a2_correctness_admission_only_no_performance_claim",
        "bound_pad": BOUND_PAD,
        "fp32_distance_guard": FP32_DISTANCE_GUARD,
        "analytical_upper_comparisons": N * (N + 1) // 2,
        "oracle_upper_pair_count": int(oracle_upper.size),
        "batch_count": len(batch_records),
        "batches": batch_records,
        "direct_accept_upper_pairs": int(direct_ids.size),
        "ambiguous_upper_pairs": int(ambiguous.size),
        "fp32_accept_upper_pairs": int(fp32_accept.size),
        "fp32_reject_upper_pairs": int(fp32_reject.size),
        "fp64_refine_upper_pairs": int(fp64_candidates.size),
        "bitwise_equal_ambiguous_upper_pairs": total_equality,
        "direct_duplicates": direct_duplicates,
        "ambiguous_duplicates": ambiguous_duplicates,
        "cross_stage_duplicates": cross_stage_duplicates,
        "accepted_upper_duplicates": accepted_upper_duplicates,
        "directed_duplicates": directed_duplicates,
        "invalid_or_lower_triangle_upper_ids": invalid_upper,
        "int8_false_accepts": int8_false_accepts,
        "int8_false_rejects": int8_false_rejects,
        "fp32_false_accepts": fp32_false_accepts,
        "fp32_false_rejects": fp32_false_rejects,
        "overflow_events": total_overflow,
        "accepted_upper_pair_count": int(accepted_upper.size),
        "accepted_upper_raw_u64_sha256": sha256_u64(accepted_upper),
        "canonical_pair_count": int(directed.size),
        "canonical_raw_u64_sha256": sha256_u64(directed),
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
