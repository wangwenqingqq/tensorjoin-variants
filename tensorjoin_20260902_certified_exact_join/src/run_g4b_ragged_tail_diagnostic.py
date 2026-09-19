#!/usr/bin/env python3
"""Localize the first failing G4B Fashion boundary with an in-process G3B A/B."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    analytic_certificate_compact_i64,
    outward_float32,
)
from run_g4b_public_opportunity import quantize_with_analytic_residual


PROJECT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT / "data/g4b_public/fashion784"
N = 1_024
D = 784
TARGET_DEGREE = 1
EXPECTED_VECTOR_FILE_SHA256 = (
    "16149e1a1deaa2afeb205a0d22d49d21d1b784ca654851985603777e9c8b29d8"
)
EXPECTED_ORACLE_FILE_SHA256 = (
    "6b2573b3441999438311706077087759cd8993664fd247d09099203509e06d19"
)
EXPECTED_ORACLE_RAW_SHA256 = (
    "15132c49796e6e52059bb395acb0f84bb993fcebcd7fa4735112922720357311"
)


@triton.jit
def analytic_certificate_ragged_safe_i64(
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
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        a = tl.load(
            codes + offsets_m[:, None] * K + k[None, :],
            mask=(offsets_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            codes_transposed + k[:, None] * N_ + offsets_n[None, :],
            mask=(k[:, None] < K) & (offsets_n[None, :] < N_),
            other=0,
        )
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

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
    final_radius = final_relative_radius * final_magnitude + final_absolute_radius
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


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u8").tobytes(order="C")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
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


def run_classification(
    kernel,
    buffers: dict[str, torch.Tensor],
    scheduled_tiles: int,
    capacity: int,
    epsilon_lower: float,
    epsilon_upper: float,
) -> dict[str, object]:
    buffers["counters"].zero_()
    kernel[(scheduled_tiles,)](
        buffers["codes"],
        buffers["codes_t"],
        buffers["scales"],
        buffers["norms"],
        buffers["errors"],
        buffers["tile_rows"],
        buffers["tile_columns"],
        buffers["result_ids"],
        buffers["ambiguous_ids"],
        buffers["counters"],
        epsilon_lower,
        epsilon_upper,
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
    counters = buffers["counters"].cpu().numpy().astype(np.int64)
    direct_count = int(counters[0])
    ambiguous_count = int(counters[1])
    overflow = int(counters[2])
    direct = sorted_ids(buffers["result_ids"], min(direct_count, capacity))
    ambiguous = sorted_ids(
        buffers["ambiguous_ids"], min(ambiguous_count, capacity)
    )
    return {
        "direct": direct,
        "ambiguous": ambiguous,
        "direct_count": direct_count,
        "ambiguous_count": ambiguous_count,
        "reject_count": N * (N + 1) // 2 - direct_count - ambiguous_count,
        "overflow": overflow,
    }


def summarize_classification(
    classification: dict[str, object], oracle: np.ndarray
) -> dict[str, object]:
    direct = classification["direct"]
    ambiguous = classification["ambiguous"]
    covered = np.union1d(direct, ambiguous)
    unsafe_accepts = np.setdiff1d(direct, oracle)
    unsafe_rejects = np.setdiff1d(oracle, covered)
    direct_duplicates = int(np.count_nonzero(direct[1:] == direct[:-1]))
    ambiguous_duplicates = int(np.count_nonzero(ambiguous[1:] == ambiguous[:-1]))
    partition_pass = bool(
        classification["direct_count"]
        + classification["ambiguous_count"]
        + classification["reject_count"]
        == N * (N + 1) // 2
    )
    sound = bool(
        partition_pass
        and classification["overflow"] == 0
        and direct_duplicates == 0
        and ambiguous_duplicates == 0
        and unsafe_accepts.size == 0
        and unsafe_rejects.size == 0
    )
    return {
        "direct_count": classification["direct_count"],
        "ambiguous_count": classification["ambiguous_count"],
        "reject_count": classification["reject_count"],
        "direct_raw_u64_sha256": sha256_u64(direct),
        "ambiguous_raw_u64_sha256": sha256_u64(ambiguous),
        "direct_duplicate_count": direct_duplicates,
        "ambiguous_duplicate_count": ambiguous_duplicates,
        "overflow": classification["overflow"],
        "partition_pass": partition_pass,
        "unsafe_direct_accepts": int(unsafe_accepts.size),
        "unsafe_direct_rejects": int(unsafe_rejects.size),
        "first_unsafe_direct_accepts": unsafe_accepts[:16].tolist(),
        "first_unsafe_direct_rejects": unsafe_rejects[:16].tolist(),
        "sound": sound,
    }


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Diagnostic requires physical GPU1 as sole visible device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / "results/g4b_ragged_tail_diagnostic.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    vector_path = DATASET_ROOT / "vectors_f32.npy"
    oracle_path = DATASET_ROOT / "n1024/k1/oracle_upper_ids_u64.npy"
    metadata_path = DATASET_ROOT / "n1024/k1/metadata.json"
    if sha256_file(vector_path) != EXPECTED_VECTOR_FILE_SHA256:
        raise RuntimeError("Vector file hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")
    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False)[:N], dtype=np.float32
    )
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Oracle raw hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    epsilon_lower, epsilon_upper = outward_float32(float(metadata["epsilon"]))
    codes, scales, norms, errors = quantize_with_analytic_residual(vectors)
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)
    capacity = scheduled_tiles * BLOCK_M * BLOCK_N
    device = torch.device("cuda:0")
    buffers = {
        "codes": torch.from_numpy(codes).to(device),
        "codes_t": torch.from_numpy(codes.T.copy()).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "norms": torch.from_numpy(norms).to(device),
        "errors": torch.from_numpy(errors).to(device),
        "tile_rows": torch.from_numpy(tile_rows).to(device),
        "tile_columns": torch.from_numpy(tile_columns).to(device),
        "result_ids": torch.empty(capacity, dtype=torch.int64, device=device),
        "ambiguous_ids": torch.empty(capacity, dtype=torch.int64, device=device),
        "counters": torch.zeros(3, dtype=torch.int32, device=device),
    }

    suspect = run_classification(
        analytic_certificate_compact_i64,
        buffers,
        scheduled_tiles,
        capacity,
        float(epsilon_lower),
        float(epsilon_upper),
    )
    suspect_summary = summarize_classification(suspect, oracle)
    corrected = run_classification(
        analytic_certificate_ragged_safe_i64,
        buffers,
        scheduled_tiles,
        capacity,
        float(epsilon_lower),
        float(epsilon_upper),
    )
    corrected_summary = summarize_classification(corrected, oracle)
    direct_changed = np.setxor1d(suspect["direct"], corrected["direct"])
    ambiguous_changed = np.setxor1d(suspect["ambiguous"], corrected["ambiguous"])
    diagnosis_pass = bool(not suspect_summary["sound"] and corrected_summary["sound"])
    result = {
        "experiment_id": "tensorjoin_20260903_g4b_ragged_tail_diagnostic",
        "measurement_status": "in_process_correctness_ab_not_performance",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "dataset_id": "fashion784",
        "shape": [N, D],
        "target_average_directed_nonself_degree": TARGET_DEGREE,
        "oracle_upper_count": int(oracle.size),
        "oracle_file_sha256": sha256_file(oracle_path),
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "suspect": suspect_summary,
        "ragged_safe": corrected_summary,
        "direct_classification_changed_pairs": int(direct_changed.size),
        "ambiguous_classification_changed_pairs": int(ambiguous_changed.size),
        "first_direct_changed_pairs": direct_changed[:16].tolist(),
        "first_ambiguous_changed_pairs": ambiguous_changed[:16].tolist(),
        "first_failing_boundary": (
            "G3B final K tile load mask" if diagnosis_pass else "not_localized"
        ),
        "source_reading_hypothesis": {
            "dimension": D,
            "block_k": BLOCK_K,
            "final_block_start": (D // BLOCK_K) * BLOCK_K,
            "valid_coordinates_in_final_block": D % BLOCK_K,
            "suspect_unmasked_coordinates_in_final_block": BLOCK_K - D % BLOCK_K,
            "suspect_mask_uses_static_offsets_k_less_than_K": True,
            "ragged_safe_mask_uses_block_start_plus_offsets_k_less_than_K": True,
        },
        "diagnosis_pass": diagnosis_pass,
        "performance_claim_allowed": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "suspect_source_sha256": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
        "generic_runner_sha256": sha256_file(
            PROJECT / "src/run_g4b_public_opportunity.py"
        ),
        "protocol_sha256": sha256_file(
            PROJECT / "PROTOCOL_G4B_RAGGED_DIAGNOSTIC.md"
        ),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if diagnosis_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
