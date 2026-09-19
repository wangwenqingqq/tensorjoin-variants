#!/usr/bin/env python3
"""Validate one G4B-R1 public dataset with the ragged-safe G3B kernel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton

from run_g2a_tensorjoin import refine_ambiguous_fp64_i64
from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    outward_float32,
)
from g4b_r1_ragged_safe_kernel import analytic_certificate_ragged_safe_i64
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    certified_fp32_filter_i64,
)


EXPERIMENT_ID = "tensorjoin_20260903_g4b_r1_public_breadth_opportunity"
PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data/g4b_public"
SCALES = (1_024, 2_048, 4_096)
TARGET_DEGREES = (1, 16, 64)
DATASETS = {
    "sift128": {
        "dimension": 128,
        "vector_file_sha256": (
            "91648432b84009381b22aedc3fba487e787ee4e659ed0aaf5392f8e1042d9126"
        ),
        "row_file_sha256": (
            "fcd014f989df767832e5028eda18607d5e726dcdfd5d9f17abbad3c1ac739fa5"
        ),
    },
    "cifar_gist512": {
        "dimension": 512,
        "vector_file_sha256": (
            "cea1987b6a07df43a71d5361afa5f1630c8f1abbbc44493d60af0898e6394702"
        ),
        "row_file_sha256": (
            "b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c"
        ),
    },
    "fashion784": {
        "dimension": 784,
        "vector_file_sha256": (
            "16149e1a1deaa2afeb205a0d22d49d21d1b784ca654851985603777e9c8b29d8"
        ),
        "row_file_sha256": (
            "b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c"
        ),
    },
}
FP64_BLOCK_K = 256
EXPECTED_DIAGNOSTIC_RESULT_SHA256 = (
    "dce93811383fd81fe4144a07ee373650eec1c733c2331d4d7ffd9b13eb3fc5a8"
)
EXPECTED_DIAGNOSTIC_MANIFEST_SHA256 = (
    "d8ff9266a67b3ab6d2ccbf80b4e18aaf8a16d9064b92d633ffb8febc35660ce4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return sha256_array(values, "<u8")


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


def upward_float32(values: np.ndarray) -> np.ndarray:
    rounded = values.astype(np.float32)
    increment = rounded.astype(np.float64) < values.astype(np.float64)
    rounded[increment] = np.nextafter(rounded[increment], np.float32(np.inf))
    return rounded


def quantize_with_analytic_residual(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply G3B-R1 quantization with the cell's native dimension in gamma."""
    dimension = int(vectors.shape[1])
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
    operations = 2 * dimension + 2
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
    outputs = (codes, scales, reconstructed_norm2, residual_error)
    if any(not np.isfinite(value).all() for value in outputs[1:]):
        raise RuntimeError("Non-finite quantization metadata")
    return tuple(np.ascontiguousarray(value) for value in outputs)


def sorted_ids(tensor: torch.Tensor, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    return np.sort(tensor[:count].cpu().numpy().astype(np.uint64, copy=False))


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def validate_cell(
    *,
    dataset: str,
    n: int,
    dimension: int,
    target_degree: int,
    vectors: np.ndarray,
    vector_prefix_sha256: str,
    row_prefix_sha256: str,
    dataset_root: Path,
    device_buffers: dict[str, torch.Tensor],
    scheduled_tiles: int,
    capacity: int,
) -> dict[str, object]:
    cell_root = dataset_root / f"n{n}/k{target_degree}"
    oracle_path = cell_root / "oracle_upper_ids_u64.npy"
    metadata_path = cell_root / "metadata.json"
    if not oracle_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(cell_root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if metadata["dataset_id"] != dataset or int(metadata["n"]) != n:
        raise RuntimeError("Oracle metadata cell mismatch")
    if int(metadata["dimension"]) != dimension:
        raise RuntimeError("Oracle dimension mismatch")
    if int(metadata["target_average_directed_nonself_degree"]) != target_degree:
        raise RuntimeError("Oracle target-degree mismatch")
    if metadata["vector_prefix_raw_sha256"] != vector_prefix_sha256:
        raise RuntimeError("Vector-prefix hash mismatch")
    if metadata["source_row_prefix_raw_sha256"] != row_prefix_sha256:
        raise RuntimeError("Row-prefix hash mismatch")
    oracle_raw_sha256 = sha256_u64(oracle)
    if metadata["oracle_upper_raw_sha256"] != oracle_raw_sha256:
        raise RuntimeError("Oracle raw hash mismatch")
    if metadata["oracle_upper_file_sha256"] != sha256_file(oracle_path):
        raise RuntimeError("Oracle file hash mismatch")

    threshold_d2 = float(metadata["threshold_d2"])
    epsilon = float(metadata["epsilon"])
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    counters = device_buffers["counters"]
    result_ids = device_buffers["result_ids"]
    ambiguous_ids = device_buffers["ambiguous_ids"]
    fp64_ids = device_buffers["fp64_ids"]
    counters.zero_()

    analytic_certificate_ragged_safe_i64[(scheduled_tiles,)](
        device_buffers["codes"],
        device_buffers["codes_t"],
        device_buffers["scales"],
        device_buffers["norms"],
        device_buffers["errors"],
        device_buffers["tile_rows"],
        device_buffers["tile_columns"],
        result_ids,
        ambiguous_ids,
        counters,
        float(epsilon_lower),
        float(epsilon_upper),
        EXPRESSION_RELATIVE_RADIUS,
        EXPRESSION_ABSOLUTE_RADIUS,
        SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
        SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
        M=n,
        N_=n,
        K=dimension,
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
    g3b_overflow = int(g3b_counts[2])
    if max(g3b_direct_count, g3b_ambiguous_count) > capacity or g3b_overflow:
        raise RuntimeError("G3B capacity overflow")
    g3b_direct = sorted_ids(result_ids, g3b_direct_count)
    g3b_ambiguous = sorted_ids(ambiguous_ids, g3b_ambiguous_count)

    if g3b_ambiguous_count:
        certified_fp32_filter_i64[(g3b_ambiguous_count,)](
            device_buffers["vectors"],
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
            N_=n,
            K=dimension,
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
    overflow = int(fp32_counts[2])
    if max(accepted_before_fp64_count, fp64_count) > capacity or overflow:
        raise RuntimeError("G3C capacity overflow")
    accepted_before_fp64 = sorted_ids(result_ids, accepted_before_fp64_count)
    fp32_accept = np.setdiff1d(accepted_before_fp64, g3b_direct)
    fp64_pairs = sorted_ids(fp64_ids, fp64_count)
    fp32_rejected = np.setdiff1d(
        g3b_ambiguous, np.union1d(fp32_accept, fp64_pairs)
    )

    if fp64_count:
        refine_ambiguous_fp64_i64[(fp64_count,)](
            device_buffers["vectors"],
            device_buffers["vectors"],
            fp64_ids,
            result_ids,
            counters,
            threshold_d2,
            N_=n,
            K=dimension,
            CAPACITY=capacity,
            BLOCK_K=FP64_BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
    final_counts = counters.cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    overflow = int(final_counts[2])
    if final_count > capacity:
        overflow += 1
    accepted_upper = sorted_ids(result_ids, min(final_count, capacity))

    g3b_covered = np.union1d(g3b_direct, g3b_ambiguous)
    unsafe_g3b_accepts = np.setdiff1d(g3b_direct, oracle)
    unsafe_g3b_rejects = np.setdiff1d(oracle, g3b_covered)
    unsafe_fp32_accepts = np.setdiff1d(fp32_accept, oracle)
    unsafe_fp32_rejects = np.intersect1d(fp32_rejected, oracle)
    missing = np.setdiff1d(oracle, accepted_upper)
    extra = np.setdiff1d(accepted_upper, oracle)
    rows = accepted_upper // np.uint64(n)
    columns = accepted_upper - rows * np.uint64(n)
    invalid_or_lower = int(
        np.count_nonzero(
            (accepted_upper >= np.uint64(n) * np.uint64(n)) | (rows > columns)
        )
    )
    duplicate_counts = {
        "g3b_direct": duplicate_count(g3b_direct),
        "g3b_ambiguous": duplicate_count(g3b_ambiguous),
        "fp32_accept": duplicate_count(fp32_accept),
        "fp32_reject": duplicate_count(fp32_rejected),
        "fp64": duplicate_count(fp64_pairs),
        "final": duplicate_count(accepted_upper),
    }
    total_upper = n * (n + 1) // 2
    g3b_reject_count = total_upper - g3b_direct_count - g3b_ambiguous_count
    stage_partitions_pass = bool(
        g3b_direct_count + g3b_ambiguous_count + g3b_reject_count == total_upper
        and fp32_accept_count + fp32_reject_count + fp64_count
        == g3b_ambiguous_count
        and fp32_accept_count == int(fp32_accept.size)
        and fp32_reject_count == int(fp32_rejected.size)
    )
    exact_match = bool(
        accepted_upper.size == oracle.size
        and sha256_u64(accepted_upper) == oracle_raw_sha256
        and np.array_equal(accepted_upper, oracle)
    )
    cell_gate_pass = bool(
        stage_partitions_pass
        and all(value == 0 for value in duplicate_counts.values())
        and unsafe_g3b_accepts.size == 0
        and unsafe_g3b_rejects.size == 0
        and unsafe_fp32_accepts.size == 0
        and unsafe_fp32_rejects.size == 0
        and missing.size == 0
        and extra.size == 0
        and invalid_or_lower == 0
        and overflow == 0
        and exact_match
    )
    return {
        "dataset_id": dataset,
        "n": n,
        "dimension": dimension,
        "target_average_directed_nonself_degree": target_degree,
        "actual_average_directed_nonself_degree": metadata[
            "actual_average_directed_nonself_degree"
        ],
        "threshold_d2": threshold_d2,
        "epsilon": epsilon,
        "threshold_d2_lower_float32": float(threshold_lower),
        "threshold_d2_upper_float32": float(threshold_upper),
        "epsilon_lower_float32": float(epsilon_lower),
        "epsilon_upper_float32": float(epsilon_upper),
        "vector_prefix_raw_sha256": vector_prefix_sha256,
        "source_row_prefix_raw_sha256": row_prefix_sha256,
        "oracle_metadata_sha256": sha256_file(metadata_path),
        "oracle_file_sha256": sha256_file(oracle_path),
        "oracle_upper_raw_u64_sha256": oracle_raw_sha256,
        "oracle_upper_count_including_self": int(oracle.size),
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
        "total_upper_pairs": total_upper,
        "g3b_direct_accept_upper_pairs": g3b_direct_count,
        "g3b_direct_reject_upper_pairs": g3b_reject_count,
        "g3b_ambiguous_upper_pairs": g3b_ambiguous_count,
        "g3b_ambiguity_fraction_of_all_upper_pairs": (
            g3b_ambiguous_count / total_upper
        ),
        "fp32_direct_accept_upper_pairs": fp32_accept_count,
        "fp32_direct_reject_upper_pairs": fp32_reject_count,
        "fp32_bitwise_equal_upper_pairs": bitwise_equal_count,
        "fp64_refined_upper_pairs": fp64_count,
        "fp64_refine_fraction_of_g3b_ambiguity": (
            fp64_count / g3b_ambiguous_count if g3b_ambiguous_count else 0.0
        ),
        "final_accepted_upper_pairs": final_count,
        "g3b_direct_raw_u64_sha256": sha256_u64(g3b_direct),
        "g3b_ambiguous_raw_u64_sha256": sha256_u64(g3b_ambiguous),
        "fp32_accept_raw_u64_sha256": sha256_u64(fp32_accept),
        "fp32_reject_raw_u64_sha256": sha256_u64(fp32_rejected),
        "fp64_raw_u64_sha256": sha256_u64(fp64_pairs),
        "accepted_upper_raw_u64_sha256": sha256_u64(accepted_upper),
        "duplicate_counts": duplicate_counts,
        "stage_partitions_pass": stage_partitions_pass,
        "unsafe_g3b_direct_accepts": int(unsafe_g3b_accepts.size),
        "unsafe_g3b_direct_rejects": int(unsafe_g3b_rejects.size),
        "unsafe_fp32_direct_accepts": int(unsafe_fp32_accepts.size),
        "unsafe_fp32_direct_rejects": int(unsafe_fp32_rejects.size),
        "missing_upper_pairs": int(missing.size),
        "extra_upper_pairs": int(extra.size),
        "invalid_or_lower_triangle_pairs": invalid_or_lower,
        "overflow_events": overflow,
        "first_unsafe_g3b_direct_accepts": unsafe_g3b_accepts[:16].tolist(),
        "first_unsafe_g3b_direct_rejects": unsafe_g3b_rejects[:16].tolist(),
        "first_unsafe_fp32_direct_accepts": unsafe_fp32_accepts[:16].tolist(),
        "first_unsafe_fp32_direct_rejects": unsafe_fp32_rejects[:16].tolist(),
        "first_missing_upper_pairs": missing[:16].tolist(),
        "first_extra_upper_pairs": extra[:16].tolist(),
        "exact_match": exact_match,
        "cell_gate_pass": cell_gate_pass,
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("G4B requires physical GPU1 as the sole visible device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g4b_r1_opportunity_{args.dataset}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    diagnostic_path = PROJECT / "results/g4b_ragged_tail_diagnostic.json"
    diagnostic_manifest_path = (
        PROJECT / "results/g4b_ragged_tail_diagnostic_manifest.json"
    )
    if sha256_file(diagnostic_path) != EXPECTED_DIAGNOSTIC_RESULT_SHA256:
        raise RuntimeError("Ragged-tail diagnostic result hash mismatch")
    if (
        sha256_file(diagnostic_manifest_path)
        != EXPECTED_DIAGNOSTIC_MANIFEST_SHA256
    ):
        raise RuntimeError("Ragged-tail diagnostic manifest hash mismatch")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_manifest = json.loads(
        diagnostic_manifest_path.read_text(encoding="utf-8")
    )
    if not diagnostic.get("diagnosis_pass") or not diagnostic_manifest.get(
        "admitted"
    ):
        raise RuntimeError("Ragged-tail correction has not passed its A/B gate")

    contract = DATASETS[args.dataset]
    dataset_root = DATA_ROOT / args.dataset
    vector_path = dataset_root / "vectors_f32.npy"
    row_path = dataset_root / "source_row_ids_u32.npy"
    if sha256_file(vector_path) != contract["vector_file_sha256"]:
        raise RuntimeError("Prepared vector-file hash mismatch")
    if sha256_file(row_path) != contract["row_file_sha256"]:
        raise RuntimeError("Prepared row-file hash mismatch")
    all_vectors = np.load(vector_path, allow_pickle=False)
    all_rows = np.load(row_path, allow_pickle=False)
    dimension = int(contract["dimension"])
    if all_vectors.shape != (max(SCALES), dimension) or all_vectors.dtype != np.float32:
        raise RuntimeError("Prepared vector shape/dtype mismatch")
    if all_rows.shape != (max(SCALES),) or all_rows.dtype != np.uint32:
        raise RuntimeError("Prepared row-index shape/dtype mismatch")
    if not all_vectors.flags.c_contiguous or not np.isfinite(all_vectors).all():
        raise RuntimeError("Prepared vectors are not finite contiguous float32")
    accumulator_limit = dimension * 127 * 127
    if accumulator_limit > min(2**24, np.iinfo(np.int32).max):
        raise RuntimeError("Exact INT32-to-FP32 accumulator precondition failed")

    started = time.perf_counter()
    cells: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats()
    for n in SCALES:
        vectors = np.ascontiguousarray(all_vectors[:n], dtype=np.float32)
        vector_prefix_sha256 = sha256_array(vectors, "<f4")
        row_prefix_sha256 = sha256_array(all_rows[:n], "<u4")
        codes, scales, norms, errors = quantize_with_analytic_residual(vectors)
        tile_extent = (n + BLOCK_M - 1) // BLOCK_M
        tile_rows, tile_columns = np.triu_indices(tile_extent)
        tile_rows = np.asarray(tile_rows, dtype=np.int32)
        tile_columns = np.asarray(tile_columns, dtype=np.int32)
        scheduled_tiles = int(tile_rows.size)
        capacity = scheduled_tiles * BLOCK_M * BLOCK_N
        device = torch.device("cuda:0")
        buffers = {
            "vectors": torch.from_numpy(vectors).to(device),
            "codes": torch.from_numpy(codes).to(device),
            "codes_t": torch.from_numpy(codes.T.copy()).to(device),
            "scales": torch.from_numpy(scales).to(device),
            "norms": torch.from_numpy(norms).to(device),
            "errors": torch.from_numpy(errors).to(device),
            "tile_rows": torch.from_numpy(tile_rows).to(device),
            "tile_columns": torch.from_numpy(tile_columns).to(device),
            "result_ids": torch.empty(capacity, dtype=torch.int64, device=device),
            "ambiguous_ids": torch.empty(capacity, dtype=torch.int64, device=device),
            "fp64_ids": torch.empty(capacity, dtype=torch.int64, device=device),
            "counters": torch.zeros(6, dtype=torch.int32, device=device),
        }
        for target_degree in TARGET_DEGREES:
            cell = validate_cell(
                dataset=args.dataset,
                n=n,
                dimension=dimension,
                target_degree=target_degree,
                vectors=vectors,
                vector_prefix_sha256=vector_prefix_sha256,
                row_prefix_sha256=row_prefix_sha256,
                dataset_root=dataset_root,
                device_buffers=buffers,
                scheduled_tiles=scheduled_tiles,
                capacity=capacity,
            )
            cells.append(cell)
            print("CELL_COMPLETE " + json.dumps(cell, sort_keys=True), flush=True)
            if not cell["cell_gate_pass"]:
                break
        del buffers
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if cells and not cells[-1]["cell_gate_pass"]:
            break

    process_gate_pass = bool(
        len(cells) == len(SCALES) * len(TARGET_DEGREES)
        and all(cell["cell_gate_pass"] for cell in cells)
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "correctness_and_routing_geometry_not_performance",
        "dataset_id": args.dataset,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "dimension": dimension,
        "scales": list(SCALES),
        "target_average_directed_nonself_degrees": list(TARGET_DEGREES),
        "prepared_vector_file_sha256": sha256_file(vector_path),
        "prepared_row_file_sha256": sha256_file(row_path),
        "accumulator_limit": accumulator_limit,
        "ragged_tail_diagnostic_result_sha256": sha256_file(diagnostic_path),
        "ragged_tail_diagnostic_manifest_sha256": sha256_file(
            diagnostic_manifest_path
        ),
        "g3b_analytic_constants": {
            "expression_relative_radius": EXPRESSION_RELATIVE_RADIUS,
            "expression_absolute_radius": EXPRESSION_ABSOLUTE_RADIUS,
            "sqrt_residual_final_relative_radius": (
                SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS
            ),
            "sqrt_residual_final_absolute_radius": (
                SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS
            ),
            "block_m": BLOCK_M,
            "block_n": BLOCK_N,
            "block_k": BLOCK_K,
        },
        "fp32_analytic_constants": {
            "distance_relative_radius": FP32_DISTANCE_RELATIVE_RADIUS,
            "input_magnitude_radius": FP32_INPUT_MAGNITUDE_RADIUS,
            "final_relative_radius": FP32_FINAL_RELATIVE_RADIUS,
            "absolute_radius": FP32_ABSOLUTE_RADIUS,
            "block_k": FP32_BLOCK_K,
        },
        "fp64_block_k": FP64_BLOCK_K,
        "cells": cells,
        "cell_count": len(cells),
        "process_gate_pass": process_gate_pass,
        "performance_claim_allowed": False,
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "diagnostic_wall_seconds": time.perf_counter() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4B_R1_OPPORTUNITY.md"),
        "ragged_safe_kernel_sha256": sha256_file(
            PROJECT / "src/g4b_r1_ragged_safe_kernel.py"
        ),
        "g3b_r1_runner_sha256": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
        "g3c_b_r1_runner_sha256": sha256_file(
            PROJECT / "src/run_g3c_b_r1_gpu_cascade.py"
        ),
        "fp64_refinement_source_sha256": sha256_file(
            PROJECT / "src/run_g2a_tensorjoin.py"
        ),
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if process_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
