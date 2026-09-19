#!/usr/bin/env python3
"""Run one D3B process on standardized Indian Pines patch tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import triton
import triton.language as tl


EXPERIMENT_ID = "tensorjoin_20260903_indian_pines_triton_d3b"
SEED = 20260903
QUERY_COUNT = 512
BASE_COUNT = 4096
DIMENSION = 1984
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
CAPACITY = 65536
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def spatially_disjoint_sample(
    features: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    query_pool = np.flatnonzero(rows <= 47)
    base_pool = np.flatnonzero(rows >= 51)
    query_indices = rng.choice(query_pool, size=QUERY_COUNT, replace=False)
    base_indices = rng.choice(base_pool, size=BASE_COUNT, replace=False)
    query_coords = np.stack((rows[query_indices], columns[query_indices]), axis=1)
    base_coords = np.stack((rows[base_indices], columns[base_indices]), axis=1)
    if int(query_coords[:, 0].max()) + 2 >= int(base_coords[:, 0].min()):
        raise AssertionError("Query/base source-pixel overlap")
    return (
        np.ascontiguousarray(features[query_indices], dtype=np.float32),
        np.ascontiguousarray(features[base_indices], dtype=np.float32),
        query_coords,
        base_coords,
    )


def squared_distances_float64(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    distances = np.empty((len(q), len(x)), dtype=np.float64)
    for start in range(0, len(q), 4):
        stop = min(start + 4, len(q))
        difference = q[start:stop, None, :] - x[None, :, :]
        distances[start:stop] = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )
    return distances


def quantize_per_vector(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = (codes.astype(np.float32) * scales[:, None]).astype(np.float32)
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    errors = np.nextafter(np.linalg.norm(residual, axis=1), np.inf)
    return codes, scales, errors


def tie_aware_midgap_threshold(exact_d2: np.ndarray, target: int) -> dict[str, float]:
    flat = exact_d2.reshape(-1)
    rank = QUERY_COUNT * target - 1
    lower = float(np.partition(flat, rank)[rank])
    greater = flat[flat > lower]
    if not len(greater):
        raise ValueError("No strictly greater distance for a mid-gap threshold")
    upper = float(np.min(greater))
    threshold = lower + (upper - lower) / 2.0
    return {
        "lower_distance_d2": lower,
        "upper_distance_d2": upper,
        "gap_d2": upper - lower,
        "threshold_d2": threshold,
    }


@triton.jit
def fused_int8_certificate_compact(
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
    N: tl.constexpr,
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
    base_ptrs = base_codes_transposed + offsets_k[:, None] * N + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        query_block = tl.load(
            query_ptrs,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        base_block = tl.load(
            base_ptrs,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0,
        )
        accumulator = tl.dot(query_block, base_block, accumulator, out_dtype=tl.int32)
        query_ptrs += BLOCK_K
        base_ptrs += BLOCK_K * N

    query_scale = tl.load(query_scales + offsets_m, mask=offsets_m < M, other=0.0)
    base_scale = tl.load(base_scales + offsets_n, mask=offsets_n < N, other=0.0)
    query_norm2 = tl.load(
        query_reconstructed_norm2 + offsets_m, mask=offsets_m < M, other=0.0
    )
    base_norm2 = tl.load(
        base_reconstructed_norm2 + offsets_n, mask=offsets_n < N, other=0.0
    )
    query_error = tl.load(query_errors + offsets_m, mask=offsets_m < M, other=0.0)
    base_error = tl.load(base_errors + offsets_n, mask=offsets_n < N, other=0.0)
    dot = accumulator.to(tl.float32) * query_scale[:, None] * base_scale[None, :]
    reconstructed_d2 = query_norm2[:, None] + base_norm2[None, :] - 2.0 * dot
    reconstructed_d = tl.sqrt(tl.maximum(reconstructed_d2, 0.0))
    radius = query_error[:, None] + base_error[None, :] + bound_pad
    lower = tl.maximum(reconstructed_d - radius, 0.0)
    upper = reconstructed_d + radius
    lower_d2 = lower * lower
    upper_d2 = upper * upper
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    accept = output_mask & (upper_d2 <= threshold_d2)
    reject = output_mask & (lower_d2 > threshold_d2)
    ambiguous = output_mask & ~(accept | reject)
    pair_ids = (offsets_m[:, None] * N + offsets_n[None, :]).to(tl.int32)

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
def refine_ambiguous_fp64(
    query,
    base,
    ambiguous_ids,
    result_ids,
    counters,
    threshold_d2,
    N: tl.constexpr,
    K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_index = tl.program_id(axis=0)
    pair_id = tl.load(ambiguous_ids + pair_index)
    query_row = pair_id // N
    base_row = pair_id - query_row * N
    offsets_k = tl.arange(0, BLOCK_K)
    distance_d2 = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, K, BLOCK_K):
        k = block_start + offsets_k
        query_values = tl.load(
            query + query_row * K + k, mask=k < K, other=0.0
        ).to(tl.float64)
        base_values = tl.load(
            base + base_row * K + k, mask=k < K, other=0.0
        ).to(tl.float64)
        delta = query_values - base_values
        distance_d2 += tl.sum(delta * delta, axis=0)
    inside = distance_d2 <= threshold_d2
    counter_lane = tl.zeros((1,), dtype=tl.int32)
    position = tl.atomic_add(
        counters + counter_lane, 1, mask=inside, sem="relaxed"
    )
    can_store = inside & (position < CAPACITY)
    pair_value = pair_id + counter_lane
    tl.store(result_ids + position, pair_value, mask=can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=inside & ~can_store,
        sem="relaxed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--order", choices=("AB", "BA"), default="AB")
    parser.add_argument("--target-results-per-query", type=int, choices=(1, 64), required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--stress-launches", type=int, default=0)
    return parser.parse_args()


def upward_float32(values: np.ndarray) -> np.ndarray:
    rounded = values.astype(np.float32)
    increment = rounded.astype(np.float64) < values.astype(np.float64)
    rounded[increment] = np.nextafter(rounded[increment], np.float32(np.inf))
    return rounded


def time_callable(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmups: int, observations: int
) -> list[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(observations):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0))
    return samples


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("D3B requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.feature_cache, allow_pickle=False) as data:
        query, base, query_coords, base_coords = spatially_disjoint_sample(
            data["features"], data["row"], data["col"]
        )
    if query.shape != (QUERY_COUNT, DIMENSION) or base.shape != (BASE_COUNT, DIMENSION):
        raise ValueError((query.shape, base.shape))
    exact_d2 = squared_distances_float64(query, base)
    threshold_info = tie_aware_midgap_threshold(exact_d2, args.target_results_per_query)
    threshold = threshold_info["threshold_d2"]
    oracle_ids = np.flatnonzero(exact_d2.reshape(-1) <= threshold).astype(np.int64)
    q_codes, q_scales, q_errors = quantize_per_vector(query)
    x_codes, x_scales, x_errors = quantize_per_vector(base)
    q_hat_norm2 = (
        np.sum(q_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * q_scales.astype(np.float64) ** 2
    )
    x_hat_norm2 = (
        np.sum(x_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * x_scales.astype(np.float64) ** 2
    )

    device = torch.device("cuda:0")
    q32 = torch.from_numpy(query).to(device)
    x32 = torch.from_numpy(base).to(device)
    q64 = q32.to(torch.float64)
    x64 = x32.to(torch.float64)
    x32_t = x32.T.contiguous()
    qnorm32 = torch.sum(q32 * q32, dim=1)
    xnorm32 = torch.sum(x32 * x32, dim=1)
    q_codes_gpu = torch.from_numpy(q_codes).to(device)
    x_codes_t_gpu = torch.from_numpy(x_codes.T.copy()).to(device)
    q_scales_gpu = torch.from_numpy(q_scales).to(device)
    x_scales_gpu = torch.from_numpy(x_scales).to(device)
    q_norm_gpu = torch.from_numpy(q_hat_norm2.astype(np.float32)).to(device)
    x_norm_gpu = torch.from_numpy(x_hat_norm2.astype(np.float32)).to(device)
    q_error_gpu = torch.from_numpy(upward_float32(q_errors)).to(device)
    x_error_gpu = torch.from_numpy(upward_float32(x_errors)).to(device)
    result_ids = torch.empty(CAPACITY, dtype=torch.int32, device=device)
    ambiguous_ids = torch.empty(CAPACITY, dtype=torch.int32, device=device)
    counters = torch.zeros(3, dtype=torch.int32, device=device)
    scan_grid = (triton.cdiv(QUERY_COUNT, BLOCK_M), triton.cdiv(BASE_COUNT, BLOCK_N))

    def scan_compact() -> None:
        fused_int8_certificate_compact[scan_grid](
            q_codes_gpu,
            x_codes_t_gpu,
            q_scales_gpu,
            x_scales_gpu,
            q_norm_gpu,
            x_norm_gpu,
            q_error_gpu,
            x_error_gpu,
            result_ids,
            ambiguous_ids,
            counters,
            threshold,
            BOUND_PAD,
            M=QUERY_COUNT,
            N=BASE_COUNT,
            K=DIMENSION,
            CAPACITY=CAPACITY,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

    def refine(count: int) -> None:
        if count:
            refine_ambiguous_fp64[(count,)](
                q32,
                x32,
                ambiguous_ids,
                result_ids,
                counters,
                threshold,
                N=BASE_COUNT,
                K=DIMENSION,
                CAPACITY=CAPACITY,
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
                num_stages=1,
            )

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        counters.zero_()
        scan_compact()
        ambiguous_count = min(int(counters[1].item()), CAPACITY)
        refine(ambiguous_count)
        return result_ids, counters

    def keeper() -> tuple[torch.Tensor, torch.Tensor]:
        dots = torch.mm(q32, x32_t)
        d2 = torch.clamp_min(qnorm32[:, None] + xnorm32[None, :] - 2.0 * dots, 0.0)
        accept = d2 <= threshold - FP32_DISTANCE_GUARD
        reject = d2 > threshold + FP32_DISTANCE_GUARD
        ambiguous = ~(accept | reject)
        ids = torch.nonzero(ambiguous.reshape(-1), as_tuple=False).reshape(-1)
        rows = torch.div(ids, BASE_COUNT, rounding_mode="floor")
        columns = torch.remainder(ids, BASE_COUNT)
        delta = q64[rows] - x64[columns]
        refined = ids[torch.sum(delta * delta, dim=1) <= threshold]
        accepted = torch.nonzero(accept.reshape(-1), as_tuple=False).reshape(-1)
        return torch.cat((accepted, refined)), counters

    # Compile and obtain direct-accept/ambiguity state before refinement.
    counters.zero_()
    scan_compact()
    torch.cuda.synchronize()
    direct_count = int(counters[0].item())
    ambiguous_count = int(counters[1].item())
    overflow_before_refine = int(counters[2].item())
    direct_ids = np.sort(result_ids[: min(direct_count, CAPACITY)].cpu().numpy().astype(np.int64))
    ambiguous_cpu = np.sort(
        ambiguous_ids[: min(ambiguous_count, CAPACITY)].cpu().numpy().astype(np.int64)
    )
    direct_false_accept = int(np.count_nonzero(~np.isin(direct_ids, oracle_ids)))
    refine(min(ambiguous_count, CAPACITY))
    torch.cuda.synchronize()
    final_count = int(counters[0].item())
    overflow_final = int(counters[2].item())
    final_ids = np.sort(result_ids[: min(final_count, CAPACITY)].cpu().numpy().astype(np.int64))
    final_mismatch = int(len(np.setxor1d(oracle_ids, final_ids)))
    duplicates = int(len(final_ids) - len(np.unique(final_ids)))
    correctness = {
        "oracle_pairs": int(len(oracle_ids)),
        "direct_accept_pairs": direct_count,
        "ambiguous_pairs": ambiguous_count,
        "ambiguous_fraction": ambiguous_count / exact_d2.size,
        "direct_false_accepts": direct_false_accept,
        "overflow_before_refine": overflow_before_refine,
        "overflow_final": overflow_final,
        "final_pairs": final_count,
        "final_mismatch": final_mismatch,
        "duplicates": duplicates,
        "ambiguous_id_sha256": hashlib.sha256(ambiguous_cpu.tobytes()).hexdigest(),
        "final_id_sha256": hashlib.sha256(final_ids.tobytes()).hexdigest(),
    }
    keeper_ids = np.sort(keeper()[0].cpu().numpy().astype(np.int64))
    correctness["keeper_mismatch"] = int(len(np.setxor1d(oracle_ids, keeper_ids)))
    print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)

    stress_pass = True
    stress_hashes: set[str] = set()
    for _ in range(args.stress_launches):
        candidate()
        torch.cuda.synchronize()
        count = int(counters[0].item())
        values = np.sort(result_ids[: min(count, CAPACITY)].cpu().numpy().astype(np.int64))
        stress_hashes.add(hashlib.sha256(values.tobytes()).hexdigest())
        if count != len(oracle_ids) or len(np.setxor1d(oracle_ids, values)) != 0:
            stress_pass = False
            break

    metrics: dict[str, list[float]] = {}
    if not args.validation_only:
        variants = (("baseline", keeper), ("candidate", candidate))
        if args.order == "BA":
            variants = tuple(reversed(variants))
        for name, function in variants:
            metrics[name] = time_callable(function, args.warmups, args.observations)
            print(
                f"TIMING metric={name} median_us={np.median(metrics[name]):.6f} "
                f"p10_us={np.percentile(metrics[name],10):.6f} "
                f"p90_us={np.percentile(metrics[name],90):.6f}",
                flush=True,
            )

    correctness_pass = (
        direct_false_accept == 0
        and overflow_before_refine == 0
        and overflow_final == 0
        and final_mismatch == 0
        and duplicates == 0
        and correctness["keeper_mismatch"] == 0
        and ambiguous_count <= CAPACITY
        and stress_pass
    )
    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": EXPERIMENT_ID,
        "process_id": args.process_id,
        "order": args.order,
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [QUERY_COUNT, BASE_COUNT, DIMENSION],
        "query_row_range": [int(query_coords[:, 0].min()), int(query_coords[:, 0].max())],
        "base_row_range": [int(base_coords[:, 0].min()), int(base_coords[:, 0].max())],
        "source_pixel_overlap": False,
        "target_results_per_query": args.target_results_per_query,
        "actual_results_per_query": len(oracle_ids) / QUERY_COUNT,
        "capacity": CAPACITY,
        "threshold_d2": threshold,
        "threshold_calibration": threshold_info,
        "bound_pad": BOUND_PAD,
        "fp32_distance_guard": FP32_DISTANCE_GUARD,
        "warmups": args.warmups,
        "observations": args.observations,
        "validation_only": args.validation_only,
        "stress_launches": args.stress_launches,
        "stress_pass": stress_pass,
        "stress_hashes": sorted(stress_hashes),
        "correctness": correctness,
        "correctness_pass": correctness_pass,
        "metrics_us": metrics,
        "script_sha256": sha256_file(Path(__file__)),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PROCESS_SUMMARY "
        + json.dumps(
            {
                "correctness_pass": correctness_pass,
                "process_id": args.process_id,
                "order": args.order,
                "target_results_per_query": args.target_results_per_query,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if correctness_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
