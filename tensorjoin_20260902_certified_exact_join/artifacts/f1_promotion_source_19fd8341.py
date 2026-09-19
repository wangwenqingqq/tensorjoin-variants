#!/usr/bin/env python3
"""Run the frozen target-64 GPU precision-cascade smoke experiment."""

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

from run_d1_triton import (
    fused_int8_certificate_compact,
    quantize_per_vector,
    refine_ambiguous_fp64,
    squared_distances_float64,
    tie_aware_midgap_threshold,
    upward_float32,
)


EXPERIMENT_ID = "tensorjoin_20260903_gpu_cascade_f1"
QUERY_COUNT = 512
BASE_COUNT = 4096
TARGET_RESULTS_PER_QUERY = 64
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
CAPACITY = 65536
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
DIMENSIONS = {"audio": 2048, "video": 512, "hsi": 1984}
EXPECTED_FP64 = {"audio": 366, "video": 1061, "hsi": 928}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def audio_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    features, clips = data["features"], data["clip"]
    rng = np.random.default_rng(20260902)
    units = rng.permutation(np.unique(clips))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(clips == units[cursor]).tolist())
        cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(clips == units[cursor]).tolist())
        cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    overlap = np.intersect1d(np.unique(clips[query_ids]), np.unique(clips[base_ids]))
    if len(overlap):
        raise AssertionError("Audio clip overlap")
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
        {"clip_overlap": 0},
    )


def video_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    features, clips, groups = data["features"], data["clip"], data["group"]
    rng = np.random.default_rng(20260903)
    units = rng.permutation(np.unique(groups))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(groups == units[cursor]).tolist())
        cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(groups == units[cursor]).tolist())
        cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    group_overlap = len(np.intersect1d(groups[query_ids], groups[base_ids]))
    clip_overlap = len(np.intersect1d(clips[query_ids], clips[base_ids]))
    if group_overlap or clip_overlap:
        raise AssertionError("Video source overlap")
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
        {"group_overlap": 0, "clip_overlap": 0},
    )


def hsi_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    features, rows = data["features"], data["row"]
    rng = np.random.default_rng(20260903)
    query_ids = rng.choice(np.flatnonzero(rows <= 47), QUERY_COUNT, replace=False)
    base_ids = rng.choice(np.flatnonzero(rows >= 51), BASE_COUNT, replace=False)
    query_range = [int(rows[query_ids].min()), int(rows[query_ids].max())]
    base_range = [int(rows[base_ids].min()), int(rows[base_ids].max())]
    if query_range[1] + 2 >= base_range[0]:
        raise AssertionError("HSI source-pixel overlap")
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
        {"query_row_range": query_range, "base_row_range": base_range, "source_pixel_overlap": False},
    )


def load_split(path: Path, modality: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if modality == "audio":
            return audio_split(data)
        if modality == "video":
            return video_split(data)
        return hsi_split(data)


@triton.jit
def filter_ambiguous_fp32(
    query,
    base,
    ambiguous_ids,
    result_ids,
    fp64_ids,
    counters,
    threshold_d2,
    fp32_guard,
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
    pair_value = pair_id + counter_lane
    tl.store(result_ids + position, pair_value, mask=can_store)
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
    tl.store(fp64_ids + fp64_position, pair_value, mask=fp64_can_store)
    tl.atomic_add(
        counters + 2 + counter_lane,
        1,
        mask=needs_fp64 & ~fp64_can_store,
        sem="relaxed",
    )
    tl.atomic_add(
        counters + 4 + counter_lane, 1, mask=bitwise_equal, sem="relaxed"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=tuple(DIMENSIONS), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=("GTC", "GCT", "TGC", "TCG", "CGT", "CTG"),
        default="GTC",
        help="G=guarded FP32, T=two-stage, C=cascade",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--stress-launches", type=int, default=0)
    return parser.parse_args()


def time_callable(function: Callable[[], None], warmups: int, observations: int) -> list[float]:
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


def sorted_ids(result_ids: torch.Tensor, count: int) -> np.ndarray:
    return np.sort(result_ids[: min(count, CAPACITY)].cpu().numpy().astype(np.int64))


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("F1 requires physical GPU 0 as the sole visible device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite F1 evidence")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dimension = DIMENSIONS[args.modality]
    query, base, split_metadata = load_split(args.feature_cache, args.modality)
    if query.shape != (QUERY_COUNT, dimension) or base.shape != (BASE_COUNT, dimension):
        raise ValueError((query.shape, base.shape))

    exact_d2 = squared_distances_float64(query, base)
    threshold_info = tie_aware_midgap_threshold(exact_d2, TARGET_RESULTS_PER_QUERY)
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
    fp64_ids = torch.empty(CAPACITY, dtype=torch.int32, device=device)
    counters = torch.zeros(5, dtype=torch.int32, device=device)
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
            K=dimension,
            CAPACITY=CAPACITY,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

    def fp64_refine(ids: torch.Tensor, count: int) -> None:
        if count:
            refine_ambiguous_fp64[(count,)](
                q32,
                x32,
                ids,
                result_ids,
                counters,
                threshold,
                N=BASE_COUNT,
                K=dimension,
                CAPACITY=CAPACITY,
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
                num_stages=1,
            )

    def fp32_filter(count: int) -> None:
        if count:
            filter_ambiguous_fp32[(count,)](
                q32,
                x32,
                ambiguous_ids,
                result_ids,
                fp64_ids,
                counters,
                threshold,
                FP32_DISTANCE_GUARD,
                N=BASE_COUNT,
                K=dimension,
                CAPACITY=CAPACITY,
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
                num_stages=1,
            )

    def two_stage() -> None:
        counters.zero_()
        scan_compact()
        ambiguous_count = min(int(counters[1].item()), CAPACITY)
        fp64_refine(ambiguous_ids, ambiguous_count)

    def cascade() -> None:
        counters.zero_()
        scan_compact()
        ambiguous_count = min(int(counters[1].item()), CAPACITY)
        fp32_filter(ambiguous_count)
        fp64_count = min(int(counters[3].item()), CAPACITY)
        fp64_refine(fp64_ids, fp64_count)

    def guarded_fp32() -> None:
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
        torch.cat((torch.nonzero(accept.reshape(-1), as_tuple=False).reshape(-1), refined))

    # Stage-specific correctness accounting for the cascade.
    counters.zero_()
    scan_compact()
    torch.cuda.synchronize()
    int8_direct_count = int(counters[0].item())
    ambiguous_count = int(counters[1].item())
    overflow_after_scan = int(counters[2].item())
    int8_direct_ids = sorted_ids(result_ids, int8_direct_count)
    ambiguous_cpu = sorted_ids(ambiguous_ids, ambiguous_count)
    int8_false_accepts = int(np.count_nonzero(~np.isin(int8_direct_ids, oracle_ids)))

    fp32_filter(min(ambiguous_count, CAPACITY))
    torch.cuda.synchronize()
    after_fp32_count = int(counters[0].item())
    fp64_count = int(counters[3].item())
    equality_count = int(counters[4].item())
    overflow_after_fp32 = int(counters[2].item())
    fp32_accept_ids = np.sort(
        result_ids[int8_direct_count:min(after_fp32_count, CAPACITY)]
        .cpu().numpy().astype(np.int64)
    )
    fp64_cpu = sorted_ids(fp64_ids, fp64_count)
    fp32_reject_ids = np.setdiff1d(
        ambiguous_cpu,
        np.union1d(fp32_accept_ids, fp64_cpu),
        assume_unique=False,
    )
    fp32_false_accepts = int(np.count_nonzero(~np.isin(fp32_accept_ids, oracle_ids)))
    fp32_false_rejects = int(np.count_nonzero(np.isin(fp32_reject_ids, oracle_ids)))
    fp64_refine(fp64_ids, min(fp64_count, CAPACITY))
    torch.cuda.synchronize()
    cascade_count = int(counters[0].item())
    cascade_overflow = int(counters[2].item())
    cascade_ids = sorted_ids(result_ids, cascade_count)
    cascade_mismatch = int(len(np.setxor1d(oracle_ids, cascade_ids)))
    cascade_duplicates = int(len(cascade_ids) - len(np.unique(cascade_ids)))

    two_stage()
    torch.cuda.synchronize()
    two_stage_count = int(counters[0].item())
    two_stage_overflow = int(counters[2].item())
    two_stage_ids = sorted_ids(result_ids, two_stage_count)
    two_stage_mismatch = int(len(np.setxor1d(oracle_ids, two_stage_ids)))
    two_stage_duplicates = int(len(two_stage_ids) - len(np.unique(two_stage_ids)))

    guarded_fp32()
    torch.cuda.synchronize()
    # Recompute once for a retained output-ID oracle comparison.
    dots = torch.mm(q32, x32_t)
    d2 = torch.clamp_min(qnorm32[:, None] + xnorm32[None, :] - 2.0 * dots, 0.0)
    accept = d2 <= threshold - FP32_DISTANCE_GUARD
    reject = d2 > threshold + FP32_DISTANCE_GUARD
    guard_ambiguous = ~(accept | reject)
    guard_ids = torch.nonzero(guard_ambiguous.reshape(-1), as_tuple=False).reshape(-1)
    guard_rows = torch.div(guard_ids, BASE_COUNT, rounding_mode="floor")
    guard_columns = torch.remainder(guard_ids, BASE_COUNT)
    guard_delta = q64[guard_rows] - x64[guard_columns]
    guard_refined = guard_ids[torch.sum(guard_delta * guard_delta, dim=1) <= threshold]
    guard_output = torch.cat(
        (torch.nonzero(accept.reshape(-1), as_tuple=False).reshape(-1), guard_refined)
    )
    guarded_ids = np.sort(guard_output.cpu().numpy().astype(np.int64))
    guarded_mismatch = int(len(np.setxor1d(oracle_ids, guarded_ids)))

    correctness = {
        "oracle_pairs": int(len(oracle_ids)),
        "actual_results_per_query": len(oracle_ids) / QUERY_COUNT,
        "int8_direct_accept_pairs": int8_direct_count,
        "int8_ambiguous_pairs": ambiguous_count,
        "int8_false_accepts": int8_false_accepts,
        "fp32_direct_accept_pairs": int(len(fp32_accept_ids)),
        "fp32_direct_reject_pairs": int(len(fp32_reject_ids)),
        "fp32_false_accepts": fp32_false_accepts,
        "fp32_false_rejects": fp32_false_rejects,
        "bitwise_equal_pairs": equality_count,
        "fp64_refine_pairs": fp64_count,
        "expected_fp64_refine_pairs": EXPECTED_FP64[args.modality],
        "two_stage_mismatch": two_stage_mismatch,
        "two_stage_duplicates": two_stage_duplicates,
        "two_stage_overflow": two_stage_overflow,
        "cascade_mismatch": cascade_mismatch,
        "cascade_duplicates": cascade_duplicates,
        "cascade_overflow": cascade_overflow,
        "guarded_fp32_mismatch": guarded_mismatch,
        "overflow_after_scan": overflow_after_scan,
        "overflow_after_fp32": overflow_after_fp32,
        "ambiguous_id_sha256": hashlib.sha256(ambiguous_cpu.tobytes()).hexdigest(),
        "fp64_id_sha256": hashlib.sha256(fp64_cpu.tobytes()).hexdigest(),
        "cascade_id_sha256": hashlib.sha256(cascade_ids.tobytes()).hexdigest(),
    }
    correctness_pass = all(
        value == 0
        for key, value in correctness.items()
        if "mismatch" in key
        or "duplicates" in key
        or "overflow" in key
        or key in ("int8_false_accepts", "fp32_false_accepts", "fp32_false_rejects")
    ) and fp64_count == EXPECTED_FP64[args.modality]
    print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)

    stress_hashes: set[str] = set()
    stress_pass = True
    for _ in range(args.stress_launches):
        cascade()
        torch.cuda.synchronize()
        count = int(counters[0].item())
        values = sorted_ids(result_ids, count)
        stress_hashes.add(hashlib.sha256(values.tobytes()).hexdigest())
        if count != len(oracle_ids) or len(np.setxor1d(oracle_ids, values)) != 0:
            stress_pass = False
            break
    if args.stress_launches:
        print(
            "STRESS "
            + json.dumps(
                {
                    "launches_requested": args.stress_launches,
                    "hashes": sorted(stress_hashes),
                    "stress_pass": stress_pass,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    functions = {"G": guarded_fp32, "T": two_stage, "C": cascade}
    names = {"G": "guarded_fp32", "T": "two_stage", "C": "cascade"}
    metrics: dict[str, list[float]] = {}
    if not args.validation_only:
        for key in args.order:
            name = names[key]
            metrics[name] = time_callable(functions[key], args.warmups, args.observations)
            print(
                f"TIMING metric={name} median_us={np.median(metrics[name]):.6f} "
                f"p10_us={np.percentile(metrics[name], 10):.6f} "
                f"p90_us={np.percentile(metrics[name], 90):.6f}",
                flush=True,
            )

    medians = {name: float(np.median(values)) for name, values in metrics.items()}
    ratios = (
        {
            "two_stage_over_cascade": medians["two_stage"] / medians["cascade"],
            "guarded_fp32_over_cascade": medians["guarded_fp32"] / medians["cascade"],
        }
        if metrics
        else {}
    )
    required_guarded_ratio = 1.50 if args.modality == "audio" else 1.35
    smoke_gate_pass = (
        bool(
            correctness_pass
            and stress_pass
            and ratios["two_stage_over_cascade"] >= 1.15
            and ratios["guarded_fp32_over_cascade"] >= required_guarded_ratio
        )
        if metrics
        else None
    )
    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": EXPERIMENT_ID,
        "modality": args.modality,
        "process_id": args.process_id,
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [QUERY_COUNT, BASE_COUNT, dimension],
        "target_results_per_query": TARGET_RESULTS_PER_QUERY,
        "threshold_calibration": threshold_info,
        "bound_pad": BOUND_PAD,
        "fp32_distance_guard": FP32_DISTANCE_GUARD,
        "capacity": CAPACITY,
        "order": args.order,
        "warmups": args.warmups,
        "observations": args.observations,
        "validation_only": args.validation_only,
        "stress_launches": args.stress_launches,
        "stress_hashes": sorted(stress_hashes),
        "stress_pass": stress_pass,
        "split_metadata": split_metadata,
        "correctness": correctness,
        "correctness_pass": correctness_pass,
        "metrics_us": metrics,
        "medians_us": medians,
        "ratios": ratios,
        "required_guarded_ratio": required_guarded_ratio,
        "smoke_gate_pass": smoke_gate_pass,
        "script_sha256": sha256_file(Path(__file__)),
        "stage1_source_sha256": sha256_file(Path(__file__).with_name("run_d1_triton.py")),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(record, sort_keys=True), flush=True)
    return 0 if correctness_pass and stress_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
