#!/usr/bin/env python3
"""Run one fresh-process PyTorch GPU B0 observation block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


EXPERIMENT_ID = "tensorjoin_20260902_pytorch_gpu_exact_join_b0"
SEED = 20260902
QUERY_COUNT = 512
BASE_COUNT = 4096
DIMENSION = 1024
TARGET_RESULTS_PER_QUERY = (1, 8, 64)
BOUND_PAD = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--order", choices=("AB", "BA"), required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--observations", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def disjoint_clip_sample(
    features: np.ndarray, clips: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    unique_clips = np.unique(clips)
    permuted_clips = rng.permutation(unique_clips)
    query_indices: list[int] = []
    base_indices: list[int] = []
    cursor = 0
    while len(query_indices) < QUERY_COUNT:
        query_indices.extend(np.flatnonzero(clips == permuted_clips[cursor]).tolist())
        cursor += 1
    while len(base_indices) < BASE_COUNT:
        base_indices.extend(np.flatnonzero(clips == permuted_clips[cursor]).tolist())
        cursor += 1
    query_indices = rng.permutation(query_indices)[:QUERY_COUNT]
    base_indices = rng.permutation(base_indices)[:BASE_COUNT]
    if len(np.intersect1d(np.unique(clips[query_indices]), np.unique(clips[base_indices]))):
        raise AssertionError("Query/base clip leakage")
    return (
        np.ascontiguousarray(features[query_indices]),
        np.ascontiguousarray(features[base_indices]),
    )


def squared_distances_float64(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    distances = (
        np.sum(q * q, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (q @ x.T)
    )
    np.maximum(distances, 0.0, out=distances)
    return distances


def midpoint_thresholds(exact_d2: np.ndarray) -> list[tuple[int, float]]:
    flat = exact_d2.reshape(-1)
    thresholds: list[tuple[int, float]] = []
    for target in TARGET_RESULTS_PER_QUERY:
        count = exact_d2.shape[0] * target
        values = np.partition(flat, (count - 1, count))
        lower = float(values[count - 1])
        upper = float(values[count])
        if not lower < upper:
            raise ValueError(f"Cannot place strict midpoint for target {target}: {lower}, {upper}")
        thresholds.append((target, lower + (upper - lower) / 2.0))
    return thresholds


def quantize_per_vector(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = (codes.astype(np.float32) * scales[:, None]).astype(np.float32)
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    errors = np.nextafter(np.sqrt(np.sum(residual * residual, axis=1)), np.inf)
    return codes, scales, errors


def time_callable(
    function: Callable[[], torch.Tensor], warmups: int, observations: int
) -> list[float]:
    for _ in range(warmups):
        result = function()
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(observations):
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0))
    if result.numel() < 0:
        raise AssertionError("Unreachable result guard")
    return samples


def sorted_cpu_ids(ids: torch.Tensor) -> np.ndarray:
    values = ids.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.sort(values)


def nvidia_smi_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,pstate,power.draw,power.limit,"
        "clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ]
    return subprocess.check_output(command, text=True).strip()


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("B0 requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with np.load(args.feature_cache, allow_pickle=False) as data:
        features = data["features"]
        clips = data["clip"]
    query, base = disjoint_clip_sample(features, clips)
    if query.shape != (QUERY_COUNT, DIMENSION) or base.shape != (BASE_COUNT, DIMENSION):
        raise AssertionError((query.shape, base.shape))
    exact_d2 = squared_distances_float64(query, base)
    thresholds = midpoint_thresholds(exact_d2)
    q_codes, q_scales, q_errors = quantize_per_vector(query)
    x_codes, x_scales, x_errors = quantize_per_vector(base)

    device = torch.device("cuda:0")
    q32 = torch.from_numpy(query).to(device)
    x32 = torch.from_numpy(base).to(device)
    q64 = q32.to(torch.float64)
    x64 = x32.to(torch.float64)
    x32_t = x32.T.contiguous()
    x64_t = x64.T.contiguous()
    qi8 = torch.from_numpy(q_codes).to(device)
    xi8_t = torch.from_numpy(x_codes.T.copy()).to(device)
    qscale64 = torch.from_numpy(q_scales.astype(np.float64)).to(device)
    xscale64 = torch.from_numpy(x_scales.astype(np.float64)).to(device)
    qerror64 = torch.from_numpy(q_errors).to(device)
    xerror64 = torch.from_numpy(x_errors).to(device)
    qhat_norm2 = torch.from_numpy(
        np.sum(q_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * q_scales.astype(np.float64) ** 2
    ).to(device)
    xhat_norm2 = torch.from_numpy(
        np.sum(x_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * x_scales.astype(np.float64) ** 2
    ).to(device)
    qnorm64 = torch.sum(q64 * q64, dim=1)
    xnorm64 = torch.sum(x64 * x64, dim=1)

    def dense_int8() -> torch.Tensor:
        return torch._int_mm(qi8, xi8_t)

    def dense_fp32() -> torch.Tensor:
        return torch.mm(q32, x32_t)

    def dense_fp64() -> torch.Tensor:
        return torch.mm(q64, x64_t)

    def baseline_ids(threshold: float) -> torch.Tensor:
        dots = torch.mm(q64, x64_t)
        d2 = qnorm64[:, None] + xnorm64[None, :] - 2.0 * dots
        d2 = torch.clamp_min(d2, 0.0)
        return torch.nonzero(d2.reshape(-1) <= threshold, as_tuple=False).reshape(-1)

    def candidate_components(threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        int_dots = torch._int_mm(qi8, xi8_t)
        scale_products = qscale64[:, None] * xscale64[None, :]
        dhat2 = (
            qhat_norm2[:, None]
            + xhat_norm2[None, :]
            - 2.0 * int_dots.to(torch.float64) * scale_products
        )
        dhat = torch.sqrt(torch.clamp_min(dhat2, 0.0))
        radius = qerror64[:, None] + xerror64[None, :] + BOUND_PAD
        lower = torch.clamp_min(dhat - radius, 0.0) ** 2
        upper = (dhat + radius) ** 2
        accept = upper <= threshold
        reject = lower > threshold
        ambiguous = ~(accept | reject)
        return accept, reject, ambiguous

    def candidate_scan_bounds(threshold: float) -> torch.Tensor:
        _, _, ambiguous = candidate_components(threshold)
        return ambiguous

    def candidate_ids(threshold: float) -> torch.Tensor:
        accept, _, ambiguous = candidate_components(threshold)
        ambiguous_ids = torch.nonzero(ambiguous.reshape(-1), as_tuple=False).reshape(-1)
        rows = torch.div(ambiguous_ids, BASE_COUNT, rounding_mode="floor")
        columns = torch.remainder(ambiguous_ids, BASE_COUNT)
        delta = q64[rows] - x64[columns]
        refined_d2 = torch.sum(delta * delta, dim=1)
        refined_ids = ambiguous_ids[refined_d2 <= threshold]
        accepted_ids = torch.nonzero(accept.reshape(-1), as_tuple=False).reshape(-1)
        return torch.cat((accepted_ids, refined_ids))

    # Force library initialization before correctness and timing.
    _ = dense_int8()
    _ = dense_fp32()
    _ = dense_fp64()
    torch.cuda.synchronize()

    int_dot_gpu = dense_int8().cpu().numpy()
    int_dot_cpu = q_codes.astype(np.int32) @ x_codes.astype(np.int32).T
    int_dot_mismatch = int(np.count_nonzero(int_dot_gpu != int_dot_cpu))
    correctness: dict[str, object] = {"int_dot_mismatch": int_dot_mismatch, "thresholds": {}}
    for target, threshold in thresholds:
        oracle_ids = np.flatnonzero(exact_d2.reshape(-1) <= threshold).astype(np.int64)
        keeper_ids = sorted_cpu_ids(baseline_ids(threshold))
        candidate_result_ids = sorted_cpu_ids(candidate_ids(threshold))
        _, _, ambiguous = candidate_components(threshold)
        ambiguous_count = int(torch.count_nonzero(ambiguous).item())
        target_result = {
            "threshold_d2": threshold,
            "oracle_pairs": int(len(oracle_ids)),
            "ambiguous_pairs": ambiguous_count,
            "ambiguous_fraction": ambiguous_count / exact_d2.size,
            "keeper_mismatch": int(
                len(np.setxor1d(oracle_ids, keeper_ids, assume_unique=True))
            ),
            "candidate_mismatch": int(
                len(np.setxor1d(oracle_ids, candidate_result_ids, assume_unique=True))
            ),
            "candidate_duplicate_ids": int(
                len(candidate_result_ids) - len(np.unique(candidate_result_ids))
            ),
        }
        correctness["thresholds"][str(target)] = target_result
        print("CORRECTNESS " + json.dumps({"target": target, **target_result}, sort_keys=True), flush=True)

    metrics: dict[str, list[float]] = {}
    dense_order = (
        (("dense_fp32", dense_fp32), ("dense_int8", dense_int8), ("dense_fp64", dense_fp64))
        if args.order == "AB"
        else (("dense_int8", dense_int8), ("dense_fp32", dense_fp32), ("dense_fp64", dense_fp64))
    )
    for name, function in dense_order:
        metrics[name] = time_callable(function, args.warmups, args.observations)
        print(
            f"TIMING metric={name} median_us={np.median(metrics[name]):.6f} "
            f"p10_us={np.percentile(metrics[name],10):.6f} p90_us={np.percentile(metrics[name],90):.6f}",
            flush=True,
        )
    for target, threshold in thresholds:
        functions = (
            (
                (f"baseline_e2e_s{target}", lambda t=threshold: baseline_ids(t)),
                (f"candidate_e2e_s{target}", lambda t=threshold: candidate_ids(t)),
            )
            if args.order == "AB"
            else (
                (f"candidate_e2e_s{target}", lambda t=threshold: candidate_ids(t)),
                (f"baseline_e2e_s{target}", lambda t=threshold: baseline_ids(t)),
            )
        )
        for name, function in functions:
            metrics[name] = time_callable(function, args.warmups, args.observations)
            print(
                f"TIMING metric={name} median_us={np.median(metrics[name]):.6f} "
                f"p10_us={np.percentile(metrics[name],10):.6f} p90_us={np.percentile(metrics[name],90):.6f}",
                flush=True,
            )
        scan_name = f"candidate_scan_bounds_s{target}"
        metrics[scan_name] = time_callable(
            lambda t=threshold: candidate_scan_bounds(t), args.warmups, args.observations
        )
        print(
            f"TIMING metric={scan_name} median_us={np.median(metrics[scan_name]):.6f}",
            flush=True,
        )

    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": EXPERIMENT_ID,
        "process_id": args.process_id,
        "order": args.order,
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_sm_count": properties.multi_processor_count,
        "gpu_total_memory": properties.total_memory,
        "nvidia_smi": nvidia_smi_snapshot(),
        "feature_cache": str(args.feature_cache),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "script_sha256": sha256_file(Path(__file__)),
        "shape": [QUERY_COUNT, BASE_COUNT, DIMENSION],
        "warmups": args.warmups,
        "observations": args.observations,
        "thresholds": {str(target): threshold for target, threshold in thresholds},
        "correctness": correctness,
        "metrics_us": metrics,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    correctness_pass = int_dot_mismatch == 0 and all(
        row["keeper_mismatch"] == 0
        and row["candidate_mismatch"] == 0
        and row["candidate_duplicate_ids"] == 0
        for row in correctness["thresholds"].values()
    )
    print(
        "PROCESS_SUMMARY "
        + json.dumps(
            {
                "process_id": args.process_id,
                "order": args.order,
                "correctness_pass": correctness_pass,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if correctness_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())

