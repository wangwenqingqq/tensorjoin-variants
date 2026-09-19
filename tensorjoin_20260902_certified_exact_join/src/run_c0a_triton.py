#!/usr/bin/env python3
"""Compile, validate, and time the frozen C0A fused Triton status kernel."""

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

from run_b0_gpu import (
    BASE_COUNT,
    DIMENSION,
    QUERY_COUNT,
    disjoint_clip_sample,
    midpoint_thresholds,
    quantize_per_vector,
    sha256_file,
    squared_distances_float64,
)


EXPERIMENT_ID = "tensorjoin_20260902_triton_fused_status_c0a"
TARGET_RESULTS_PER_QUERY = 1
BOUND_PAD = 1e-4
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def fused_int8_certificate_status(
    query_codes,
    base_codes_transposed,
    query_scales,
    base_scales,
    query_reconstructed_norm2,
    base_reconstructed_norm2,
    query_errors,
    base_errors,
    status_output,
    threshold_d2,
    bound_pad,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    lower_d2 = lower * lower
    upper = reconstructed_d + radius
    upper_d2 = upper * upper
    status = tl.where(upper_d2 <= threshold_d2, 1, tl.where(lower_d2 > threshold_d2, 0, 2))
    output_offsets = offsets_m[:, None] * N + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(status_output + output_offsets, status.to(tl.uint8), mask=output_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--observations", type=int, default=200)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def upward_float32(values: np.ndarray) -> np.ndarray:
    rounded = values.astype(np.float32)
    needs_increment = rounded.astype(np.float64) < values.astype(np.float64)
    rounded[needs_increment] = np.nextafter(rounded[needs_increment], np.float32(np.inf))
    return rounded


def cache_artifacts(cache_dir: Path | None) -> list[dict[str, object]]:
    if cache_dir is None or not cache_dir.exists():
        return []
    artifacts = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return artifacts


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("C0A requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    with np.load(args.feature_cache, allow_pickle=False) as data:
        query, base = disjoint_clip_sample(data["features"], data["clip"])
    exact_d2 = squared_distances_float64(query, base)
    threshold = dict(midpoint_thresholds(exact_d2))[TARGET_RESULTS_PER_QUERY]
    exact_inside = exact_d2 <= threshold
    q_codes, q_scales, q_errors = quantize_per_vector(query)
    x_codes, x_scales, x_errors = quantize_per_vector(base)
    q_norm2 = (
        np.sum(q_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * q_scales.astype(np.float64) ** 2
    )
    x_norm2 = (
        np.sum(x_codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * x_scales.astype(np.float64) ** 2
    )

    device = torch.device("cuda:0")
    tensors = {
        "q_codes": torch.from_numpy(q_codes).to(device),
        "x_codes_t": torch.from_numpy(x_codes.T.copy()).to(device),
        "q_scales": torch.from_numpy(q_scales).to(device),
        "x_scales": torch.from_numpy(x_scales).to(device),
        "q_norm2": torch.from_numpy(q_norm2.astype(np.float32)).to(device),
        "x_norm2": torch.from_numpy(x_norm2.astype(np.float32)).to(device),
        "q_errors": torch.from_numpy(upward_float32(q_errors)).to(device),
        "x_errors": torch.from_numpy(upward_float32(x_errors)).to(device),
    }
    status = torch.empty((QUERY_COUNT, BASE_COUNT), dtype=torch.uint8, device=device)
    grid = (triton.cdiv(QUERY_COUNT, BLOCK_M), triton.cdiv(BASE_COUNT, BLOCK_N))

    def launch() -> None:
        fused_int8_certificate_status[grid](
            tensors["q_codes"],
            tensors["x_codes_t"],
            tensors["q_scales"],
            tensors["x_scales"],
            tensors["q_norm2"],
            tensors["x_norm2"],
            tensors["q_errors"],
            tensors["x_errors"],
            status,
            threshold,
            BOUND_PAD,
            M=QUERY_COUNT,
            N=BASE_COUNT,
            K=DIMENSION,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    launch()
    torch.cuda.synchronize()
    status_cpu = status.cpu().numpy()
    reject = status_cpu == 0
    accept = status_cpu == 1
    ambiguous = status_cpu == 2
    invalid_status = int(np.count_nonzero(~(reject | accept | ambiguous)))
    false_accept = int(np.count_nonzero(accept & ~exact_inside))
    false_reject = int(np.count_nonzero(reject & exact_inside))
    final_inside = accept | (ambiguous & exact_inside)
    final_mismatch = int(np.count_nonzero(final_inside != exact_inside))
    ambiguous_count = int(np.count_nonzero(ambiguous))
    correctness = {
        "threshold_d2": threshold,
        "exact_pairs": int(np.count_nonzero(exact_inside)),
        "direct_accept_pairs": int(np.count_nonzero(accept)),
        "direct_reject_pairs": int(np.count_nonzero(reject)),
        "ambiguous_pairs": ambiguous_count,
        "ambiguous_fraction": ambiguous_count / exact_d2.size,
        "invalid_status": invalid_status,
        "false_accept_pairs": false_accept,
        "false_reject_pairs": false_reject,
        "final_classification_mismatch": final_mismatch,
    }
    print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)

    samples: list[float] = []
    if not args.validation_only:
        for _ in range(args.warmups):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.observations):
            start.record()
            launch()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end) * 1000.0))

    cache_dir_value = os.environ.get("TRITON_CACHE_DIR")
    cache_dir = Path(cache_dir_value) if cache_dir_value else None
    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [QUERY_COUNT, BASE_COUNT, DIMENSION],
        "tile": [BLOCK_M, BLOCK_N, BLOCK_K],
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "bound_pad": BOUND_PAD,
        "warmups": args.warmups,
        "observations": args.observations,
        "validation_only": args.validation_only,
        "correctness": correctness,
        "latency_us": samples,
        "latency_p10_us": None if args.validation_only else float(np.percentile(samples, 10)),
        "latency_median_us": None if args.validation_only else float(np.median(samples)),
        "latency_p90_us": None if args.validation_only else float(np.percentile(samples, 90)),
        "script_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": sha256_file(Path(__file__).with_name("run_b0_gpu.py")),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "triton_cache_dir": cache_dir_value,
        "triton_cache_artifacts": cache_artifacts(cache_dir),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    correctness_pass = (
        invalid_status == 0
        and false_accept == 0
        and false_reject == 0
        and final_mismatch == 0
        and correctness["ambiguous_fraction"] <= 0.002
    )
    timing_pass = None if args.validation_only else (
        record["latency_median_us"] <= 100.0 and record["latency_p90_us"] <= 110.0
    )
    record["correctness_pass"] = correctness_pass
    record["timing_pass"] = timing_pass
    record["pre_sass_gate_pass"] = None if args.validation_only else bool(correctness_pass and timing_pass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "RUN_SUMMARY "
        + json.dumps(
            {
                "correctness_pass": correctness_pass,
                "timing_pass": timing_pass,
                "median_us": record["latency_median_us"],
                "p90_us": record["latency_p90_us"],
                "validation_only": args.validation_only,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if correctness_pass and (args.validation_only or timing_pass) else 2


if __name__ == "__main__":
    raise SystemExit(main())
