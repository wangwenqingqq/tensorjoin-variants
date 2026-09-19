#!/usr/bin/env python3
"""Exercise the complete G3C-B-R1 cascade for memcheck or stress gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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
    D,
    DATA,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    EXPECTED_ORACLE_FILE_SHA256,
    EXPECTED_ORACLE_RAW_SHA256,
    EXPECTED_VECTOR_SHA256,
    N,
    PROJECT,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    analytic_certificate_compact_i64,
    atomic_json,
    outward_float32,
    quantize_with_analytic_residual,
    sha256_file,
    sha256_u64,
    sorted_ids,
)
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    FP64_BLOCK_K,
    certified_fp32_filter_i64,
)


EXPERIMENT_ID = "tensorjoin_20260903_g3c_b_r1_gpu_cascade"
EXPECTED_G3B_DIRECT_COUNT = 90_946
EXPECTED_G3B_AMBIGUOUS_COUNT = 102_079
EXPECTED_FP32_ACCEPT_COUNT = 42_123
EXPECTED_FP32_REJECT_COUNT = 59_859
EXPECTED_FP64_COUNT = 97
EXPECTED_FINAL_UPPER_COUNT = 133_120
EXPECTED_G3B_DIRECT_SHA256 = (
    "25c50dbd760b3f25766153e9188686ce73c504e2f65169aea831cef6f11655d5"
)
EXPECTED_G3B_AMBIGUOUS_SHA256 = (
    "6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42"
)
EXPECTED_FP32_ACCEPT_SHA256 = (
    "454f909731f284d0351858c2394fcd5889f2b1126f8a6ad7fd758ce5e0bf71c1"
)
EXPECTED_FP32_REJECT_SHA256 = (
    "bf4c7d837e5cc9811e6fda88df77e2e31c5556de1821ab7444a7ec3eb86691bc"
)
EXPECTED_FP64_SHA256 = (
    "4a7fbf0c05fefcfbe8b5ff36d05ca39803895c18b22a1c2c7db9614690416a13"
)
EXPECTED_UPPER_SHA256 = (
    "036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("memcheck", "stress1000"))
    return parser.parse_args()


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def main() -> int:
    args = parse_args()
    iterations = 1 if args.mode == "memcheck" else 1_000
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g3c_b_r1_safety_{args.mode}.json"
    if result_path.exists():
        raise FileExistsError(result_path)

    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    if sha256_file(vector_path) != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Vector hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")
    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon_lower, epsilon_upper = outward_float32(
        float(metadata["radius"]["epsilon"])
    )
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    codes, scales, reconstructed_norm2, residual_error = (
        quantize_with_analytic_residual(vectors)
    )
    if D * 127 * 127 > min(2**24, np.iinfo(np.int32).max):
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
    buffer_sets = [
        {
            "results": torch.empty(capacity, dtype=torch.int64, device=device),
            "ambiguous": torch.empty(capacity, dtype=torch.int64, device=device),
            "fp64": torch.empty(capacity, dtype=torch.int64, device=device),
            "counters": torch.zeros(6, dtype=torch.int32, device=device),
        }
        for _ in range(2)
    ]
    torch.cuda.synchronize()

    run = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "measurement_status": "complete_cascade_safety_only_no_performance_claim",
        "iterations": iterations,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [N, D],
        "scheduled_tiles": scheduled_tiles,
        "capacity": capacity,
        "pointer_buffer_sets": len(buffer_sets),
        "candidate_source_sha256": sha256_file(
            PROJECT / "src/run_g3c_b_r1_gpu_cascade.py"
        ),
        "g3b_source_sha256": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
        "fp64_source_sha256": sha256_file(PROJECT / "src/run_g2a_tensorjoin.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vector_sha256": sha256_file(vector_path),
        "oracle_file_sha256": sha256_file(oracle_path),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    count_mismatches = 0
    hash_mismatches = 0
    duplicate_mismatches = 0
    partition_mismatches = 0
    overflow_events = 0
    exact_output_mismatches = 0
    observed_hashes: dict[str, str] = {}
    started = time.perf_counter()
    oracle_rows = oracle // np.uint64(N)
    oracle_columns = oracle - oracle_rows * np.uint64(N)
    oracle_upper = np.sort(oracle[oracle_rows <= oracle_columns])

    for iteration in range(iterations):
        buffers = buffer_sets[iteration % 2]
        counters = buffers["counters"]
        counters.zero_()
        analytic_certificate_compact_i64[(scheduled_tiles,)](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            norms_gpu,
            errors_gpu,
            tile_rows_gpu,
            tile_columns_gpu,
            buffers["results"],
            buffers["ambiguous"],
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
        if g3b_ambiguous_count:
            certified_fp32_filter_i64[(g3b_ambiguous_count,)](
                vectors_gpu,
                buffers["ambiguous"],
                buffers["results"],
                buffers["fp64"],
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
        if fp64_count:
            refine_ambiguous_fp64_i64[(fp64_count,)](
                vectors_gpu,
                vectors_gpu,
                buffers["fp64"],
                buffers["results"],
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
        overflow_events += int(final_counts[2])

        g3b_direct = sorted_ids(buffers["results"], g3b_direct_count)
        g3b_ambiguous = sorted_ids(buffers["ambiguous"], g3b_ambiguous_count)
        accepted_before_fp64 = sorted_ids(
            buffers["results"], accepted_before_fp64_count
        )
        fp32_accept = np.setdiff1d(
            accepted_before_fp64, g3b_direct, assume_unique=True
        )
        fp64_pairs = sorted_ids(buffers["fp64"], fp64_count)
        fp32_reject = np.setdiff1d(
            g3b_ambiguous,
            np.union1d(fp32_accept, fp64_pairs),
            assume_unique=True,
        )
        accepted_upper = sorted_ids(buffers["results"], final_count)
        hashes = {
            "g3b_direct": sha256_u64(g3b_direct),
            "g3b_ambiguous": sha256_u64(g3b_ambiguous),
            "fp32_accept": sha256_u64(fp32_accept),
            "fp32_reject": sha256_u64(fp32_reject),
            "fp64": sha256_u64(fp64_pairs),
            "accepted_upper": sha256_u64(accepted_upper),
        }
        if not observed_hashes:
            observed_hashes = hashes
        expected_counts = (
            EXPECTED_G3B_DIRECT_COUNT,
            EXPECTED_G3B_AMBIGUOUS_COUNT,
            EXPECTED_FP32_ACCEPT_COUNT,
            EXPECTED_FP32_REJECT_COUNT,
            EXPECTED_FP64_COUNT,
            EXPECTED_FINAL_UPPER_COUNT,
        )
        actual_counts = (
            g3b_direct_count,
            g3b_ambiguous_count,
            fp32_accept_count,
            fp32_reject_count,
            fp64_count,
            final_count,
        )
        if actual_counts != expected_counts:
            count_mismatches += 1
        expected_hashes = {
            "g3b_direct": EXPECTED_G3B_DIRECT_SHA256,
            "g3b_ambiguous": EXPECTED_G3B_AMBIGUOUS_SHA256,
            "fp32_accept": EXPECTED_FP32_ACCEPT_SHA256,
            "fp32_reject": EXPECTED_FP32_REJECT_SHA256,
            "fp64": EXPECTED_FP64_SHA256,
            "accepted_upper": EXPECTED_UPPER_SHA256,
        }
        if hashes != expected_hashes:
            hash_mismatches += 1
        if any(
            duplicate_count(values) != 0
            for values in (
                g3b_direct,
                g3b_ambiguous,
                fp32_accept,
                fp32_reject,
                fp64_pairs,
                accepted_upper,
            )
        ):
            duplicate_mismatches += 1
        if fp32_accept_count + fp32_reject_count + fp64_count != g3b_ambiguous_count:
            partition_mismatches += 1
        if args.mode == "memcheck" and not np.array_equal(accepted_upper, oracle_upper):
            exact_output_mismatches += 1

        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            print(
                "SAFETY_PROGRESS "
                + json.dumps(
                    {
                        "completed_iterations": iteration + 1,
                        "count_mismatches": count_mismatches,
                        "hash_mismatches": hash_mismatches,
                        "duplicate_mismatches": duplicate_mismatches,
                        "partition_mismatches": partition_mismatches,
                        "overflow_events": overflow_events,
                        "exact_output_mismatches": exact_output_mismatches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del (
        buffers,
        counters,
        buffer_sets,
        vectors_gpu,
        codes_gpu,
        codes_t_gpu,
        scales_gpu,
        norms_gpu,
        errors_gpu,
        tile_rows_gpu,
        tile_columns_gpu,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_after_cleanup = int(torch.cuda.memory_allocated())
    reserved_after_cleanup = int(torch.cuda.memory_reserved())
    safety_pass = bool(
        count_mismatches == 0
        and hash_mismatches == 0
        and duplicate_mismatches == 0
        and partition_mismatches == 0
        and overflow_events == 0
        and exact_output_mismatches == 0
        and observed_hashes.get("accepted_upper") == EXPECTED_UPPER_SHA256
        and allocated_after_cleanup == 0
        and reserved_after_cleanup == 0
    )
    result = {
        **run,
        "completed_iterations": iterations,
        "g3b_kernel_launches": iterations,
        "certified_fp32_kernel_launches": iterations,
        "fp64_refinement_kernel_launches": iterations,
        "expected_counts": {
            "g3b_direct": EXPECTED_G3B_DIRECT_COUNT,
            "g3b_ambiguous": EXPECTED_G3B_AMBIGUOUS_COUNT,
            "fp32_accept": EXPECTED_FP32_ACCEPT_COUNT,
            "fp32_reject": EXPECTED_FP32_REJECT_COUNT,
            "fp64": EXPECTED_FP64_COUNT,
            "final_upper": EXPECTED_FINAL_UPPER_COUNT,
        },
        "observed_hashes": observed_hashes,
        "count_mismatches": count_mismatches,
        "hash_mismatches": hash_mismatches,
        "duplicate_mismatches": duplicate_mismatches,
        "partition_mismatches": partition_mismatches,
        "overflow_events": overflow_events,
        "exact_output_mismatches": exact_output_mismatches,
        "diagnostic_elapsed_seconds": elapsed,
        "torch_peak_memory_allocated_bytes": peak_allocated,
        "torch_peak_memory_reserved_bytes": peak_reserved,
        "torch_memory_allocated_after_cleanup_bytes": allocated_after_cleanup,
        "torch_memory_reserved_after_cleanup_bytes": reserved_after_cleanup,
        "safety_pass": safety_pass,
        "performance_claim_allowed": False,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if safety_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
