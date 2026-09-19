#!/usr/bin/env python3
"""Exercise the G3B-R1 candidate for memcheck or sustained safety gates."""

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
    MAX_AMBIGUOUS,
    N,
    PROJECT,
    REFINE_BLOCK_K,
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


EXPERIMENT_ID = "tensorjoin_20260903_gpu_analytic_certificate_g3b_r1"
EXPECTED_DIRECT_COUNT = 90_946
EXPECTED_AMBIGUOUS_COUNT = 102_079
EXPECTED_FINAL_UPPER_COUNT = 133_120
EXPECTED_DIRECT_SHA256 = "25c50dbd760b3f25766153e9188686ce73c504e2f65169aea831cef6f11655d5"
EXPECTED_AMBIGUOUS_SHA256 = "6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42"
EXPECTED_UPPER_SHA256 = "036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("memcheck", "stress1000"))
    return parser.parse_args()


def launch_certificate(
    codes_gpu: torch.Tensor,
    codes_t_gpu: torch.Tensor,
    scales_gpu: torch.Tensor,
    norms_gpu: torch.Tensor,
    errors_gpu: torch.Tensor,
    tile_rows_gpu: torch.Tensor,
    tile_columns_gpu: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    scheduled_tiles: int,
    capacity: int,
    epsilon_lower: np.float32,
    epsilon_upper: np.float32,
) -> None:
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
        buffers["counters"],
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


def main() -> int:
    args = parse_args()
    iterations = 1 if args.mode == "memcheck" else 1_000
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g3b_r1_safety_{args.mode}.json"
    if result_path.exists():
        raise FileExistsError(result_path)

    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    if sha256_file(vector_path) != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Vector hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")
    vectors = np.ascontiguousarray(np.load(vector_path, allow_pickle=False), dtype=np.float32)
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon_lower, epsilon_upper = outward_float32(float(metadata["radius"]["epsilon"]))
    codes, scales, reconstructed_norm2, residual_error = quantize_with_analytic_residual(vectors)
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
    buffer_sets = [
        {
            "results": torch.empty(capacity, dtype=torch.int64, device=device),
            "ambiguous": torch.empty(capacity, dtype=torch.int64, device=device),
            "counters": torch.zeros(5, dtype=torch.int32, device=device),
        }
        for _ in range(2)
    ]
    torch.cuda.synchronize()

    run = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "measurement_status": "correctness_safety_only_no_performance_claim",
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
        "candidate_source_sha256": sha256_file(PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vector_sha256": sha256_file(vector_path),
        "oracle_file_sha256": sha256_file(oracle_path),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    count_mismatches = 0
    hash_mismatches = 0
    overflow_events = 0
    exact_output_mismatches = 0
    observed_direct_hash: str | None = None
    observed_ambiguous_hash: str | None = None
    observed_upper_hash: str | None = None
    started = time.perf_counter()
    for iteration in range(iterations):
        buffers = buffer_sets[iteration % 2]
        counters = buffers["counters"]
        counters.zero_()
        launch_certificate(
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            norms_gpu,
            errors_gpu,
            tile_rows_gpu,
            tile_columns_gpu,
            buffers,
            scheduled_tiles,
            capacity,
            epsilon_lower,
            epsilon_upper,
        )
        torch.cuda.synchronize()
        scan = counters.cpu().numpy().astype(np.int64)
        direct_count = int(scan[0])
        ambiguous_count = int(scan[1])
        overflow = int(scan[2])
        direct = sorted_ids(buffers["results"], min(direct_count, capacity))
        ambiguous = sorted_ids(buffers["ambiguous"], min(ambiguous_count, capacity))
        direct_hash = sha256_u64(direct)
        ambiguous_hash = sha256_u64(ambiguous)
        if observed_direct_hash is None:
            observed_direct_hash = direct_hash
            observed_ambiguous_hash = ambiguous_hash
        if direct_count != EXPECTED_DIRECT_COUNT or ambiguous_count != EXPECTED_AMBIGUOUS_COUNT:
            count_mismatches += 1
        if direct_hash != EXPECTED_DIRECT_SHA256 or ambiguous_hash != EXPECTED_AMBIGUOUS_SHA256:
            hash_mismatches += 1
        overflow_events += overflow

        if args.mode == "memcheck":
            if ambiguous_count:
                refine_ambiguous_fp64_i64[(ambiguous_count,)](
                    vectors_gpu,
                    vectors_gpu,
                    buffers["ambiguous"],
                    buffers["results"],
                    counters,
                    threshold_d2,
                    N_=N,
                    K=D,
                    CAPACITY=capacity,
                    BLOCK_K=REFINE_BLOCK_K,
                    num_warps=4,
                )
                torch.cuda.synchronize()
            final = counters.cpu().numpy().astype(np.int64)
            final_count = int(final[0])
            overflow_events += int(final[2])
            accepted_upper = sorted_ids(buffers["results"], min(final_count, capacity))
            observed_upper_hash = sha256_u64(accepted_upper)
            oracle_rows = oracle // np.uint64(N)
            oracle_columns = oracle - oracle_rows * np.uint64(N)
            oracle_upper = np.sort(oracle[oracle_rows <= oracle_columns])
            if (
                final_count != EXPECTED_FINAL_UPPER_COUNT
                or observed_upper_hash != EXPECTED_UPPER_SHA256
                or not np.array_equal(accepted_upper, oracle_upper)
            ):
                exact_output_mismatches += 1

        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            print(
                "SAFETY_PROGRESS "
                + json.dumps(
                    {
                        "completed_iterations": iteration + 1,
                        "count_mismatches": count_mismatches,
                        "hash_mismatches": hash_mismatches,
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
        and overflow_events == 0
        and exact_output_mismatches == 0
        and observed_direct_hash == EXPECTED_DIRECT_SHA256
        and observed_ambiguous_hash == EXPECTED_AMBIGUOUS_SHA256
        and (args.mode != "memcheck" or observed_upper_hash == EXPECTED_UPPER_SHA256)
        and allocated_after_cleanup == 0
        and reserved_after_cleanup == 0
    )
    result = {
        **run,
        "completed_iterations": iterations,
        "certificate_kernel_launches": iterations,
        "fp64_refinement_kernel_launches": 1 if args.mode == "memcheck" else 0,
        "expected_direct_count": EXPECTED_DIRECT_COUNT,
        "expected_ambiguous_count": EXPECTED_AMBIGUOUS_COUNT,
        "observed_direct_sha256": observed_direct_hash,
        "observed_ambiguous_sha256": observed_ambiguous_hash,
        "observed_upper_sha256": observed_upper_hash,
        "count_mismatches": count_mismatches,
        "hash_mismatches": hash_mismatches,
        "overflow_events": overflow_events,
        "exact_output_mismatches": exact_output_mismatches,
        "ambiguous_bound": MAX_AMBIGUOUS,
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
