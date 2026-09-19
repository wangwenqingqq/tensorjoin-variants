#!/usr/bin/env python3
"""Exercise the exact TensorJoin scalable core for memcheck and stress gates."""

from __future__ import annotations

import argparse
import gc
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

from g2b_public_common import (
    D,
    N,
    PROJECT,
    atomic_json,
    load_contract,
    load_pageable_source,
    max_rss_kib,
    sha256_file,
    sha256_u64,
)
from run_d1_triton import quantize_per_vector, upward_float32
from run_g2a_tensorjoin import (
    filter_ambiguous_fp32_i64,
    refine_ambiguous_fp64_i64,
)
from run_g2a2_tensorjoin_triangular import triangular_int8_certificate_compact_i64


EXPERIMENT_ID = "tensorjoin_20260903_g2b_tensorjoin_safety"
N_SAFE = 256
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
FULL_PAIR_FILE = PROJECT / "artifacts/g2b/tensorjoin_pairs_u64_le.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if not args.record_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe record id: {args.record_id!r}")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    return args


def select_dense_subset_and_oracle(
    full_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = full_pairs // np.uint64(N)
    columns = full_pairs - rows * np.uint64(N)
    degrees = np.bincount(rows.astype(np.int64), minlength=N)
    selected = np.lexsort((np.arange(N, dtype=np.int64), -degrees))[:N_SAFE]
    inverse = np.full(N, -1, dtype=np.int64)
    inverse[selected] = np.arange(N_SAFE, dtype=np.int64)
    keep = (inverse[rows] >= 0) & (inverse[columns] >= 0)
    local_rows = inverse[rows[keep]].astype(np.uint64)
    local_columns = inverse[columns[keep]].astype(np.uint64)
    oracle = np.sort(local_rows * np.uint64(N_SAFE) + local_columns)
    return selected, oracle


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g2b_tensorjoin_safety_{args.record_id}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if not FULL_PAIR_FILE.is_file():
        raise FileNotFoundError(FULL_PAIR_FILE)

    contract = load_contract()
    full_pairs = np.fromfile(FULL_PAIR_FILE, dtype="<u8")
    if (
        full_pairs.size != int(contract["expected_count"])
        or sha256_u64(full_pairs) != contract["expected_hash"]
    ):
        raise RuntimeError("Full accepted pair file does not match the frozen contract")
    selected, oracle = select_dense_subset_and_oracle(full_pairs)
    full_vectors = load_pageable_source()
    vectors = np.ascontiguousarray(full_vectors[selected], dtype=np.float32)
    codes, scales, errors = quantize_per_vector(vectors)
    reconstructed_norm2 = (
        np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * scales.astype(np.float64) ** 2
    )
    tile_extent = (N_SAFE + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    capacity = int(tile_rows.size) * BLOCK_M * BLOCK_N

    device = torch.device("cuda:0")
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2.astype(np.float32)).to(device)
    errors_gpu = torch.from_numpy(upward_float32(errors)).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    buffer_sets = [
        {
            "results": torch.empty(capacity, dtype=torch.int64, device=device),
            "ambiguous": torch.empty(capacity, dtype=torch.int64, device=device),
            "fp64": torch.empty(capacity, dtype=torch.int64, device=device),
            "counters": torch.zeros(5, dtype=torch.int32, device=device),
        }
        for _ in range(2)
    ]
    torch.cuda.synchronize()

    run = {
        "experiment_id": EXPERIMENT_ID,
        "record_id": args.record_id,
        "iterations": args.iterations,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "full_shape": [N, D],
        "safety_shape": [N_SAFE, D],
        "subset_rule": "top directed degree; original-id tie break",
        "selected_original_ids_sha256": hashlib.sha256(
            selected.astype("<u4").tobytes()
        ).hexdigest(),
        "oracle_pair_count": int(oracle.size),
        "oracle_raw_u64_sha256": sha256_u64(oracle),
        "full_pair_file_sha256": sha256_file(FULL_PAIR_FILE),
        "source_npy_sha256": contract["source_npy_sha256"],
        "threshold_d2": contract["threshold_d2"],
        "tile_extent": tile_extent,
        "scheduled_tiles": int(tile_rows.size),
        "capacity": capacity,
        "pointer_buffer_sets": len(buffer_sets),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    expected_stage_counts: tuple[int, int, int, int, int] | None = None
    stage_count_mismatches = 0
    output_mismatches = 0
    overflow_events = 0
    core_kernel_launches = 0
    first_output_hash: str | None = None
    started = time.perf_counter()
    for iteration in range(args.iterations):
        buffers = buffer_sets[iteration % len(buffer_sets)]
        counters = buffers["counters"]
        counters.zero_()
        triangular_int8_certificate_compact_i64[(int(tile_rows.size),)](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            scales_gpu,
            norms_gpu,
            norms_gpu,
            errors_gpu,
            errors_gpu,
            tile_rows_gpu,
            tile_columns_gpu,
            buffers["results"],
            buffers["ambiguous"],
            counters,
            float(contract["threshold_d2"]),
            BOUND_PAD,
            M=N_SAFE,
            N_=N_SAFE,
            K=D,
            CAPACITY=capacity,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        core_kernel_launches += 1
        torch.cuda.synchronize()
        scan = counters.cpu().numpy().astype(np.int64)
        direct_count = int(scan[0])
        ambiguous_count = int(scan[1])
        if ambiguous_count:
            filter_ambiguous_fp32_i64[(ambiguous_count,)](
                vectors_gpu,
                vectors_gpu,
                buffers["ambiguous"],
                buffers["results"],
                buffers["fp64"],
                counters,
                float(contract["threshold_d2"]),
                FP32_DISTANCE_GUARD,
                N_=N_SAFE,
                K=D,
                CAPACITY=capacity,
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
            )
            core_kernel_launches += 1
            torch.cuda.synchronize()
        after_fp32 = counters.cpu().numpy().astype(np.int64)
        fp32_accepted = int(after_fp32[0]) - direct_count
        fp64_count = int(after_fp32[3])
        equality_count = int(after_fp32[4])
        if fp64_count:
            refine_ambiguous_fp64_i64[(fp64_count,)](
                vectors_gpu,
                vectors_gpu,
                buffers["fp64"],
                buffers["results"],
                counters,
                float(contract["threshold_d2"]),
                N_=N_SAFE,
                K=D,
                CAPACITY=capacity,
                BLOCK_K=REFINE_BLOCK_K,
                num_warps=4,
            )
            core_kernel_launches += 1
            torch.cuda.synchronize()
        final = counters.cpu().numpy().astype(np.int64)
        final_count = int(final[0])
        iteration_overflow = int(final[2])
        overflow_events += iteration_overflow
        accepted_upper = np.sort(
            buffers["results"][: min(final_count, capacity)]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=True)
        )
        upper_rows = accepted_upper // np.uint64(N_SAFE)
        upper_columns = accepted_upper - upper_rows * np.uint64(N_SAFE)
        nonself = upper_rows < upper_columns
        reverse = upper_columns[nonself] * np.uint64(N_SAFE) + upper_rows[nonself]
        canonical = np.sort(np.concatenate((accepted_upper, reverse)).astype(np.uint64))
        output_hash = sha256_u64(canonical)
        if first_output_hash is None:
            first_output_hash = output_hash
        if not np.array_equal(canonical, oracle):
            output_mismatches += 1
        stage_counts = (
            direct_count,
            ambiguous_count,
            fp32_accepted,
            fp64_count,
            equality_count,
        )
        if expected_stage_counts is None:
            expected_stage_counts = stage_counts
        elif stage_counts != expected_stage_counts:
            stage_count_mismatches += 1
        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == args.iterations:
            print(
                "STRESS_PROGRESS "
                + json.dumps(
                    {
                        "completed_iterations": iteration + 1,
                        "output_mismatches": output_mismatches,
                        "stage_count_mismatches": stage_count_mismatches,
                        "overflow_events": overflow_events,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    elapsed_seconds = time.perf_counter() - started
    all_stages_exercised = bool(
        expected_stage_counts is not None
        and all(value > 0 for value in expected_stage_counts[:4])
    )
    safety_pass = bool(
        output_mismatches == 0
        and stage_count_mismatches == 0
        and overflow_events == 0
        and first_output_hash == sha256_u64(oracle)
        and all_stages_exercised
    )
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    # Do not make PyTorch's caching allocator look like an application leak to
    # compute-sanitizer's full leak check. All device state is dead here.
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
        safety_pass
        and allocated_after_cleanup == 0
        and reserved_after_cleanup == 0
    )
    result = {
        **run,
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "elapsed_seconds_diagnostic": elapsed_seconds,
        "completed_iterations": args.iterations,
        "total_core_kernel_launches": core_kernel_launches,
        "stage_counts_direct_ambiguous_fp32_accept_fp64_refine_equality": expected_stage_counts,
        "all_three_precision_stages_exercised": all_stages_exercised,
        "output_mismatches": output_mismatches,
        "stage_count_mismatches": stage_count_mismatches,
        "overflow_events": overflow_events,
        "observed_output_hash": first_output_hash,
        "max_rss_kib_after_run": max_rss_kib(),
        "torch_peak_memory_allocated_bytes": peak_allocated_bytes,
        "torch_peak_memory_reserved_bytes": peak_reserved_bytes,
        "torch_memory_allocated_after_cleanup_bytes": allocated_after_cleanup,
        "torch_memory_reserved_after_cleanup_bytes": reserved_after_cleanup,
        "safety_pass": safety_pass,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if safety_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
