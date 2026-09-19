#!/usr/bin/env python3
"""Run one TensorJoin process under the frozen G2B public denominator."""

from __future__ import annotations

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

from g2b_public_common import (
    D,
    N,
    PROJECT,
    atomic_json,
    load_contract,
    load_pageable_source,
    max_rss_kib,
    parse_public_args,
    sha256_file,
    validate_canonical,
)
from run_d1_triton import quantize_per_vector, upward_float32
from run_g2a_tensorjoin import (
    filter_ambiguous_fp32_i64,
    refine_ambiguous_fp64_i64,
)
from run_g2a2_tensorjoin_triangular import triangular_int8_certificate_compact_i64


EXPERIMENT_ID = "tensorjoin_20260903_g2b_public_screen"
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
TILES_PER_BATCH = 4_096


def compile_exact_specializations(capacity: int, threshold_d2: float) -> None:
    """Compile exact full-run specializations without retaining method state."""
    device = torch.device("cuda:0")
    warm_codes = torch.zeros((BLOCK_M, D), dtype=torch.int8, device=device)
    # N is a compile-time leading stride for the transposed operand.
    warm_codes_t = torch.zeros((D, N), dtype=torch.int8, device=device)
    warm_vector = torch.ones(BLOCK_M, dtype=torch.float32, device=device)
    warm_tile_rows = torch.zeros(1, dtype=torch.int32, device=device)
    warm_tile_columns = torch.zeros(1, dtype=torch.int32, device=device)
    warm_results = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_ambiguous = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_fp64 = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_counters = torch.zeros(5, dtype=torch.int32, device=device)
    warm_vectors = torch.zeros((BLOCK_M, D), dtype=torch.float32, device=device)

    triangular_int8_certificate_compact_i64[(1,)](
        warm_codes,
        warm_codes_t,
        warm_vector,
        warm_vector,
        warm_vector,
        warm_vector,
        warm_vector,
        warm_vector,
        warm_tile_rows,
        warm_tile_columns,
        warm_results,
        warm_ambiguous,
        warm_counters,
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
    warm_ambiguous[0] = 0
    warm_counters.zero_()
    filter_ambiguous_fp32_i64[(1,)](
        warm_vectors,
        warm_vectors,
        warm_ambiguous,
        warm_results,
        warm_fp64,
        warm_counters,
        threshold_d2,
        FP32_DISTANCE_GUARD,
        N_=N,
        K=D,
        CAPACITY=capacity,
        BLOCK_K=REFINE_BLOCK_K,
        num_warps=4,
    )
    warm_fp64[0] = 0
    warm_counters.zero_()
    refine_ambiguous_fp64_i64[(1,)](
        warm_vectors,
        warm_vectors,
        warm_fp64,
        warm_results,
        warm_counters,
        threshold_d2,
        N_=N,
        K=D,
        CAPACITY=capacity,
        BLOCK_K=REFINE_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    del (
        warm_codes,
        warm_codes_t,
        warm_vector,
        warm_tile_rows,
        warm_tile_columns,
        warm_results,
        warm_ambiguous,
        warm_fp64,
        warm_counters,
        warm_vectors,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:
    args = parse_public_args(__doc__ or "TensorJoin public adapter")
    record_id = args.record_id
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    result_path = PROJECT / f"results/g2b_public_{args.phase}_tensorjoin_{record_id}.json"
    if result_path.exists():
        raise FileExistsError(result_path)

    contract = load_contract()
    vectors = load_pageable_source()
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    scheduled_tiles = tile_extent * (tile_extent + 1) // 2
    capacity = TILES_PER_BATCH * BLOCK_M * BLOCK_N
    torch.cuda.init()
    compile_exact_specializations(capacity, float(contract["threshold_d2"]))

    run = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "record_id": record_id,
        "method": "tensorjoin_exact_mixed_precision",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [N, D],
        "source_dtype": "float32",
        "epsilon": contract["epsilon"],
        "threshold_d2": contract["threshold_d2"],
        "source_npy_sha256": contract["source_npy_sha256"],
        "metadata_sha256": contract["metadata_sha256"],
        "smoke_summary_sha256": contract["smoke_summary_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "triangular_kernel_source_sha256": sha256_file(
            PROJECT / "src/run_g2a2_tensorjoin_triangular.py"
        ),
        "refinement_source_sha256": sha256_file(
            PROJECT / "src/run_g2a_tensorjoin.py"
        ),
        "quantization_source_sha256": sha256_file(PROJECT / "src/run_d1_triton.py"),
        "timing_scope": "f32 pageable host -> sorted canonical u64 host",
        "excluded_before_timer": [
            "file_io",
            "process_startup",
            "cuda_context_initialization",
            "triton_compilation",
        ],
        "excluded_after_timer": ["hashing", "correctness_checks", "json_write"],
        "tile_extent": tile_extent,
        "scheduled_tiles": scheduled_tiles,
        "tiles_per_batch": TILES_PER_BATCH,
        "buffer_capacity_per_stage": capacity,
        "capacity_rule": "tiles_per_batch * 64 * 64 worst-case upper-pair slots",
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    torch.cuda.reset_peak_memory_stats()
    public_started = time.perf_counter()

    codes, scales, errors = quantize_per_vector(vectors)
    reconstructed_norm2 = (
        np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * scales.astype(np.float64) ** 2
    )
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    if tile_rows.size != scheduled_tiles:
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
    result_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    ambiguous_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    fp64_ids = torch.empty(capacity, dtype=torch.int64, device=device)

    accepted_parts: list[np.ndarray] = []
    batch_records: list[dict[str, int]] = []
    total_direct = 0
    total_ambiguous = 0
    total_fp32_accept = 0
    total_fp64 = 0
    total_equality = 0
    total_overflow = 0
    for batch_index, tile_start in enumerate(range(0, scheduled_tiles, TILES_PER_BATCH)):
        tile_stop = min(tile_start + TILES_PER_BATCH, scheduled_tiles)
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
            float(contract["threshold_d2"]),
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

        if ambiguous_count:
            filter_ambiguous_fp32_i64[(ambiguous_count,)](
                vectors_gpu,
                vectors_gpu,
                ambiguous_ids,
                result_ids,
                fp64_ids,
                counters,
                float(contract["threshold_d2"]),
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

        if fp64_count:
            refine_ambiguous_fp64_i64[(fp64_count,)](
                vectors_gpu,
                vectors_gpu,
                fp64_ids,
                result_ids,
                counters,
                float(contract["threshold_d2"]),
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
        accepted_parts.append(
            result_ids[: min(final_count, capacity)]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=True)
        )
        total_direct += direct_count
        total_ambiguous += ambiguous_count
        total_fp32_accept += after_fp32_count - direct_count
        total_fp64 += fp64_count
        total_equality += equality_count
        total_overflow += overflow
        batch_records.append(
            {
                "batch_index": batch_index,
                "tile_start": tile_start,
                "tile_stop": tile_stop,
                "scheduled_tiles": batch_tiles,
                "direct_accept_upper_pairs": direct_count,
                "ambiguous_upper_pairs": ambiguous_count,
                "fp32_accept_upper_pairs": after_fp32_count - direct_count,
                "fp64_refine_upper_pairs": fp64_count,
                "final_accept_upper_pairs": final_count,
                "overflow_events": overflow,
            }
        )

    torch.cuda.synchronize()
    accepted_upper = np.sort(
        np.concatenate(accepted_parts).astype(np.uint64, copy=False)
    )
    upper_rows = accepted_upper // np.uint64(N)
    upper_columns = accepted_upper - upper_rows * np.uint64(N)
    nonself = upper_rows < upper_columns
    reverse = upper_columns[nonself] * np.uint64(N) + upper_rows[nonself]
    canonical = np.sort(
        np.concatenate((accepted_upper, reverse)).astype(np.uint64, copy=False)
    )
    public_seconds = time.perf_counter() - public_started

    correctness = validate_canonical(canonical, contract)
    correctness["upper_duplicates"] = int(
        np.count_nonzero(accepted_upper[1:] == accepted_upper[:-1])
    )
    correctness["invalid_or_lower_upper_pairs"] = int(
        np.count_nonzero(
            (accepted_upper >= np.uint64(N) * np.uint64(N))
            | (upper_rows > upper_columns)
        )
    )
    correctness["overflow_events"] = total_overflow
    correctness["exact_contract_pass"] = bool(
        correctness["exact_contract_pass"]
        and correctness["upper_duplicates"] == 0
        and correctness["invalid_or_lower_upper_pairs"] == 0
        and total_overflow == 0
    )
    result = {
        **run,
        "measurement_status": (
            "formal_public_denominator_exact_candidate"
            if args.phase == "formal"
            else "diagnostic_public_denominator_cheap_screen"
        ),
        "public_seconds": public_seconds,
        "batch_count": len(batch_records),
        "batches": batch_records,
        "direct_accept_upper_pairs": total_direct,
        "ambiguous_upper_pairs": total_ambiguous,
        "fp32_accept_upper_pairs": total_fp32_accept,
        "fp64_refine_upper_pairs": total_fp64,
        "equality_upper_pairs": total_equality,
        "accepted_upper_pairs": int(accepted_upper.size),
        "torch_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_rss_kib_after_run": max_rss_kib(),
        "correctness": correctness,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if correctness["exact_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
