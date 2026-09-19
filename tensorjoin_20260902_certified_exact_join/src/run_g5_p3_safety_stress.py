#!/usr/bin/env python3
"""Exercise the exact G5 full-N specializations on oracle-checked safety tiles."""

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
from g4b_r1_ragged_safe_kernel import analytic_certificate_ragged_safe_i64
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
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    certified_fp32_filter_i64,
)
from run_g4b_r1_public_opportunity import quantize_with_analytic_residual
from run_g5_tensorjoin_public import (
    CAPACITY,
    FP64_BLOCK_K,
    compile_exact_specializations,
    require_frozen_sources,
    selected_cache_artifacts,
)


EXPERIMENT_ID = "tensorjoin_20260903_g5_unified_public_cifar60k"
TILES = (
    (0, 0),
    (0, 790),
    (1, 13),
    (1, 561),
    (1, 576),
    (1, 716),
    (1, 864),
    (2, 251),
    (2, 560),
)
EXPECTED_CUBINS = {
    "analytic_certificate_ragged_safe_i64": (
        "abe3e1403e839ef60f13af32d546593f451becf1488bb5b11b940e57b00e992d"
    ),
    "certified_fp32_filter_i64": (
        "1278324244afd13961e75cbd0a554a9bb9502c8e61193d361c015b7856208e20"
    ),
    "refine_ambiguous_fp64_i64": (
        "c2b3362119abfbeb3df85e96788705f6ff4c79414e3a74e2d5d5f3355eff00f8"
    ),
}
EXPECTED_PTX = {
    "analytic_certificate_ragged_safe_i64": (
        "c1faa40dd01b0d187836de9bc7f30bbb750aefe6c720abe92cef7fe13c67e701"
    ),
    "certified_fp32_filter_i64": (
        "dd60c5856a950e63fe64ff8d790ec623d73915ba334c6271c42f58ff05995f41"
    ),
    "refine_ambiguous_fp64_i64": (
        "8c90341103eb4aea692cd0b8e6501126c0e6b6b6591668b46e1492257aea162d"
    ),
}
P1_RESULT_SHA256 = "7fcdd2786db413ee9f56347c27eceb9542e41e8fc644e9dd3a55d9ec59bff2a5"
P2_RESULT_SHA256 = "ffcfdf9c4949756ad1de8974f7e4453378dc26690abba3c292a551553c48482b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if not args.record_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe record ID: {args.record_id!r}")
    if args.iterations < 1:
        raise ValueError("Iterations must be positive")
    return args


def build_oracle(vectors: np.ndarray, threshold_d2: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    for tile_row, tile_column in TILES:
        rows = np.arange(tile_row * BLOCK_M, (tile_row + 1) * BLOCK_M)
        columns = np.arange(tile_column * BLOCK_N, (tile_column + 1) * BLOCK_N)
        rows = rows[rows < N]
        columns = columns[columns < N]
        x = vectors[rows].astype(np.float64)
        y = vectors[columns].astype(np.float64)
        distances = np.sum(
            (x[:, None, :] - y[None, :, :]) ** 2, axis=2, dtype=np.float64
        )
        valid = rows[:, None] <= columns[None, :]
        accepted_row, accepted_column = np.nonzero(valid & (distances <= threshold_d2))
        pair_ids = (
            rows[accepted_row].astype(np.uint64) * np.uint64(N)
            + columns[accepted_column].astype(np.uint64)
        )
        parts.append(pair_ids)
    oracle = np.sort(np.concatenate(parts).astype(np.uint64, copy=False))
    if oracle.size > 1 and np.any(oracle[1:] == oracle[:-1]):
        raise RuntimeError("Frozen safety tiles overlap")
    return oracle


def binary_identity(artifacts: dict[str, dict[str, object]]) -> bool:
    return all(
        artifacts[name]["cubin_sha256"] == EXPECTED_CUBINS[name]
        and artifacts[name]["ptx_sha256"] == EXPECTED_PTX[name]
        for name in EXPECTED_CUBINS
    )


def main() -> int:
    args = parse_args()
    physical_gpu = os.environ.get("G5_PHYSICAL_GPU", "2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu:
        raise RuntimeError("G5 safety must run on its one declared visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    cache_text = os.environ.get("TRITON_CACHE_DIR")
    if not cache_text:
        raise RuntimeError("G5 safety requires a fresh TRITON_CACHE_DIR")
    cache = Path(cache_text).resolve()
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError(cache)
    cache.mkdir(parents=True, exist_ok=True)
    result_path = PROJECT / f"results/g5_safety_{args.record_id}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if sha256_file(PROJECT / "results/g5_compatibility_tensorjoin_p1_r1_a0.json") != P1_RESULT_SHA256:
        raise RuntimeError("G5 P1 result changed")
    if sha256_file(PROJECT / "results/g5_p2_generated_code_audit_p1.json") != P2_RESULT_SHA256:
        raise RuntimeError("G5 P2 result changed")
    upstream_hashes = require_frozen_sources()
    contract = load_contract()
    vectors = load_pageable_source()
    oracle = build_oracle(vectors, float(contract["threshold_d2"]))
    epsilon_lower, epsilon_upper = outward_float32(float(contract["epsilon"]))
    threshold_lower, threshold_upper = outward_float32(float(contract["threshold_d2"]))

    torch.cuda.init()
    compile_exact_specializations(
        float(epsilon_lower),
        float(epsilon_upper),
        float(threshold_lower),
        float(threshold_upper),
        float(contract["threshold_d2"]),
    )
    artifacts_before = selected_cache_artifacts(cache)
    if not binary_identity(artifacts_before):
        raise RuntimeError("Safety specialization differs from G5 P1")

    codes, scales, reconstructed_norm2, residual_error = (
        quantize_with_analytic_residual(vectors)
    )
    tile_rows = np.asarray([row for row, _ in TILES], dtype=np.int32)
    tile_columns = np.asarray([column for _, column in TILES], dtype=np.int32)
    device = torch.device("cuda:0")
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
            "results": torch.empty(CAPACITY, dtype=torch.int64, device=device),
            "ambiguous": torch.empty(CAPACITY, dtype=torch.int64, device=device),
            "fp64": torch.empty(CAPACITY, dtype=torch.int64, device=device),
            "counters": torch.zeros(6, dtype=torch.int32, device=device),
        }
        for _ in range(2)
    ]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    expected_stage_counts: tuple[int, int, int, int, int, int] | None = None
    stage_count_mismatches = 0
    output_mismatches = 0
    overflow_events = 0
    duplicate_or_invalid_outputs = 0
    first_output_hash: str | None = None
    started = time.perf_counter()
    for iteration in range(args.iterations):
        buffers = buffer_sets[iteration % 2]
        counters = buffers["counters"]
        counters.zero_()
        analytic_certificate_ragged_safe_i64[(len(TILES),)](
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
            CAPACITY=CAPACITY,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        scan = counters.cpu().numpy().astype(np.int64)
        direct_count = int(scan[0])
        ambiguous_count = int(scan[1])
        if ambiguous_count:
            certified_fp32_filter_i64[(ambiguous_count,)](
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
                CAPACITY=CAPACITY,
                BLOCK_K=FP32_BLOCK_K,
                num_warps=4,
            )
        after_fp32 = counters.cpu().numpy().astype(np.int64)
        fp32_accept_count = int(after_fp32[0]) - direct_count
        fp64_count = int(after_fp32[3])
        fp32_reject_count = int(after_fp32[4])
        equality_count = int(after_fp32[5])
        if fp64_count:
            refine_ambiguous_fp64_i64[(fp64_count,)](
                vectors_gpu,
                vectors_gpu,
                buffers["fp64"],
                buffers["results"],
                counters,
                float(contract["threshold_d2"]),
                N_=N,
                K=D,
                CAPACITY=CAPACITY,
                BLOCK_K=FP64_BLOCK_K,
                num_warps=4,
            )
        final = counters.cpu().numpy().astype(np.int64)
        final_count = int(final[0])
        overflow = int(final[2])
        overflow_events += overflow
        accepted = np.sort(
            buffers["results"][: min(final_count, CAPACITY)]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=True)
        )
        output_hash = sha256_u64(accepted)
        if first_output_hash is None:
            first_output_hash = output_hash
        if not np.array_equal(accepted, oracle):
            output_mismatches += 1
        rows = accepted // np.uint64(N)
        columns = accepted - rows * np.uint64(N)
        if (
            np.count_nonzero(accepted[1:] == accepted[:-1])
            or np.count_nonzero(accepted >= np.uint64(N) * np.uint64(N))
            or np.count_nonzero(rows > columns)
        ):
            duplicate_or_invalid_outputs += 1
        stage_counts = (
            direct_count,
            ambiguous_count,
            fp32_accept_count,
            fp32_reject_count,
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
                        "stage_counts": stage_counts,
                        "output_mismatches": output_mismatches,
                        "stage_count_mismatches": stage_count_mismatches,
                        "overflow_events": overflow_events,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    elapsed_seconds = time.perf_counter() - started
    artifacts_after = selected_cache_artifacts(cache)
    binary_gate = bool(
        artifacts_after == artifacts_before and binary_identity(artifacts_after)
    )
    all_stages_exercised = bool(
        expected_stage_counts is not None
        and all(value > 0 for value in expected_stage_counts[:5])
    )
    pre_cleanup_pass = bool(
        output_mismatches == 0
        and stage_count_mismatches == 0
        and overflow_events == 0
        and duplicate_or_invalid_outputs == 0
        and first_output_hash == sha256_u64(oracle)
        and all_stages_exercised
        and binary_gate
    )
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
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    safety_pass = bool(
        pre_cleanup_pass and allocated_after == 0 and reserved_after == 0
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "record_id": args.record_id,
        "iterations": args.iterations,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "physical_gpu": physical_gpu,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "full_shape": [N, D],
        "selected_tiles": [list(tile) for tile in TILES],
        "selected_valid_upper_pairs": int(
            sum(
                BLOCK_M * BLOCK_N if row < column else BLOCK_M * (BLOCK_M + 1) // 2
                for row, column in TILES
            )
        ),
        "oracle_upper_count": int(oracle.size),
        "oracle_upper_raw_u64_sha256": sha256_u64(oracle),
        "source_npy_sha256": contract["source_npy_sha256"],
        "upstream_source_hashes": upstream_hashes,
        "p1_result_sha256": P1_RESULT_SHA256,
        "p2_result_sha256": P2_RESULT_SHA256,
        "specialization_contract": {
            "N_": N,
            "K": D,
            "CAPACITY": CAPACITY,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "FP32_BLOCK_K": FP32_BLOCK_K,
            "FP64_BLOCK_K": FP64_BLOCK_K,
            "num_warps": 4,
            "stage1_num_stages": 3,
        },
        "selected_cache_artifacts_before": artifacts_before,
        "selected_cache_artifacts_after": artifacts_after,
        "p1_binary_identity_pass": binary_gate,
        "pointer_buffer_sets": 2,
        "completed_iterations": args.iterations,
        "elapsed_seconds_diagnostic": elapsed_seconds,
        "stage_counts_direct_ambiguous_fp32_accept_fp32_reject_fp64_equality": expected_stage_counts,
        "all_three_precision_stages_exercised": all_stages_exercised,
        "output_mismatches": output_mismatches,
        "stage_count_mismatches": stage_count_mismatches,
        "duplicate_or_invalid_outputs": duplicate_or_invalid_outputs,
        "overflow_events": overflow_events,
        "observed_output_hash": first_output_hash,
        "torch_peak_memory_allocated_bytes": peak_allocated,
        "torch_peak_memory_reserved_bytes": peak_reserved,
        "torch_memory_allocated_after_cleanup_bytes": allocated_after,
        "torch_memory_reserved_after_cleanup_bytes": reserved_after,
        "max_rss_kib_after_run": max_rss_kib(),
        "safety_pass": safety_pass,
        "correctness": {"exact_contract_pass": safety_pass},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G5_P3_SAFETY_STRESS.md"),
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if safety_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

