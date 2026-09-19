#!/usr/bin/env python3
"""Run the G5 audited three-stage router under the full public denominator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
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
    validate_canonical,
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
from gpu_preparation import prepare_device


EXPERIMENT_ID = "tensorjoin_20260903_g5_unified_public_cifar60k"
PHASES = ("compatibility", "screen", "formal")
RECORD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
FP64_BLOCK_K = 256
TILES_PER_BATCH = 4_096
CAPACITY = TILES_PER_BATCH * BLOCK_M * BLOCK_N
KERNEL_NAMES = (
    "analytic_certificate_ragged_safe_i64",
    "certified_fp32_filter_i64",
    "refine_ambiguous_fp64_i64",
)
UPSTREAM_HASHES = {
    "src/g4b_r1_ragged_safe_kernel.py": (
        "439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6"
    ),
    "src/run_g3b_r1_gpu_analytic_certificate.py": (
        "057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec"
    ),
    "src/run_g3c_b_r1_gpu_cascade.py": (
        "637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d"
    ),
    "src/run_g2a_tensorjoin.py": (
        "84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8"
    ),
    "src/run_g4b_r1_public_opportunity.py": (
        "c1d539458233fb3268149e74847d8e6a5c8f4e3a3400954e7a5ee44a74f36234"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--write-pairs",
        action="store_true",
        help="Serialize canonical IDs after the public timer for compatibility evidence.",
    )
    args = parser.parse_args()
    if not RECORD_ID_PATTERN.fullmatch(args.record_id):
        raise ValueError(f"Unsafe record ID: {args.record_id!r}")
    if args.write_pairs and args.phase != "compatibility":
        raise ValueError("Pair serialization is compatibility-only")
    return args


def sha256_array(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def require_frozen_sources() -> dict[str, str]:
    observed = {path: sha256_file(PROJECT / path) for path in UPSTREAM_HASHES}
    if observed != UPSTREAM_HASHES:
        raise RuntimeError(
            "Frozen G5 upstream source mismatch: "
            + json.dumps({"expected": UPSTREAM_HASHES, "observed": observed}, sort_keys=True)
        )
    return observed


def exactly_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}, found {len(matches)}")
    return matches[0]


def selected_cache_artifacts(cache: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for kernel in KERNEL_NAMES:
        entries: dict[str, object] = {}
        for suffix in ("cubin", "ptx", "json"):
            path = exactly_one(cache, f"{kernel}.{suffix}")
            entries[suffix] = str(path.relative_to(PROJECT))
            entries[f"{suffix}_sha256"] = sha256_file(path)
            entries[f"{suffix}_bytes"] = path.stat().st_size
        result[kernel] = entries
    return result


def gpu_state(physical_gpu: str) -> str:
    fields = (
        "index,uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "memory.used,utilization.gpu"
    )
    return subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--id={physical_gpu}",
        ],
        text=True,
    ).strip()


def compile_exact_specializations(
    epsilon_lower: float,
    epsilon_upper: float,
    threshold_lower: float,
    threshold_upper: float,
    threshold_d2: float,
) -> None:
    """Compile the exact full-N/CAPACITY specializations outside the timer."""
    device = torch.device("cuda:0")
    warm_codes = torch.zeros((BLOCK_M, D), dtype=torch.int8, device=device)
    # The transposed operand uses N as its compile-time leading stride.
    warm_codes_t = torch.zeros((D, N), dtype=torch.int8, device=device)
    warm_vector = torch.zeros((1, D), dtype=torch.float32, device=device)
    warm_scale = torch.ones(BLOCK_M, dtype=torch.float32, device=device)
    warm_norm = torch.zeros(BLOCK_M, dtype=torch.float32, device=device)
    warm_error = torch.zeros(BLOCK_M, dtype=torch.float32, device=device)
    warm_tile_rows = torch.zeros(1, dtype=torch.int32, device=device)
    warm_tile_columns = torch.zeros(1, dtype=torch.int32, device=device)
    warm_results = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_ambiguous = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_fp64 = torch.empty(BLOCK_M * BLOCK_N, dtype=torch.int64, device=device)
    warm_counters = torch.zeros(6, dtype=torch.int32, device=device)

    analytic_certificate_ragged_safe_i64[(1,)](
        warm_codes,
        warm_codes_t,
        warm_scale,
        warm_norm,
        warm_error,
        warm_tile_rows,
        warm_tile_columns,
        warm_results,
        warm_ambiguous,
        warm_counters,
        epsilon_lower,
        epsilon_upper,
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
    warm_ambiguous[0] = 0
    warm_counters.zero_()
    certified_fp32_filter_i64[(1,)](
        warm_vector,
        warm_ambiguous,
        warm_results,
        warm_fp64,
        warm_counters,
        threshold_lower,
        threshold_upper,
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
    warm_fp64[0] = 0
    warm_counters.zero_()
    refine_ambiguous_fp64_i64[(1,)](
        warm_vector,
        warm_vector,
        warm_fp64,
        warm_results,
        warm_counters,
        threshold_d2,
        N_=N,
        K=D,
        CAPACITY=CAPACITY,
        BLOCK_K=FP64_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    del (
        warm_codes,
        warm_codes_t,
        warm_vector,
        warm_scale,
        warm_norm,
        warm_error,
        warm_tile_rows,
        warm_tile_columns,
        warm_results,
        warm_ambiguous,
        warm_fp64,
        warm_counters,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def metadata_artifacts(cache):
    result = {}
    for suffix in ("cubin", "ptx"):
        path = exactly_one(cache, "gpu_quantize_metadata." + suffix)
        result[suffix] = {"path": str(path.relative_to(PROJECT)), "sha256": sha256_file(path)}
    return result


def main() -> int:
    args = parse_args()
    physical_gpu = os.environ.get("G5_PHYSICAL_GPU", "1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != physical_gpu:
        raise RuntimeError(
            f"G5 expects CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    cache_text = os.environ.get("TRITON_CACHE_DIR")
    if not cache_text:
        raise RuntimeError("G5 requires an explicit fresh TRITON_CACHE_DIR")
    cache = Path(cache_text).resolve()
    try:
        cache.relative_to(PROJECT)
    except ValueError as error:
        raise RuntimeError("G5 Triton cache must live under the project") from error
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError(f"G5 cache is not empty: {cache}")
    cache.mkdir(parents=True, exist_ok=True)

    result_path = PROJECT / f"g16_gpu_preparation_20260905/results/gpu_prepare_{args.phase}_{args.record_id}.json"
    pair_path = PROJECT / f"artifacts/g5_{args.phase}/tensorjoin_{args.record_id}.u64.bin"
    for path in ((result_path, pair_path) if args.write_pairs else (result_path,)):
        if path.exists():
            raise FileExistsError(path)

    metadata_gate = json.loads((PROJECT / "g16_gpu_preparation_20260905/results/metadata_a0.json").read_text())
    if not metadata_gate["correctness"]["exact_contract_pass"]:
        raise RuntimeError("G16 metadata gate did not admit full join")
    upstream_hashes = require_frozen_sources()
    contract = load_contract()
    vectors = load_pageable_source()
    epsilon_lower, epsilon_upper = outward_float32(float(contract["epsilon"]))
    threshold_lower, threshold_upper = outward_float32(
        float(contract["threshold_d2"])
    )
    if D * 127 * 127 > min(2**24, np.iinfo(np.int32).max):
        raise RuntimeError("Exact INT32 accumulation/float32 conversion precondition failed")

    torch.cuda.init()
    compile_exact_specializations(
        float(epsilon_lower),
        float(epsilon_upper),
        float(threshold_lower),
        float(threshold_upper),
        float(contract["threshold_d2"]),
    )
    warm_source = torch.zeros((N, D), dtype=torch.float32, device="cuda")
    warm_metadata = prepare_device(warm_source)
    torch.cuda.synchronize()
    del warm_source, warm_metadata
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    metadata_before = metadata_artifacts(cache)
    cache_before = selected_cache_artifacts(cache)
    gpu_before = gpu_state(physical_gpu)
    run = {
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "record_id": args.record_id,
        "method": "tensorjoin_g5_audited_dynamic_router",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "physical_gpu": physical_gpu,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_state_before": gpu_before,
        "shape": [N, D],
        "source_dtype": "float32",
        "epsilon": contract["epsilon"],
        "epsilon_outward_float32": [float(epsilon_lower), float(epsilon_upper)],
        "threshold_d2": contract["threshold_d2"],
        "threshold_d2_outward_float32": [
            float(threshold_lower),
            float(threshold_upper),
        ],
        "source_npy_sha256": contract["source_npy_sha256"],
        "metadata_sha256": contract["metadata_sha256"],
        "smoke_summary_sha256": contract["smoke_summary_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "upstream_source_hashes": upstream_hashes,
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G5_COMPATIBILITY_SCREEN.md"),
        "timing_scope": "f32 pageable host -> sorted canonical u64 host",
        "excluded_before_timer": [
            "file_io",
            "process_startup",
            "cuda_context_initialization",
            "triton_compilation",
            "gpu_preflight",
        ],
        "excluded_after_timer": [
            "hashing",
            "correctness_checks",
            "json_write",
            "optional_pair_serialization",
        ],
        "tiles_per_batch": TILES_PER_BATCH,
        "buffer_capacity_per_stage": CAPACITY,
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
        "selected_cache_artifacts_before_timer": cache_before,
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    torch.cuda.reset_peak_memory_stats()
    public_started = time.perf_counter()

    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)

    device = torch.device("cuda:0")
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu, scales_gpu, norms_gpu, errors_gpu, codes_t_gpu = prepare_device(vectors_gpu)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    result_ids = torch.empty(CAPACITY, dtype=torch.int64, device=device)
    ambiguous_ids = torch.empty(CAPACITY, dtype=torch.int64, device=device)
    fp64_ids = torch.empty(CAPACITY, dtype=torch.int64, device=device)

    accepted_parts: list[np.ndarray] = []
    batch_records: list[dict[str, object]] = []
    sampled_fp64_pair_ids: list[int] = []
    totals = {
        "g3b_direct_accept_upper_pairs": 0,
        "g3b_ambiguous_upper_pairs": 0,
        "fp32_direct_accept_upper_pairs": 0,
        "fp32_direct_reject_upper_pairs": 0,
        "fp32_bitwise_equal_upper_pairs": 0,
        "fp64_refined_upper_pairs": 0,
        "accepted_upper_pairs": 0,
        "overflow_events": 0,
    }

    for batch_index, tile_start in enumerate(
        range(0, scheduled_tiles, TILES_PER_BATCH)
    ):
        tile_stop = min(tile_start + TILES_PER_BATCH, scheduled_tiles)
        batch_tiles = tile_stop - tile_start
        counters = torch.zeros(6, dtype=torch.int32, device=device)
        analytic_certificate_ragged_safe_i64[(batch_tiles,)](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            norms_gpu,
            errors_gpu,
            tile_rows_gpu[tile_start:tile_stop],
            tile_columns_gpu[tile_start:tile_stop],
            result_ids,
            ambiguous_ids,
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
        if args.phase == "compatibility" and fp64_count and len(sampled_fp64_pair_ids) < 64:
            sample_count = min(64 - len(sampled_fp64_pair_ids), fp64_count)
            sampled_fp64_pair_ids.extend(
                int(value)
                for value in fp64_ids[:sample_count].cpu().numpy().astype(np.int64)
            )

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
                CAPACITY=CAPACITY,
                BLOCK_K=FP64_BLOCK_K,
                num_warps=4,
            )
        final = counters.cpu().numpy().astype(np.int64)
        final_count = int(final[0])
        overflow = int(final[2])
        if max(direct_count, ambiguous_count, final_count, fp64_count) > CAPACITY:
            overflow += 1
        accepted_parts.append(
            result_ids[: min(final_count, CAPACITY)]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=True)
        )

        record = {
            "batch_index": batch_index,
            "tile_start": tile_start,
            "tile_stop": tile_stop,
            "scheduled_tiles": batch_tiles,
            "g3b_direct_accept_upper_pairs": direct_count,
            "g3b_ambiguous_upper_pairs": ambiguous_count,
            "fp32_direct_accept_upper_pairs": fp32_accept_count,
            "fp32_direct_reject_upper_pairs": fp32_reject_count,
            "fp32_bitwise_equal_upper_pairs": equality_count,
            "fp64_refined_upper_pairs": fp64_count,
            "accepted_upper_pairs": final_count,
            "overflow_events": overflow,
        }
        batch_records.append(record)
        for key in totals:
            totals[key] += int(record[key])
        if batch_index == 0 or (batch_index + 1) % 25 == 0 or tile_stop == scheduled_tiles:
            print(
                "BATCH_PROGRESS "
                + json.dumps(
                    {
                        "completed_batches": batch_index + 1,
                        "completed_tiles": tile_stop,
                        "total_tiles": scheduled_tiles,
                        "latest_stage_counts": record,
                    },
                    sort_keys=True,
                ),
                flush=True,
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

    metadata_after = metadata_artifacts(cache)
    cache_after = selected_cache_artifacts(cache)
    cache_unchanged = cache_after == cache_before and metadata_after == metadata_before
    correctness = validate_canonical(canonical, contract)
    correctness.update(
        {
            "upper_duplicates": int(
                np.count_nonzero(accepted_upper[1:] == accepted_upper[:-1])
            ),
            "invalid_or_lower_upper_pairs": int(
                np.count_nonzero(
                    (accepted_upper >= np.uint64(N) * np.uint64(N))
                    | (upper_rows > upper_columns)
                )
            ),
            "overflow_events": totals["overflow_events"],
        }
    )
    correctness["exact_contract_pass"] = bool(
        correctness["exact_contract_pass"]
        and correctness["upper_duplicates"] == 0
        and correctness["invalid_or_lower_upper_pairs"] == 0
        and totals["overflow_events"] == 0
        and cache_unchanged
    )

    pair_receipt: dict[str, object] | None = None
    if args.write_pairs:
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = pair_path.with_name(f".{pair_path.name}.tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        canonical.astype("<u8", copy=False).tofile(temporary)
        os.replace(temporary, pair_path)
        pair_receipt = {
            "path": str(pair_path.relative_to(PROJECT)),
            "bytes": pair_path.stat().st_size,
            "sha256": sha256_file(pair_path),
            "raw_u64_sha256": sha256_array(canonical, "<u8"),
        }

    gpu_after = gpu_state(physical_gpu)
    result = {
        **run,
        "measurement_status": (
            "compatibility_only_no_performance_claim"
            if args.phase == "compatibility"
            else "diagnostic_public_denominator_cheap_screen"
            if args.phase == "screen"
            else "formal_public_denominator_candidate"
        ),
        "public_seconds": public_seconds,
        "tile_extent": tile_extent,
        "scheduled_tiles": scheduled_tiles,
        "batch_count": len(batch_records),
        "batches": batch_records,
        **totals,
        "sampled_fp64_pair_ids_compatibility_only": sampled_fp64_pair_ids,
        "selected_cache_artifacts_after_timer": cache_after,
        "selected_cache_artifacts_unchanged_during_timer": cache_unchanged,
        "cache_path": str(cache.relative_to(PROJECT)),
        "metadata_artifacts_before": metadata_before,
        "metadata_artifacts_after": metadata_after,
        "metadata_source_sha256": sha256_file(PROJECT / "g16_gpu_preparation_20260905/src/gpu_preparation.py"),
        "pair_receipt": pair_receipt,
        "torch_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_rss_kib_after_run": max_rss_kib(),
        "gpu_state_after": gpu_after,
        "correctness": correctness,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if correctness["exact_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

