#!/usr/bin/env python3
"""Run same-shape adversarial/metamorphic tests for the G3C-B-R1 cascade."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
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
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    FP64_BLOCK_K,
    certified_fp32_filter_i64,
)


EXPERIMENT_ID = "tensorjoin_20260903_g3c_b_r1_gpu_cascade_adversarial"
CANDIDATE_SHA256 = "637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d"
EXPECTED_CUBIN_SHA256 = "db750295875f8363679cfe9f9762cef884e1ecd371e3e419fd500d91c3a20670"
SEED = 20_260_903


def normal_or_zero(values: np.ndarray) -> bool:
    values = np.asarray(values)
    return bool(
        np.all(np.isfinite(values))
        and np.all((values == 0) | (np.abs(values) >= np.finfo(values.dtype).tiny))
    )


def canonical_from_upper(upper: np.ndarray) -> np.ndarray:
    rows = upper // np.uint64(N)
    columns = upper - rows * np.uint64(N)
    reverse = columns[rows < columns] * np.uint64(N) + rows[rows < columns]
    return np.sort(np.concatenate((upper, reverse)).astype(np.uint64, copy=False))


def upper_from_canonical(canonical: np.ndarray) -> np.ndarray:
    rows = canonical // np.uint64(N)
    columns = canonical - rows * np.uint64(N)
    return np.sort(canonical[rows <= columns])


def remap_canonical(canonical: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    inverse = np.empty(N, dtype=np.uint64)
    inverse[permutation] = np.arange(N, dtype=np.uint64)
    old_rows = canonical // np.uint64(N)
    old_columns = canonical - old_rows * np.uint64(N)
    return np.sort(inverse[old_rows] * np.uint64(N) + inverse[old_columns])


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def run_case(
    name: str,
    vectors: np.ndarray,
    epsilon: float,
    threshold_d2: float,
    oracle: np.ndarray,
    device: torch.device,
    tile_rows_gpu: torch.Tensor,
    tile_columns_gpu: torch.Tensor,
    scheduled_tiles: int,
    capacity: int,
    buffers: dict[str, torch.Tensor],
) -> dict[str, object]:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.shape != (N, D) or not np.all(np.isfinite(vectors)):
        raise RuntimeError(f"{name}: invalid source domain")
    codes, scales, norms, errors = quantize_with_analytic_residual(vectors)
    if not all(normal_or_zero(x) for x in (scales, norms, errors)):
        raise RuntimeError(f"{name}: stored terms are not finite normal-or-zero")
    if D * 127 * 127 >= 2**24:
        raise RuntimeError("Integer exact-conversion precondition failed")
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    oracle = np.sort(np.asarray(oracle, dtype=np.uint64))
    oracle_upper = upper_from_canonical(oracle)

    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(norms).to(device)
    errors_gpu = torch.from_numpy(errors).to(device)
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
    g3b_scan = counters.cpu().numpy().astype(np.int64)
    g3b_direct_count = int(g3b_scan[0])
    g3b_ambiguous_count = int(g3b_scan[1])
    g3b_direct = sorted_ids(buffers["results"], min(g3b_direct_count, capacity))
    g3b_ambiguous = sorted_ids(
        buffers["ambiguous"], min(g3b_ambiguous_count, capacity)
    )
    g3b_direct_duplicates = duplicate_count(g3b_direct)
    g3b_ambiguous_duplicates = duplicate_count(g3b_ambiguous)
    g3b_direct_ambiguous_overlap = int(
        np.intersect1d(g3b_direct, g3b_ambiguous).size
    )
    g3b_false_accepts = np.setdiff1d(
        g3b_direct, oracle_upper, assume_unique=g3b_direct_duplicates == 0
    )
    g3b_false_rejects = np.setdiff1d(
        oracle_upper, np.union1d(g3b_direct, g3b_ambiguous), assume_unique=True
    )

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
    fp32_scan = counters.cpu().numpy().astype(np.int64)
    accepted_before_fp64_count = int(fp32_scan[0])
    fp32_accept_count = accepted_before_fp64_count - g3b_direct_count
    fp64_count = int(fp32_scan[3])
    fp32_reject_count = int(fp32_scan[4])
    bitwise_equal_count = int(fp32_scan[5])
    accepted_before_fp64 = sorted_ids(
        buffers["results"], min(accepted_before_fp64_count, capacity)
    )
    fp32_accept = np.setdiff1d(
        accepted_before_fp64, g3b_direct, assume_unique=True
    )
    fp64_pairs = sorted_ids(buffers["fp64"], min(fp64_count, capacity))
    rejected = np.setdiff1d(
        g3b_ambiguous,
        np.union1d(fp32_accept, fp64_pairs),
        assume_unique=True,
    )
    fp32_false_accepts = np.setdiff1d(
        fp32_accept, oracle_upper, assume_unique=True
    )
    fp32_false_rejects = np.intersect1d(rejected, oracle_upper, assume_unique=True)
    stage_partition_pass = bool(
        fp32_accept_count + fp32_reject_count + fp64_count == g3b_ambiguous_count
        and fp32_accept_count == int(fp32_accept.size)
        and fp32_reject_count == int(rejected.size)
    )

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
    final = counters.cpu().numpy().astype(np.int64)
    final_count = int(final[0])
    overflow = int(final[2])
    accepted_upper = sorted_ids(buffers["results"], min(final_count, capacity))
    final_duplicates = duplicate_count(accepted_upper)
    missing = np.setdiff1d(
        oracle_upper, accepted_upper, assume_unique=final_duplicates == 0
    )
    extra = np.setdiff1d(
        accepted_upper, oracle_upper, assume_unique=final_duplicates == 0
    )
    rows = accepted_upper // np.uint64(N)
    columns = accepted_upper - rows * np.uint64(N)
    invalid = int(
        np.count_nonzero((accepted_upper >= np.uint64(N * N)) | (rows > columns))
    )
    canonical = canonical_from_upper(accepted_upper)
    exact_match = bool(canonical.size == oracle.size and np.array_equal(canonical, oracle))
    case_pass = bool(
        g3b_direct_duplicates == 0
        and g3b_ambiguous_duplicates == 0
        and duplicate_count(fp32_accept) == 0
        and duplicate_count(fp64_pairs) == 0
        and final_duplicates == 0
        and g3b_direct_ambiguous_overlap == 0
        and g3b_false_accepts.size == 0
        and g3b_false_rejects.size == 0
        and fp32_false_accepts.size == 0
        and fp32_false_rejects.size == 0
        and stage_partition_pass
        and missing.size == 0
        and extra.size == 0
        and invalid == 0
        and overflow == 0
        and exact_match
    )
    result = {
        "case": name,
        "epsilon": epsilon,
        "threshold_d2": threshold_d2,
        "source_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "oracle_count": int(oracle.size),
        "oracle_sha256": sha256_u64(oracle),
        "max_abs_code": int(np.max(np.abs(codes.astype(np.int16)))),
        "max_code_norm": int(np.max(np.sum(codes.astype(np.int64) ** 2, axis=1))),
        "stored_terms_normal_or_zero": True,
        "g3b_direct_count": g3b_direct_count,
        "g3b_ambiguous_count": g3b_ambiguous_count,
        "fp32_accept_count": fp32_accept_count,
        "fp32_reject_count": fp32_reject_count,
        "fp32_bitwise_equal_count": bitwise_equal_count,
        "fp64_count": fp64_count,
        "g3b_direct_sha256": sha256_u64(g3b_direct),
        "g3b_ambiguous_sha256": sha256_u64(g3b_ambiguous),
        "fp32_accept_sha256": sha256_u64(fp32_accept),
        "fp32_reject_sha256": sha256_u64(rejected),
        "fp64_sha256": sha256_u64(fp64_pairs),
        "final_upper_count": final_count,
        "final_upper_sha256": sha256_u64(accepted_upper),
        "canonical_sha256": sha256_u64(canonical),
        "stage_partition_pass": stage_partition_pass,
        "unsafe_g3b_direct_accepts": int(g3b_false_accepts.size),
        "unsafe_g3b_direct_rejects": int(g3b_false_rejects.size),
        "unsafe_fp32_direct_accepts": int(fp32_false_accepts.size),
        "unsafe_fp32_direct_rejects": int(fp32_false_rejects.size),
        "missing_upper": int(missing.size),
        "extra_upper": int(extra.size),
        "invalid_upper": invalid,
        "overflow_events": overflow,
        "exact_match": exact_match,
        "case_pass": case_pass,
    }
    print("CASE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    del vectors_gpu, codes_gpu, codes_t_gpu, scales_gpu, norms_gpu, errors_gpu
    return result

def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Expected CUDA_VISIBLE_DEVICES=0")
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected host: {platform.node()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible CUDA device")
    candidate = PROJECT / "src/run_g3c_b_r1_gpu_cascade.py"
    protocol = PROJECT / "PROTOCOL_G3C_B_R1.md"
    result_path = PROJECT / "results/g3c_b_r1_adversarial.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if sha256_file(candidate) != CANDIDATE_SHA256:
        raise RuntimeError("Frozen candidate hash mismatch")
    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    if sha256_file(vector_path) != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Vector hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")
    base = np.ascontiguousarray(np.load(vector_path, allow_pickle=False), dtype=np.float32)
    base_oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if sha256_u64(base_oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Oracle raw hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    epsilon = float(metadata["radius"]["epsilon"])
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    rng = np.random.default_rng(SEED)
    row_permutation = rng.permutation(N)
    dimension_permutation = rng.permutation(D)
    dimension_signs = np.where(np.arange(D) % 2 == 0, 1.0, -1.0).astype(np.float32)

    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)
    capacity = scheduled_tiles * BLOCK_M * BLOCK_N
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    buffers = {
        "results": torch.empty(capacity, dtype=torch.int64, device=device),
        "ambiguous": torch.empty(capacity, dtype=torch.int64, device=device),
        "fp64": torch.empty(capacity, dtype=torch.int64, device=device),
        "counters": torch.zeros(6, dtype=torch.int32, device=device),
    }
    torch.cuda.synchronize()

    cases: list[dict[str, object]] = []
    metamorphic = [
        ("negate", np.ascontiguousarray(-base), epsilon, threshold_d2, base_oracle),
        (
            "row_permutation",
            np.ascontiguousarray(base[row_permutation]),
            epsilon,
            threshold_d2,
            remap_canonical(base_oracle, row_permutation),
        ),
        (
            "dimension_permutation_sign_flip",
            np.ascontiguousarray(base[:, dimension_permutation] * dimension_signs),
            epsilon,
            threshold_d2,
            base_oracle,
        ),
        (
            "scale_2^-8",
            np.ascontiguousarray(base * np.float32(2.0**-8)),
            epsilon * 2.0**-8,
            threshold_d2 * 2.0**-16,
            base_oracle,
        ),
        (
            "scale_2^8",
            np.ascontiguousarray(base * np.float32(2.0**8)),
            epsilon * 2.0**8,
            threshold_d2 * 2.0**16,
            base_oracle,
        ),
    ]
    for args in metamorphic:
        cases.append(
            run_case(
                *args,
                device,
                tile_rows_gpu,
                tile_columns_gpu,
                scheduled_tiles,
                capacity,
                buffers,
            )
        )

    all_directed = np.arange(N * N, dtype=np.uint64)
    cases.append(
        run_case(
            "all_zero_ftz_boundary",
            np.zeros((N, D), dtype=np.float32),
            2.0**-20,
            2.0**-40,
            all_directed,
            device,
            tile_rows_gpu,
            tile_columns_gpu,
            scheduled_tiles,
            capacity,
            buffers,
        )
    )
    half = N // 2
    signs = np.ones((N, 1), dtype=np.float32)
    signs[half:] = -1.0
    max_vectors = np.broadcast_to(signs, (N, D)).copy()
    first = (
        np.arange(half, dtype=np.uint64)[:, None] * np.uint64(N)
        + np.arange(half, dtype=np.uint64)[None, :]
    ).reshape(-1)
    second_ids = np.arange(half, N, dtype=np.uint64)
    second = (second_ids[:, None] * np.uint64(N) + second_ids[None, :]).reshape(-1)
    max_oracle = np.concatenate((first, second))
    cases.append(
        run_case(
            "signed_max_accumulator",
            max_vectors,
            0.0,
            0.0,
            max_oracle,
            device,
            tile_rows_gpu,
            tile_columns_gpu,
            scheduled_tiles,
            capacity,
            buffers,
        )
    )

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del buffers, tile_rows_gpu, tile_columns_gpu
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    scale_down = next(case for case in cases if case["case"] == "scale_2^-8")
    scale_down_selectivity_pass = bool(scale_down["fp64_count"] <= 10_326)
    gate_pass = bool(
        len(cases) == 7
        and all(case["case_pass"] for case in cases)
        and scale_down_selectivity_pass
        and allocated_after == 0
        and reserved_after == 0
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "adversarial_correctness_only_no_performance_claim",
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [N, D],
        "seed": SEED,
        "case_count": len(cases),
        "cases": cases,
        "scale_2^-8_maximum_fp64_pairs": 10_326,
        "scale_2^-8_observed_fp64_pairs": int(scale_down["fp64_count"]),
        "scale_2^-8_selectivity_pass": scale_down_selectivity_pass,
        "candidate_sha256": sha256_file(candidate),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol),
        "expected_cubin_sha256": EXPECTED_CUBIN_SHA256,
        "torch_peak_memory_allocated_bytes": peak_allocated,
        "torch_peak_memory_reserved_bytes": peak_reserved,
        "torch_memory_allocated_after_cleanup_bytes": allocated_after,
        "torch_memory_reserved_after_cleanup_bytes": reserved_after,
        "safety_pass": gate_pass,
        "performance_claim_allowed": False,
    }
    atomic_json(result_path, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
