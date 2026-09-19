#!/usr/bin/env python3
"""Run one full-Cifar triangular TensorJoin G2B correctness/resource smoke."""

from __future__ import annotations

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

from run_d1_triton import quantize_per_vector, upward_float32
from run_g2a_tensorjoin import (
    filter_ambiguous_fp32_i64,
    refine_ambiguous_fp64_i64,
)
from run_g2a2_tensorjoin_triangular import triangular_int8_certificate_compact_i64


EXPERIMENT_ID = "tensorjoin_20260903_tensorjoin_cifar60000_g2b_smoke"
N = 60_000
D = 512
BOUND_PAD = 1e-4
FP32_DISTANCE_GUARD = 1e-3
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
REFINE_BLOCK_K = 256
TILES_PER_BATCH = 4_096
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2b_cifar60000"
RESULT = PROJECT / "results/g2b_tensorjoin_smoke.json"
PAIR_FILE = PROJECT / "artifacts/g2b/tensorjoin_pairs_u64_le.bin"
GDS_PAIR_FILE = PROJECT / "artifacts/g2b/gds_pairs_u64_le.bin"


def sha256_file(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u8").tobytes(order="C")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_pair_record(pair_id: int, vectors: np.ndarray, threshold_d2: float) -> dict[str, object]:
    row = pair_id // N
    column = pair_id - row * N
    delta = vectors[row].astype(np.float64) - vectors[column].astype(np.float64)
    distance_d2 = float(np.dot(delta, delta))
    return {
        "pair_id": pair_id,
        "row": row,
        "column": column,
        "distance_d2": distance_d2,
        "threshold_d2": threshold_d2,
        "distance_d2_minus_threshold": distance_d2 - threshold_d2,
        "inside_direct_fp64_contract": distance_d2 <= threshold_d2,
    }


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            f"G2B TensorJoin smoke requires CUDA_VISIBLE_DEVICES=0, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    temporary_pair = PAIR_FILE.with_name(f".{PAIR_FILE.name}.tmp.{os.getpid()}")
    for path in (RESULT, PAIR_FILE, temporary_pair):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    for path in (DATA_DIR / "vectors_f32.npy", DATA_DIR / "metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    total_started = time.perf_counter()
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    threshold_d2 = float(metadata["effective_epsilon_d2"])
    expected_count = int(metadata["diagnostic_expected_exact_count"]["value"])
    vectors = np.ascontiguousarray(
        np.load(DATA_DIR / "vectors_f32.npy", allow_pickle=False), dtype=np.float32
    )
    if vectors.shape != (N, D):
        raise ValueError(vectors.shape)

    codes, scales, errors = quantize_per_vector(vectors)
    reconstructed_norm2 = (
        np.sum(codes.astype(np.int64) ** 2, axis=1).astype(np.float64)
        * scales.astype(np.float64) ** 2
    )
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = tile_extent * (tile_extent + 1) // 2
    if tile_rows.size != scheduled_tiles:
        raise AssertionError("Triangular tile schedule size mismatch")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(reconstructed_norm2.astype(np.float32)).to(device)
    errors_gpu = torch.from_numpy(upward_float32(errors)).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)

    capacity = TILES_PER_BATCH * BLOCK_M * BLOCK_N
    result_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    ambiguous_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    fp64_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    run = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": visible,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "shape": [N, D],
        "epsilon": float(metadata["epsilon"]),
        "threshold_d2": threshold_d2,
        "input_sha256": sha256_file(DATA_DIR / "vectors_f32.npy"),
        "metadata_sha256": sha256_file(DATA_DIR / "metadata.json"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "triangular_kernel_source_sha256": sha256_file(
            PROJECT / "src/run_g2a2_tensorjoin_triangular.py"
        ),
        "refinement_source_sha256": sha256_file(PROJECT / "src/run_g2a_tensorjoin.py"),
        "quantization_source_sha256": sha256_file(PROJECT / "src/run_d1_triton.py"),
        "diagnostic_expected_count": expected_count,
        "tile_extent": tile_extent,
        "scheduled_tiles": scheduled_tiles,
        "tiles_per_batch": TILES_PER_BATCH,
        "buffer_capacity_per_stage": capacity,
        "capacity_rule": "tiles_per_batch * 64 * 64 worst-case upper-pair slots",
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    accepted_parts: list[np.ndarray] = []
    batch_records: list[dict[str, int]] = []
    total_direct = 0
    total_ambiguous = 0
    total_fp32_accept = 0
    total_fp64 = 0
    total_equality = 0
    total_overflow = 0
    search_started = time.perf_counter()
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
                threshold_d2,
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
                threshold_d2,
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
        accepted = (
            result_ids[: min(final_count, capacity)]
            .cpu()
            .numpy()
            .astype(np.uint64, copy=True)
        )
        accepted_parts.append(accepted)
        total_direct += direct_count
        total_ambiguous += ambiguous_count
        total_fp32_accept += after_fp32_count - direct_count
        total_fp64 += fp64_count
        total_equality += equality_count
        total_overflow += overflow
        record = {
            "batch_index": batch_index,
            "tile_start": tile_start,
            "tile_stop": tile_stop,
            "scheduled_tiles": batch_tiles,
            "worst_case_slots": batch_tiles * BLOCK_M * BLOCK_N,
            "direct_accept_upper_pairs": direct_count,
            "ambiguous_upper_pairs": ambiguous_count,
            "fp32_accept_upper_pairs": after_fp32_count - direct_count,
            "fp64_refine_upper_pairs": fp64_count,
            "final_accept_upper_pairs": final_count,
            "overflow_events": overflow,
        }
        batch_records.append(record)
        print("BATCH_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
    diagnostic_search_wall_s = time.perf_counter() - search_started

    accepted_upper = np.sort(np.concatenate(accepted_parts).astype(np.uint64, copy=False))
    upper_duplicates = int(np.count_nonzero(accepted_upper[1:] == accepted_upper[:-1]))
    upper_rows = accepted_upper // np.uint64(N)
    upper_columns = accepted_upper - upper_rows * np.uint64(N)
    invalid_upper = int(
        np.count_nonzero(
            (accepted_upper >= np.uint64(N) * np.uint64(N))
            | (upper_rows > upper_columns)
        )
    )
    nonself = upper_rows < upper_columns
    reverse = upper_columns[nonself] * np.uint64(N) + upper_rows[nonself]
    canonical = np.sort(np.concatenate((accepted_upper, reverse)).astype(np.uint64))
    duplicate_pairs = int(np.count_nonzero(canonical[1:] == canonical[:-1]))
    self_pairs = int(
        np.count_nonzero(canonical // np.uint64(N) == canonical % np.uint64(N))
    )
    reverse_all = (canonical % np.uint64(N)) * np.uint64(N) + canonical // np.uint64(N)
    symmetry_missing = int(np.setdiff1d(reverse_all, canonical, assume_unique=True).size)

    gds_comparison: dict[str, object]
    candidate_truth_disagreements = None
    if GDS_PAIR_FILE.is_file():
        gds = np.fromfile(GDS_PAIR_FILE, dtype="<u8")
        if gds.size and np.any(gds[1:] < gds[:-1]):
            raise AssertionError("GDS comparator file is not sorted")
        gds_only = np.setdiff1d(gds, canonical, assume_unique=True)
        candidate_only = np.setdiff1d(canonical, gds, assume_unique=True)
        gds_only_records = [
            cpu_pair_record(int(pair_id), vectors, threshold_d2) for pair_id in gds_only
        ]
        candidate_only_records = [
            cpu_pair_record(int(pair_id), vectors, threshold_d2)
            for pair_id in candidate_only
        ]
        tensorjoin_false_negatives = sum(
            bool(record["inside_direct_fp64_contract"]) for record in gds_only_records
        )
        tensorjoin_false_positives = sum(
            not bool(record["inside_direct_fp64_contract"])
            for record in candidate_only_records
        )
        gds_false_positives = sum(
            not bool(record["inside_direct_fp64_contract"])
            for record in gds_only_records
        )
        gds_false_negatives = sum(
            bool(record["inside_direct_fp64_contract"])
            for record in candidate_only_records
        )
        candidate_truth_disagreements = (
            tensorjoin_false_negatives + tensorjoin_false_positives
        )
        gds_comparison = {
            "gds_pair_file_sha256": sha256_file(GDS_PAIR_FILE),
            "gds_canonical_pair_count": int(gds.size),
            "gds_only_pairs": int(gds_only.size),
            "tensorjoin_only_pairs": int(candidate_only.size),
            "gds_only_direct_fp64_records": gds_only_records,
            "tensorjoin_only_direct_fp64_records": candidate_only_records,
            "tensorjoin_false_negatives_by_direct_fp64": tensorjoin_false_negatives,
            "tensorjoin_false_positives_by_direct_fp64": tensorjoin_false_positives,
            "gds_false_positives_by_direct_fp64": gds_false_positives,
            "gds_false_negatives_by_direct_fp64": gds_false_negatives,
        }
    else:
        gds_comparison = {"status": "GDS pair file unavailable"}

    count_match = canonical.size == expected_count
    structural_pass = (
        upper_duplicates == 0
        and invalid_upper == 0
        and duplicate_pairs == 0
        and self_pairs == N
        and symmetry_missing == 0
        and total_overflow == 0
    )
    provisional_contract_pass = (
        structural_pass
        and count_match
        and candidate_truth_disagreements == 0
    )
    PAIR_FILE.parent.mkdir(parents=True, exist_ok=True)
    canonical.astype("<u8", copy=False).tofile(temporary_pair)
    os.replace(temporary_pair, PAIR_FILE)
    result = {
        **run,
        "measurement_status": "full_correctness_resource_smoke_no_performance_claim",
        "oracle_status": "provisional_until_independent_mistic_full_hash",
        "bound_pad": BOUND_PAD,
        "fp32_distance_guard": FP32_DISTANCE_GUARD,
        "analytical_upper_comparisons": N * (N + 1) // 2,
        "batch_count": len(batch_records),
        "batches": batch_records,
        "direct_accept_upper_pairs": total_direct,
        "ambiguous_upper_pairs": total_ambiguous,
        "fp32_accept_upper_pairs": total_fp32_accept,
        "fp64_refine_upper_pairs": total_fp64,
        "bitwise_equal_ambiguous_upper_pairs": total_equality,
        "overflow_events": total_overflow,
        "accepted_upper_pair_count": int(accepted_upper.size),
        "accepted_upper_raw_u64_sha256": sha256_u64(accepted_upper),
        "upper_duplicate_pairs": upper_duplicates,
        "invalid_or_lower_triangle_upper_ids": invalid_upper,
        "canonical_pair_count": int(canonical.size),
        "canonical_raw_u64_sha256": sha256_u64(canonical),
        "pair_file": str(PAIR_FILE.relative_to(PROJECT)),
        "pair_file_bytes": PAIR_FILE.stat().st_size,
        "pair_file_sha256": sha256_file(PAIR_FILE),
        "duplicate_pairs": duplicate_pairs,
        "self_pairs": self_pairs,
        "symmetry_missing_pairs": symmetry_missing,
        "diagnostic_count_match": count_match,
        "gds_comparison": gds_comparison,
        "structural_smoke_pass": structural_pass,
        "provisional_direct_fp64_contract_pass": provisional_contract_pass,
        "torch_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "diagnostic_search_wall_s": diagnostic_search_wall_s,
        "diagnostic_total_wall_s": time.perf_counter() - total_started,
    }
    atomic_json(RESULT, result)
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if provisional_contract_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
