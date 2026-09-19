#!/usr/bin/env python3
"""Run the frozen CPU-only UCF101 R3D-18 certificate geometry gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_ucf101_r3d18_geometry_d2a"
SEED = 20260903
QUERY_COUNT = 512
BASE_COUNT = 4096
DIMENSION = 512
TARGET_RESULTS_PER_QUERY = (1, 64)
BOUND_PAD = 1e-4
CONTAINMENT_GUARD = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def group_disjoint_sample(
    features: np.ndarray, clips: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    permuted_groups = rng.permutation(unique_groups)
    query_indices: list[int] = []
    base_indices: list[int] = []
    cursor = 0
    while len(query_indices) < QUERY_COUNT:
        query_indices.extend(np.flatnonzero(groups == permuted_groups[cursor]).tolist())
        cursor += 1
    while len(base_indices) < BASE_COUNT:
        base_indices.extend(np.flatnonzero(groups == permuted_groups[cursor]).tolist())
        cursor += 1
    query_indices = rng.permutation(query_indices)[:QUERY_COUNT]
    base_indices = rng.permutation(base_indices)[:BASE_COUNT]
    if len(np.intersect1d(groups[query_indices], groups[base_indices])):
        raise AssertionError("Query/base source-group leakage")
    if len(np.intersect1d(clips[query_indices], clips[base_indices])):
        raise AssertionError("Query/base clip leakage")
    return (
        np.ascontiguousarray(features[query_indices], dtype=np.float32),
        np.ascontiguousarray(features[base_indices], dtype=np.float32),
        clips[query_indices],
        clips[base_indices],
        groups[query_indices],
        groups[base_indices],
    )


def squared_distances_float64(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    distances = np.empty((len(q), len(x)), dtype=np.float64)
    for start in range(0, len(q), 16):
        stop = min(start + 16, len(q))
        difference = q[start:stop, None, :] - x[None, :, :]
        distances[start:stop] = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )
    return distances


def quantize_per_vector(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = (codes.astype(np.float32) * scales[:, None]).astype(np.float32)
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    errors = np.nextafter(np.linalg.norm(residual, axis=1), np.inf)
    return codes, scales, errors


def reconstructed_distances(
    q_codes: np.ndarray,
    q_scales: np.ndarray,
    x_codes: np.ndarray,
    x_scales: np.ndarray,
) -> np.ndarray:
    int_dot = q_codes.astype(np.int32) @ x_codes.astype(np.int32).T
    q_sum_sq = np.sum(q_codes.astype(np.int64) ** 2, axis=1)
    x_sum_sq = np.sum(x_codes.astype(np.int64) ** 2, axis=1)
    q_norm_sq = q_sum_sq.astype(np.float64) * q_scales.astype(np.float64) ** 2
    x_norm_sq = x_sum_sq.astype(np.float64) * x_scales.astype(np.float64) ** 2
    scales = q_scales.astype(np.float64)[:, None] * x_scales.astype(np.float64)[None, :]
    distance_d2 = q_norm_sq[:, None] + x_norm_sq[None, :] - 2.0 * int_dot * scales
    np.maximum(distance_d2, 0.0, out=distance_d2)
    return np.sqrt(distance_d2)


def midgap(exact_d2: np.ndarray, target: int) -> dict[str, float]:
    flat = exact_d2.reshape(-1)
    rank = QUERY_COUNT * target - 1
    lower = float(np.partition(flat, rank)[rank])
    greater = flat[flat > lower]
    if not len(greater):
        raise ValueError("No strictly greater distance")
    upper = float(np.min(greater))
    return {
        "lower_distance_d2": lower,
        "upper_distance_d2": upper,
        "gap_d2": upper - lower,
        "threshold_d2": lower + (upper - lower) / 2.0,
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("D2A must be CPU-only")
    if args.output.exists() or args.threshold_output.exists():
        raise FileExistsError("Refusing to overwrite D2A evidence")
    cache = args.feature_cache.resolve()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "seed": SEED,
                "feature_cache": str(cache),
                "feature_cache_sha256": sha256_file(cache),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start_time = time.time()
    with np.load(cache, allow_pickle=False) as data:
        features = np.ascontiguousarray(data["features"], dtype=np.float32)
        clips = data["clip"]
        groups = data["group"]
    if features.ndim != 2 or features.shape[1] != DIMENSION or len(features) < 4_608:
        raise ValueError(features.shape)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature cache")
    query, base, query_clips, base_clips, query_groups, base_groups = group_disjoint_sample(
        features, clips, groups
    )
    print(
        f"SPLIT query_shape={query.shape} base_shape={base.shape} "
        f"query_groups={len(np.unique(query_groups))} "
        f"base_groups={len(np.unique(base_groups))}",
        flush=True,
    )
    oracle_start = time.time()
    exact_d2 = squared_distances_float64(query, base)
    print(f"ORACLE elapsed_s={time.time()-oracle_start:.6f}", flush=True)
    q_codes, q_scales, q_errors = quantize_per_vector(query)
    x_codes, x_scales, x_errors = quantize_per_vector(base)
    reconstructed_d = reconstructed_distances(q_codes, q_scales, x_codes, x_scales)
    radius = q_errors[:, None] + x_errors[None, :] + BOUND_PAD
    lower_d2 = np.nextafter(np.maximum(reconstructed_d - radius, 0.0) ** 2, -np.inf)
    upper_d2 = np.nextafter((reconstructed_d + radius) ** 2, np.inf)

    rows: list[dict[str, int | float]] = []
    for target in TARGET_RESULTS_PER_QUERY:
        threshold_info = midgap(exact_d2, target)
        threshold = threshold_info["threshold_d2"]
        exact_inside = exact_d2 <= threshold
        accept = upper_d2 <= threshold
        reject = lower_d2 > threshold
        ambiguous = ~(accept | reject)
        final_inside = accept | (ambiguous & exact_inside)
        exact_count = int(np.count_nonzero(exact_inside))
        ambiguous_count = int(np.count_nonzero(ambiguous))
        row = {
            "target_results_per_query": target,
            **threshold_info,
            "exact_pairs": exact_count,
            "exact_results_per_query": exact_count / QUERY_COUNT,
            "direct_accept_pairs": int(np.count_nonzero(accept)),
            "direct_reject_pairs": int(np.count_nonzero(reject)),
            "ambiguous_pairs": ambiguous_count,
            "ambiguous_fraction": ambiguous_count / exact_d2.size,
            "ambiguous_to_output": ambiguous_count / max(exact_count, 1),
            "lower_containment_violations": int(
                np.count_nonzero(exact_d2 + CONTAINMENT_GUARD < lower_d2)
            ),
            "upper_containment_violations": int(
                np.count_nonzero(exact_d2 - CONTAINMENT_GUARD > upper_d2)
            ),
            "false_accept_pairs": int(np.count_nonzero(accept & ~exact_inside)),
            "false_reject_pairs": int(np.count_nonzero(reject & exact_inside)),
            "final_classification_mismatch": int(np.count_nonzero(final_inside != exact_inside)),
        }
        rows.append(row)
        print("THRESHOLD " + json.dumps(row, sort_keys=True), flush=True)

    exactness_pass = all(
        row[key] == 0
        for row in rows
        for key in (
            "lower_containment_violations",
            "upper_containment_violations",
            "false_accept_pairs",
            "false_reject_pairs",
            "final_classification_mismatch",
        )
    )
    ambiguity_pass = all(float(row["ambiguous_fraction"]) <= 0.10 for row in rows)
    accumulator_worst_case = DIMENSION * 127 * 127
    accumulator_pass = accumulator_worst_case <= np.iinfo(np.int32).max
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "seed": SEED,
        "query_shape": list(query.shape),
        "base_shape": list(base.shape),
        "query_unique_groups": int(len(np.unique(query_groups))),
        "base_unique_groups": int(len(np.unique(base_groups))),
        "group_overlap": int(len(np.intersect1d(query_groups, base_groups))),
        "clip_overlap": int(len(np.intersect1d(query_clips, base_clips))),
        "feature_cache": str(cache),
        "feature_cache_sha256": sha256_file(cache),
        "script_sha256": sha256_file(Path(__file__)),
        "bound_pad": BOUND_PAD,
        "oracle_method": "blocked_direct_fp64_difference",
        "accumulator_worst_case": accumulator_worst_case,
        "accumulator_pass": bool(accumulator_pass),
        "max_ambiguity_fraction": max(float(row["ambiguous_fraction"]) for row in rows),
        "exactness_pass": bool(exactness_pass),
        "ambiguity_pass": bool(ambiguity_pass),
        "gate_pass": bool(exactness_pass and ambiguity_pass and accumulator_pass),
        "thresholds": rows,
        "elapsed_s": time.time() - start_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.threshold_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("RUN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
