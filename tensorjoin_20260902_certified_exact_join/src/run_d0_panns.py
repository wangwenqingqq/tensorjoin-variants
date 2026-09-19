#!/usr/bin/env python3
"""Run the frozen CPU-only PANNs certificate-geometry breadth test."""

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


EXPERIMENT_ID = "tensorjoin_20260902_panns_certificate_d0"
SEED = 20260902
QUERY_COUNT = 512
BASE_COUNT = 4096
DIMENSION = 2048
TARGET_RESULTS_PER_QUERY = (1, 8, 64)
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


def disjoint_clip_sample(
    features: np.ndarray, clips: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    unique_clips = np.unique(clips)
    permuted_clips = rng.permutation(unique_clips)
    query_indices: list[int] = []
    base_indices: list[int] = []
    cursor = 0
    while len(query_indices) < QUERY_COUNT:
        query_indices.extend(np.flatnonzero(clips == permuted_clips[cursor]).tolist())
        cursor += 1
    while len(base_indices) < BASE_COUNT:
        base_indices.extend(np.flatnonzero(clips == permuted_clips[cursor]).tolist())
        cursor += 1
    query_indices = rng.permutation(query_indices)[:QUERY_COUNT]
    base_indices = rng.permutation(base_indices)[:BASE_COUNT]
    query_clips = clips[query_indices]
    base_clips = clips[base_indices]
    if len(np.intersect1d(np.unique(query_clips), np.unique(base_clips))):
        raise AssertionError("Query/base clip leakage")
    return (
        np.ascontiguousarray(features[query_indices]),
        np.ascontiguousarray(features[base_indices]),
        query_clips,
        base_clips,
    )


def squared_distances_float64(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    # Direct differences avoid catastrophic cancellation for bitwise-identical
    # and near-identical normalized embeddings. Keep the temporary below 300 MB.
    distances = np.empty((len(q), len(x)), dtype=np.float64)
    block_queries = 4
    for start in range(0, len(q), block_queries):
        stop = min(start + block_queries, len(q))
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
    residual_norm = np.nextafter(np.linalg.norm(residual, axis=1), np.inf)
    return codes, scales, residual_norm


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
    scale_products = q_scales.astype(np.float64)[:, None] * x_scales.astype(np.float64)[None, :]
    distances = q_norm_sq[:, None] + x_norm_sq[None, :] - 2.0 * int_dot * scale_products
    np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances)


def evaluate_threshold(
    target: int,
    threshold_d2: float,
    exact_d2: np.ndarray,
    reconstructed_d: np.ndarray,
    q_error: np.ndarray,
    x_error: np.ndarray,
) -> dict[str, int | float]:
    radius = q_error[:, None] + x_error[None, :]
    lower = np.nextafter(np.maximum(0.0, reconstructed_d - radius) ** 2, -np.inf)
    upper = np.nextafter((reconstructed_d + radius) ** 2, np.inf)
    exact_inside = exact_d2 <= threshold_d2
    accept = upper <= threshold_d2
    reject = lower > threshold_d2
    ambiguous = ~(accept | reject)
    final_inside = accept | (ambiguous & exact_inside)
    exact_count = int(np.count_nonzero(exact_inside))
    ambiguous_count = int(np.count_nonzero(ambiguous))
    row = {
        "target_results_per_query": target,
        "threshold_d2": threshold_d2,
        "exact_pairs": exact_count,
        "exact_results_per_query": exact_count / exact_d2.shape[0],
        "direct_accept_pairs": int(np.count_nonzero(accept)),
        "direct_reject_pairs": int(np.count_nonzero(reject)),
        "ambiguous_pairs": ambiguous_count,
        "ambiguous_fraction": ambiguous_count / exact_d2.size,
        "ambiguous_to_output": ambiguous_count / max(exact_count, 1),
        "lower_containment_violations": int(np.count_nonzero(exact_d2 + CONTAINMENT_GUARD < lower)),
        "upper_containment_violations": int(np.count_nonzero(exact_d2 - CONTAINMENT_GUARD > upper)),
        "false_accept_pairs": int(np.count_nonzero(accept & ~exact_inside)),
        "false_reject_pairs": int(np.count_nonzero(reject & exact_inside)),
        "final_classification_mismatch": int(np.count_nonzero(final_inside != exact_inside)),
    }
    print("THRESHOLD " + json.dumps(row, sort_keys=True), flush=True)
    return row


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("D0 certificate evaluation must be CPU-only")
    feature_cache = args.feature_cache.resolve()
    if args.output.exists() or args.threshold_output.exists():
        raise FileExistsError("Refusing to overwrite an existing D0 result")
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "seed": SEED,
                "feature_cache": str(feature_cache),
                "feature_cache_sha256": sha256_file(feature_cache),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start = time.time()
    with np.load(feature_cache, allow_pickle=False) as data:
        features = np.ascontiguousarray(data["features"], dtype=np.float32)
        clips = data["clip"]
    if features.shape != (10_000, DIMENSION):
        raise ValueError(features.shape)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature cache")
    query, base, query_clips, base_clips = disjoint_clip_sample(features, clips)
    print(
        f"SPLIT query_shape={query.shape} base_shape={base.shape} "
        f"query_clips={len(np.unique(query_clips))} base_clips={len(np.unique(base_clips))}",
        flush=True,
    )
    oracle_start = time.time()
    exact_d2 = squared_distances_float64(query, base)
    print(f"ORACLE elapsed_s={time.time()-oracle_start:.6f}", flush=True)
    q_codes, q_scales, q_error = quantize_per_vector(query)
    x_codes, x_scales, x_error = quantize_per_vector(base)
    int_start = time.time()
    reconstructed_d = reconstructed_distances(q_codes, q_scales, x_codes, x_scales)
    print(f"INT_DISTANCE elapsed_s={time.time()-int_start:.6f}", flush=True)
    flat = exact_d2.reshape(-1)
    rows = []
    for target in TARGET_RESULTS_PER_QUERY:
        desired_count = QUERY_COUNT * target
        threshold = float(np.partition(flat, desired_count - 1)[desired_count - 1])
        rows.append(
            evaluate_threshold(target, threshold, exact_d2, reconstructed_d, q_error, x_error)
        )
    ambiguity = [float(row["ambiguous_fraction"]) for row in rows]
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
    ambiguity_pass = max(ambiguity) <= 0.05 and float(np.median(ambiguity)) <= 0.01
    accumulator_worst_case = DIMENSION * 127 * 127
    accumulator_pass = accumulator_worst_case <= np.iinfo(np.int32).max
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "seed": SEED,
        "query_shape": list(query.shape),
        "base_shape": list(base.shape),
        "query_unique_clips": int(len(np.unique(query_clips))),
        "base_unique_clips": int(len(np.unique(base_clips))),
        "clip_overlap": int(len(np.intersect1d(np.unique(query_clips), np.unique(base_clips)))),
        "feature_cache": str(feature_cache),
        "feature_cache_sha256": sha256_file(feature_cache),
        "script_sha256": sha256_file(Path(__file__)),
        "oracle_method": "blocked_direct_fp64_difference",
        "accumulator_worst_case": accumulator_worst_case,
        "accumulator_pass": bool(accumulator_pass),
        "max_ambiguity_fraction": max(ambiguity),
        "median_ambiguity_fraction": float(np.median(ambiguity)),
        "exactness_pass": bool(exactness_pass),
        "ambiguity_pass": bool(ambiguity_pass),
        "gate_pass": bool(exactness_pass and ambiguity_pass and accumulator_pass),
        "thresholds": rows,
        "elapsed_s": time.time() - start,
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
