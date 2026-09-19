#!/usr/bin/env python3
"""Run the frozen CPU-only H0 multi-vector certificate opportunity test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_multivector_certificate_h0"
SEED = 20260903
QUERY_OBJECTS = 128
BASE_OBJECTS = 512
TARGETS = (1, 8)
CONTAINMENT_GUARD = 2e-12
MAX_OBJECT_AMBIGUITY = 0.05
MAX_GLOBAL_REFINEMENT = 0.025


@dataclass(frozen=True)
class ObjectBatch:
    vectors: np.ndarray
    object_ids: np.ndarray
    slices: tuple[slice, ...]
    token_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_once(vectors: np.ndarray) -> np.ndarray:
    vectors64 = vectors.astype(np.float64)
    norms = np.sqrt(np.sum(vectors64 * vectors64, axis=1, dtype=np.float64))
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ValueError("H0 requires finite, nonzero token vectors")
    normalized = (vectors64 / norms[:, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized token")
    return np.ascontiguousarray(normalized)


def build_object_batch(
    vectors: np.ndarray, ids: np.ndarray, selected_ids: np.ndarray
) -> ObjectBatch:
    chunks: list[np.ndarray] = []
    slices: list[slice] = []
    token_counts: list[int] = []
    cursor = 0
    for object_id in selected_ids:
        indices = np.flatnonzero(ids == object_id)
        if len(indices) == 0:
            raise AssertionError(f"Empty object {object_id!r}")
        chunk = np.ascontiguousarray(vectors[indices])
        chunks.append(chunk)
        start = cursor
        cursor += len(chunk)
        slices.append(slice(start, cursor))
        token_counts.append(len(chunk))
    return ObjectBatch(
        vectors=np.ascontiguousarray(np.concatenate(chunks, axis=0)),
        object_ids=np.asarray(selected_ids),
        slices=tuple(slices),
        token_counts=np.asarray(token_counts, dtype=np.int32),
    )


def deterministic_split(
    vectors: np.ndarray, ids: np.ndarray, rng: np.random.Generator
) -> tuple[ObjectBatch, ObjectBatch]:
    unique_ids = np.unique(ids)
    if len(unique_ids) < QUERY_OBJECTS + BASE_OBJECTS:
        raise ValueError(f"Only {len(unique_ids)} objects")
    selected = rng.permutation(unique_ids)[: QUERY_OBJECTS + BASE_OBJECTS]
    query_ids = selected[:QUERY_OBJECTS]
    base_ids = selected[QUERY_OBJECTS:]
    if len(np.intersect1d(query_ids, base_ids)):
        raise AssertionError("Query/base object leakage")
    return (
        build_object_batch(vectors, ids, query_ids),
        build_object_batch(vectors, ids, base_ids),
    )


def direct_squared_distances(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    query64 = query.astype(np.float64)
    base64 = base.astype(np.float64)
    distances = np.empty((len(query), len(base)), dtype=np.float64)
    # Keep the largest FP64 difference temporary below roughly 180 MiB.
    block_queries = max(1, min(4, (180 << 20) // max(8 * len(base) * query.shape[1], 1)))
    for start in range(0, len(query), block_queries):
        stop = min(start + block_queries, len(query))
        difference = query64[start:stop, None, :] - base64[None, :, :]
        distances[start:stop] = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )
    return distances


def quantize_per_vector(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0.0, np.float32(1.0), scales).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = (codes.astype(np.float32) * scales[:, None]).astype(np.float32)
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    residual_norm = np.nextafter(
        np.sqrt(np.sum(residual * residual, axis=1, dtype=np.float64)), np.inf
    )
    residual_norm[residual_norm == 0.0] = 0.0
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(residual_norm),
    )


def reconstructed_distance_intervals(
    query: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q_codes, q_scales, q_error = quantize_per_vector(query)
    x_codes, x_scales, x_error = quantize_per_vector(base)
    int_dot = q_codes.astype(np.int32) @ x_codes.astype(np.int32).T
    q_sum_sq = np.sum(q_codes.astype(np.int64) ** 2, axis=1)
    x_sum_sq = np.sum(x_codes.astype(np.int64) ** 2, axis=1)
    q_norm_sq = q_sum_sq.astype(np.float64) * q_scales.astype(np.float64) ** 2
    x_norm_sq = x_sum_sq.astype(np.float64) * x_scales.astype(np.float64) ** 2
    scale_products = (
        q_scales.astype(np.float64)[:, None]
        * x_scales.astype(np.float64)[None, :]
    )
    reconstructed_d2 = (
        q_norm_sq[:, None] + x_norm_sq[None, :] - 2.0 * int_dot * scale_products
    )
    np.maximum(reconstructed_d2, 0.0, out=reconstructed_d2)
    reconstructed_d = np.sqrt(reconstructed_d2)
    residual_radius = q_error[:, None] + x_error[None, :]
    lower = np.nextafter(
        np.maximum(0.0, reconstructed_d - residual_radius) ** 2, -np.inf
    )
    upper = np.nextafter((reconstructed_d + residual_radius) ** 2, np.inf)
    return lower, upper


def symmetric_chamfer(matrix: np.ndarray) -> float:
    return 0.5 * (
        float(np.mean(np.min(matrix, axis=1), dtype=np.float64))
        + float(np.mean(np.min(matrix, axis=0), dtype=np.float64))
    )


def aggregate_object_matrices(
    exact_tokens: np.ndarray,
    lower_tokens: np.ndarray,
    upper_tokens: np.ndarray,
    query: ObjectBatch,
    base: ObjectBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(query.slices), len(base.slices))
    exact_objects = np.empty(shape, dtype=np.float64)
    lower_objects = np.empty(shape, dtype=np.float64)
    upper_objects = np.empty(shape, dtype=np.float64)
    total_cells = np.empty(shape, dtype=np.int32)
    competitive_cells = np.empty(shape, dtype=np.int32)
    for qi, qslice in enumerate(query.slices):
        for bi, bslice in enumerate(base.slices):
            exact = exact_tokens[qslice, bslice]
            lower = lower_tokens[qslice, bslice]
            upper = upper_tokens[qslice, bslice]
            exact_objects[qi, bi] = symmetric_chamfer(exact)
            lower_objects[qi, bi] = symmetric_chamfer(lower)
            upper_objects[qi, bi] = symmetric_chamfer(upper)
            total_cells[qi, bi] = exact.size

            forward_possible = lower <= np.min(upper, axis=1)[:, None]
            reverse_possible = lower <= np.min(upper, axis=0)[None, :]
            competitive = forward_possible | reverse_possible
            competitive_cells[qi, bi] = int(np.count_nonzero(competitive))

            # Prove that the frozen competitive-cell rule retains every exact
            # row minimum needed by both directed Chamfer terms.
            refined = np.where(competitive, exact, np.inf)
            refined_score = symmetric_chamfer(refined)
            if refined_score != exact_objects[qi, bi]:
                raise AssertionError(
                    f"Competitive cells changed exact score at {(qi, bi)}: "
                    f"{refined_score} != {exact_objects[qi, bi]}"
                )
    return (
        exact_objects,
        lower_objects,
        upper_objects,
        total_cells,
        competitive_cells,
    )


def strict_midgap_threshold(scores: np.ndarray, target: int) -> tuple[float, float, float]:
    desired = QUERY_OBJECTS * target
    ordered = np.sort(scores.reshape(-1), kind="stable")
    lower_value = float(ordered[desired - 1])
    upper_value = float(ordered[desired])
    if not lower_value < upper_value:
        raise RuntimeError(
            f"No strict object-score gap at target {target}: "
            f"{lower_value} vs {upper_value}"
        )
    threshold = lower_value / 2.0 + upper_value / 2.0
    if not lower_value < threshold < upper_value:
        threshold = float(np.nextafter(lower_value, upper_value))
    if not lower_value < threshold < upper_value:
        raise RuntimeError("Unable to construct strict midpoint")
    return threshold, lower_value, upper_value


def evaluate_threshold(
    target: int,
    exact: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total_cells: np.ndarray,
    competitive_cells: np.ndarray,
) -> dict[str, int | float | bool]:
    threshold, gap_lower, gap_upper = strict_midgap_threshold(exact, target)
    exact_inside = exact <= threshold
    accept = upper <= threshold
    reject = lower > threshold
    ambiguous = ~(accept | reject)
    final_inside = accept | (ambiguous & exact_inside)
    ambiguous_object_count = int(np.count_nonzero(ambiguous))
    total_token_cells = int(np.sum(total_cells, dtype=np.int64))
    ambiguous_token_cells = int(np.sum(total_cells[ambiguous], dtype=np.int64))
    refinement_cells = int(np.sum(competitive_cells[ambiguous], dtype=np.int64))
    object_ambiguity_fraction = ambiguous_object_count / exact.size
    global_refinement_fraction = refinement_cells / max(total_token_cells, 1)
    row: dict[str, int | float | bool] = {
        "target_results_per_query": target,
        "threshold": threshold,
        "gap_lower": gap_lower,
        "gap_upper": gap_upper,
        "exact_object_pairs": int(np.count_nonzero(exact_inside)),
        "exact_results_per_query": float(np.count_nonzero(exact_inside) / exact.shape[0]),
        "direct_accept_object_pairs": int(np.count_nonzero(accept)),
        "direct_reject_object_pairs": int(np.count_nonzero(reject)),
        "ambiguous_object_pairs": ambiguous_object_count,
        "object_ambiguity_fraction": object_ambiguity_fraction,
        "total_token_cells": total_token_cells,
        "ambiguous_object_token_cells": ambiguous_token_cells,
        "competitive_refinement_token_cells": refinement_cells,
        "global_refinement_fraction": global_refinement_fraction,
        "ambiguous_local_refinement_fraction": refinement_cells
        / max(ambiguous_token_cells, 1),
        "lower_containment_violations": int(
            np.count_nonzero(exact + CONTAINMENT_GUARD < lower)
        ),
        "upper_containment_violations": int(
            np.count_nonzero(exact - CONTAINMENT_GUARD > upper)
        ),
        "unsafe_direct_accepts": int(np.count_nonzero(accept & ~exact_inside)),
        "unsafe_direct_rejects": int(np.count_nonzero(reject & exact_inside)),
        "final_decision_mismatches": int(np.count_nonzero(final_inside != exact_inside)),
    }
    exactness_keys = (
        "lower_containment_violations",
        "upper_containment_violations",
        "unsafe_direct_accepts",
        "unsafe_direct_rejects",
        "final_decision_mismatches",
    )
    row["exactness_pass"] = all(int(row[key]) == 0 for key in exactness_keys)
    row["selectivity_pass"] = bool(
        object_ambiguity_fraction <= MAX_OBJECT_AMBIGUITY
        and global_refinement_fraction <= MAX_GLOBAL_REFINEMENT
    )
    row["gate_pass"] = bool(row["exactness_pass"] and row["selectivity_pass"])
    return row


def run_dataset(
    name: str, cache: Path, id_key: str, expected_dim: int
) -> dict[str, object]:
    start = time.time()
    with np.load(cache, allow_pickle=False) as data:
        vectors = np.ascontiguousarray(data["features"], dtype=np.float32)
        ids = np.asarray(data[id_key])
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim or len(ids) != len(vectors):
        raise ValueError((name, vectors.shape, ids.shape))
    vectors = normalize_once(vectors)
    rng = np.random.default_rng(SEED)
    query, base = deterministic_split(vectors, ids, rng)
    if len(np.intersect1d(query.object_ids, base.object_ids)):
        raise AssertionError("Object overlap")
    print(
        "DATASET_START "
        + json.dumps(
            {
                "dataset": name,
                "query_objects": len(query.slices),
                "base_objects": len(base.slices),
                "query_tokens": len(query.vectors),
                "base_tokens": len(base.vectors),
                "query_token_count_range": [
                    int(query.token_counts.min()),
                    int(query.token_counts.max()),
                ],
                "base_token_count_range": [
                    int(base.token_counts.min()),
                    int(base.token_counts.max()),
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    oracle_start = time.time()
    exact_tokens = direct_squared_distances(query.vectors, base.vectors)
    print(f"ORACLE dataset={name} elapsed_s={time.time()-oracle_start:.6f}", flush=True)
    certificate_start = time.time()
    lower_tokens, upper_tokens = reconstructed_distance_intervals(
        query.vectors, base.vectors
    )
    token_lower_violations = int(
        np.count_nonzero(exact_tokens + CONTAINMENT_GUARD < lower_tokens)
    )
    token_upper_violations = int(
        np.count_nonzero(exact_tokens - CONTAINMENT_GUARD > upper_tokens)
    )
    (
        exact_objects,
        lower_objects,
        upper_objects,
        total_cells,
        competitive_cells,
    ) = aggregate_object_matrices(
        exact_tokens, lower_tokens, upper_tokens, query, base
    )
    print(
        f"CERTIFICATE dataset={name} elapsed_s={time.time()-certificate_start:.6f}",
        flush=True,
    )
    thresholds = [
        evaluate_threshold(
            target,
            exact_objects,
            lower_objects,
            upper_objects,
            total_cells,
            competitive_cells,
        )
        for target in TARGETS
    ]
    for row in thresholds:
        print(
            "CELL " + json.dumps({"dataset": name, **row}, sort_keys=True),
            flush=True,
        )
    accumulator_worst_case = expected_dim * 127 * 127
    accumulator_pass = accumulator_worst_case <= np.iinfo(np.int32).max
    result: dict[str, object] = {
        "dataset": name,
        "cache": str(cache),
        "cache_sha256": sha256_file(cache),
        "id_key": id_key,
        "source_shape": list(vectors.shape),
        "source_object_count": int(len(np.unique(ids))),
        "query_object_count": len(query.slices),
        "base_object_count": len(base.slices),
        "object_overlap": int(
            len(np.intersect1d(query.object_ids, base.object_ids))
        ),
        "query_token_count": len(query.vectors),
        "base_token_count": len(base.vectors),
        "query_token_count_min": int(query.token_counts.min()),
        "query_token_count_max": int(query.token_counts.max()),
        "base_token_count_min": int(base.token_counts.min()),
        "base_token_count_max": int(base.token_counts.max()),
        "token_lower_containment_violations": token_lower_violations,
        "token_upper_containment_violations": token_upper_violations,
        "accumulator_worst_case": int(accumulator_worst_case),
        "accumulator_pass": bool(accumulator_pass),
        "thresholds": thresholds,
        "gate_pass": bool(
            accumulator_pass
            and token_lower_violations == 0
            and token_upper_violations == 0
            and all(bool(row["gate_pass"]) for row in thresholds)
        ),
        "elapsed_s": time.time() - start,
    }
    print("DATASET_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("H0 must run with CUDA_VISIBLE_DEVICES empty or -1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    audio_cache = args.audio_cache.resolve()
    video_cache = args.video_cache.resolve()
    script_path = Path(__file__).resolve()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "seed": SEED,
                "script": str(script_path),
                "script_sha256": sha256_file(script_path),
                "audio_cache": str(audio_cache),
                "audio_cache_sha256": sha256_file(audio_cache),
                "video_cache": str(video_cache),
                "video_cache_sha256": sha256_file(video_cache),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.time()
    datasets = [
        run_dataset("esc50_panns", audio_cache, "clip", 2048),
        run_dataset("ucf101_r3d18", video_cache, "group", 512),
    ]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "seed": SEED,
        "query_objects": QUERY_OBJECTS,
        "base_objects": BASE_OBJECTS,
        "targets": list(TARGETS),
        "score": "symmetric_chamfer_squared_distance",
        "normalization": "fp64_l2_then_single_float32_round",
        "oracle": "direct_fp64_token_differences_then_exact_object_aggregation",
        "max_object_ambiguity": MAX_OBJECT_AMBIGUITY,
        "max_global_refinement": MAX_GLOBAL_REFINEMENT,
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "datasets": datasets,
        "gate_pass": all(bool(dataset["gate_pass"]) for dataset in datasets),
        "elapsed_s": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RUN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

