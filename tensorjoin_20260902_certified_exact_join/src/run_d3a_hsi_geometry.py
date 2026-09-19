#!/usr/bin/env python3
"""Run the frozen CPU-only Indian Pines patch certificate geometry gate."""

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


EXPERIMENT_ID = "tensorjoin_20260903_indian_pines_geometry_d3a"
SEED = 20260903
QUERY_COUNT = 512
BASE_COUNT = 4096
DIMENSION = 1984
TARGETS = (1, 64)
BOUND_PAD = 1e-4
CONTAINMENT_GUARD = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            value.update(block)
    return value.hexdigest()


def spatial_split(
    features: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    query_pool = np.flatnonzero(rows <= 47)
    base_pool = np.flatnonzero(rows >= 51)
    query_ids = rng.choice(query_pool, size=QUERY_COUNT, replace=False)
    base_ids = rng.choice(base_pool, size=BASE_COUNT, replace=False)
    query_rows = rows[query_ids]
    base_rows = rows[base_ids]
    if int(query_rows.max()) + 2 >= int(base_rows.min()):
        raise AssertionError("Spatial source-pixel overlap")
    return (
        np.ascontiguousarray(features[query_ids], dtype=np.float32),
        np.ascontiguousarray(features[base_ids], dtype=np.float32),
        np.stack((rows[query_ids], cols[query_ids]), axis=1),
        np.stack((rows[base_ids], cols[base_ids]), axis=1),
    )


def exact_d2(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    output = np.empty((len(q), len(x)), dtype=np.float64)
    for start in range(0, len(q), 4):
        stop = min(start + 4, len(q))
        delta = q[start:stop, None, :] - x[None, :, :]
        output[start:stop] = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    return output


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = np.where(max_abs == 0, np.float32(1.0), max_abs / np.float32(127.0))
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = codes.astype(np.float32) * scales[:, None]
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    errors = np.nextafter(np.linalg.norm(residual, axis=1), np.inf)
    return codes, scales.astype(np.float32), errors


def reconstructed_distance(
    qc: np.ndarray, qs: np.ndarray, xc: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    dot = qc.astype(np.int32) @ xc.astype(np.int32).T
    qn = np.sum(qc.astype(np.int64) ** 2, axis=1) * qs.astype(np.float64) ** 2
    xn = np.sum(xc.astype(np.int64) ** 2, axis=1) * xs.astype(np.float64) ** 2
    d2 = qn[:, None] + xn[None, :] - 2.0 * dot * qs.astype(np.float64)[:, None] * xs.astype(np.float64)[None, :]
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def midgap(values: np.ndarray, target: int) -> dict[str, float]:
    flat = values.reshape(-1)
    rank = QUERY_COUNT * target - 1
    lower = float(np.partition(flat, rank)[rank])
    upper = float(np.min(flat[flat > lower]))
    return {
        "lower_distance_d2": lower,
        "upper_distance_d2": upper,
        "gap_d2": upper - lower,
        "threshold_d2": lower + (upper - lower) / 2.0,
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("D3A must be CPU-only")
    if args.output.exists() or args.threshold_output.exists():
        raise FileExistsError("Refusing to overwrite D3A evidence")
    cache = args.feature_cache.resolve()
    start = time.time()
    print("RUN_START " + json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "feature_cache": str(cache),
        "feature_cache_sha256": sha256_file(cache),
        "seed": SEED,
    }, sort_keys=True), flush=True)
    with np.load(cache, allow_pickle=False) as data:
        features = np.ascontiguousarray(data["features"], dtype=np.float32)
        query, base, query_coords, base_coords = spatial_split(features, data["row"], data["col"])
    if features.shape != (143 * 143, DIMENSION):
        raise ValueError(features.shape)
    oracle = exact_d2(query, base)
    qc, qs, qe = quantize(query)
    xc, xs, xe = quantize(base)
    reconstructed = reconstructed_distance(qc, qs, xc, xs)
    radius = qe[:, None] + xe[None, :] + BOUND_PAD
    lower_d2 = np.nextafter(np.maximum(reconstructed - radius, 0.0) ** 2, -np.inf)
    upper_d2 = np.nextafter((reconstructed + radius) ** 2, np.inf)
    rows_out: list[dict[str, int | float]] = []
    for target in TARGETS:
        info = midgap(oracle, target)
        threshold = info["threshold_d2"]
        exact_inside = oracle <= threshold
        accept = upper_d2 <= threshold
        reject = lower_d2 > threshold
        ambiguous = ~(accept | reject)
        final = accept | (ambiguous & exact_inside)
        exact_count = int(np.count_nonzero(exact_inside))
        ambiguous_count = int(np.count_nonzero(ambiguous))
        row = {
            "target_results_per_query": target,
            **info,
            "exact_pairs": exact_count,
            "exact_results_per_query": exact_count / QUERY_COUNT,
            "direct_accept_pairs": int(np.count_nonzero(accept)),
            "direct_reject_pairs": int(np.count_nonzero(reject)),
            "ambiguous_pairs": ambiguous_count,
            "ambiguous_fraction": ambiguous_count / oracle.size,
            "ambiguous_to_output": ambiguous_count / max(exact_count, 1),
            "lower_containment_violations": int(np.count_nonzero(oracle + CONTAINMENT_GUARD < lower_d2)),
            "upper_containment_violations": int(np.count_nonzero(oracle - CONTAINMENT_GUARD > upper_d2)),
            "false_accept_pairs": int(np.count_nonzero(accept & ~exact_inside)),
            "false_reject_pairs": int(np.count_nonzero(reject & exact_inside)),
            "final_classification_mismatch": int(np.count_nonzero(final != exact_inside)),
        }
        rows_out.append(row)
        print("THRESHOLD " + json.dumps(row, sort_keys=True), flush=True)
    exactness = all(row[key] == 0 for row in rows_out for key in (
        "lower_containment_violations", "upper_containment_violations",
        "false_accept_pairs", "false_reject_pairs", "final_classification_mismatch"))
    ambiguity = all(float(row["ambiguous_fraction"]) <= 0.10 for row in rows_out)
    accumulator = DIMENSION * 127 * 127
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "seed": SEED,
        "query_shape": list(query.shape),
        "base_shape": list(base.shape),
        "query_row_range": [int(query_coords[:, 0].min()), int(query_coords[:, 0].max())],
        "base_row_range": [int(base_coords[:, 0].min()), int(base_coords[:, 0].max())],
        "source_pixel_overlap": False,
        "feature_cache_sha256": sha256_file(cache),
        "script_sha256": sha256_file(Path(__file__)),
        "bound_pad": BOUND_PAD,
        "accumulator_worst_case": accumulator,
        "accumulator_pass": accumulator <= np.iinfo(np.int32).max,
        "exactness_pass": exactness,
        "ambiguity_pass": ambiguity,
        "gate_pass": bool(exactness and ambiguity and accumulator <= np.iinfo(np.int32).max),
        "thresholds": rows_out,
        "elapsed_s": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.threshold_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)
    print("RUN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
