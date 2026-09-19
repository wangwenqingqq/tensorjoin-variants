#!/usr/bin/env python3
"""Build exact direct-FP64 G4B self-range-join thresholds and upper-ID oracles."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import pdist


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data/g4b_public"
RESULT = PROJECT / "results/g4b_exact_oracle_manifest.json"
SCALES = (1_024, 2_048, 4_096)
TARGET_DEGREES = (1, 16, 64)
EXPECTED_VECTORS = {
    "sift128": "91648432b84009381b22aedc3fba487e787ee4e659ed0aaf5392f8e1042d9126",
    "cifar_gist512": "cea1987b6a07df43a71d5361afa5f1630c8f1abbbc44493d60af0898e6394702",
    "fashion784": "16149e1a1deaa2afeb205a0d22d49d21d1b784ca654851985603777e9c8b29d8",
}
EXPECTED_ROWS = {
    "sift128": "fcd014f989df767832e5028eda18607d5e726dcdfd5d9f17abbad3c1ac739fa5",
    "cifar_gist512": "b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c",
    "fashion784": "b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c",
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            value.update(block)
    return value.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u8").tobytes(order="C")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def threshold_at_strict_gap(
    distances: np.ndarray, target_upper_nonself: int
) -> tuple[float, float, float, int]:
    if not 0 < target_upper_nonself < distances.size:
        raise ValueError("Target upper-pair count is outside the distance array")
    partitioned = np.partition(
        distances, (target_upper_nonself - 1, target_upper_nonself)
    )
    lower = float(partitioned[target_upper_nonself - 1])
    larger = distances[distances > lower]
    if larger.size == 0:
        raise RuntimeError("No strict distance gap exists above the target")
    upper = float(np.min(larger))
    midpoint = float(lower + (upper - lower) / 2.0)
    if not lower < midpoint < upper:
        raise RuntimeError("No representable strict midpoint between order statistics")
    actual = int(np.count_nonzero(distances <= lower))
    return lower, upper, midpoint, actual


def condensed_selected_upper_ids(
    distances: np.ndarray,
    threshold_d2: float,
    n: int,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    selected = np.flatnonzero(distances <= threshold_d2)
    if rows.size != distances.size:
        raise RuntimeError("SciPy condensed distance size mismatch")
    off_diagonal = (
        rows[selected].astype(np.uint64) * np.uint64(n)
        + columns[selected].astype(np.uint64)
    )
    diagonal = (
        np.arange(n, dtype=np.uint64) * np.uint64(n + 1)
    )
    return np.sort(np.concatenate((diagonal, off_diagonal)))


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    manifest: dict[str, object] = {
        "experiment_id": "tensorjoin_20260903_g4b_exact_oracles",
        "oracle": (
            "scipy.spatial.distance.pdist sqeuclidean on source float32 rows "
            "widened exactly to float64; strict mid-gap threshold"
        ),
        "scales": list(SCALES),
        "target_average_directed_nonself_degrees": list(TARGET_DEGREES),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4B_OPPORTUNITY.md"),
        "cells": [],
    }
    for dataset in EXPECTED_VECTORS:
        dataset_root = DATA_ROOT / dataset
        vector_path = dataset_root / "vectors_f32.npy"
        row_path = dataset_root / "source_row_ids_u32.npy"
        if sha256_file(vector_path) != EXPECTED_VECTORS[dataset]:
            raise RuntimeError(f"Prepared vector hash mismatch for {dataset}")
        if sha256_file(row_path) != EXPECTED_ROWS[dataset]:
            raise RuntimeError(f"Prepared row-index hash mismatch for {dataset}")
        all_vectors = np.load(vector_path, allow_pickle=False)
        all_rows = np.load(row_path, allow_pickle=False)
        if all_vectors.shape[0] != max(SCALES) or all_vectors.dtype != np.float32:
            raise RuntimeError(f"Prepared vector contract mismatch for {dataset}")
        if all_rows.shape != (max(SCALES),) or all_rows.dtype != np.uint32:
            raise RuntimeError(f"Prepared row-index contract mismatch for {dataset}")

        for n in SCALES:
            vectors = np.ascontiguousarray(all_vectors[:n], dtype=np.float64)
            started = time.perf_counter()
            distances = pdist(vectors, metric="sqeuclidean")
            oracle_seconds = time.perf_counter() - started
            if distances.shape != (n * (n - 1) // 2,):
                raise RuntimeError("Unexpected condensed distance shape")
            if not np.isfinite(distances).all() or np.any(distances < 0):
                raise RuntimeError("Invalid exact squared distance")
            rows, columns = np.triu_indices(n, k=1)
            if rows.size != distances.size:
                raise RuntimeError("SciPy condensed distance size mismatch")

            scale_root = dataset_root / f"n{n}"
            if scale_root.exists():
                raise FileExistsError(scale_root)
            scale_root.mkdir()
            for target_degree in TARGET_DEGREES:
                target_upper = n * target_degree // 2
                lower, upper, threshold_d2, actual_upper = threshold_at_strict_gap(
                    distances, target_upper
                )
                oracle_ids = condensed_selected_upper_ids(
                    distances, threshold_d2, n, rows, columns
                )
                expected_count = n + actual_upper
                if oracle_ids.size != expected_count:
                    raise RuntimeError("Oracle output count mismatch")
                cell_root = scale_root / f"k{target_degree}"
                cell_root.mkdir()
                oracle_path = cell_root / "oracle_upper_ids_u64.npy"
                metadata_path = cell_root / "metadata.json"
                atomic_npy(oracle_path, oracle_ids)
                record = {
                    "dataset_id": dataset,
                    "n": n,
                    "dimension": int(all_vectors.shape[1]),
                    "target_average_directed_nonself_degree": target_degree,
                    "target_upper_nonself_pairs": target_upper,
                    "lower_order_statistic_d2": lower,
                    "upper_order_statistic_d2": upper,
                    "threshold_d2": threshold_d2,
                    "epsilon": float(np.sqrt(threshold_d2)),
                    "actual_upper_nonself_pairs": actual_upper,
                    "actual_average_directed_nonself_degree": (
                        2.0 * actual_upper / n
                    ),
                    "tie_expansion_upper_pairs": actual_upper - target_upper,
                    "oracle_upper_count_including_self": int(oracle_ids.size),
                    "oracle_upper_raw_sha256": sha256_u64(oracle_ids),
                    "oracle_upper_file_sha256": sha256_file(oracle_path),
                    "oracle_seconds_for_shared_scale_distance_array": oracle_seconds,
                    "vector_prefix_raw_sha256": hashlib.sha256(
                        np.asarray(all_vectors[:n], dtype="<f4").tobytes(order="C")
                    ).hexdigest(),
                    "source_row_prefix_raw_sha256": hashlib.sha256(
                        np.asarray(all_rows[:n], dtype="<u4").tobytes(order="C")
                    ).hexdigest(),
                }
                atomic_json(metadata_path, record)
                manifest["cells"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            del distances, vectors, rows, columns

    atomic_json(RESULT, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
