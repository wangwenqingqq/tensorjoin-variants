#!/usr/bin/env python3
"""Prepare the immutable G2A Cifar60K subset and exact pair-ID oracle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import pdist


EXPERIMENT_ID = "tensorjoin_20260903_cifar4096_oracle_g2a"
SEED = 20260903
SOURCE_ROWS = 60_000
SUBSET_ROWS = 4_096
DIMENSION = 512
TARGET_RESULTS_PER_POINT = 64

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/cifar60k/cifar60k_base.fvecs"
OUTPUT_DIR = PROJECT / "data/g2a_cifar4096"
RESULT = PROJECT / "results/g2a_prepare.json"
RECEIPT = PROJECT / "receipts/g2a_artifacts_sha256.json"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_fvecs(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<i4")
    row_words = DIMENSION + 1
    if words.size % row_words:
        raise ValueError(f"Malformed fvecs word count: {words.size}")
    rows = words.reshape(-1, row_words)
    if rows.shape[0] != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS} rows, found {rows.shape[0]}")
    if not np.all(rows[:, 0] == DIMENSION):
        bad = np.flatnonzero(rows[:, 0] != DIMENSION)[:8]
        raise ValueError(f"Non-{DIMENSION} fvec headers at rows {bad.tolist()}")
    vectors = rows[:, 1:].view("<f4").copy()
    if vectors.shape != (SOURCE_ROWS, DIMENSION):
        raise AssertionError(vectors.shape)
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite Cifar60K coordinates")
    return vectors


def condensed_to_ij(indices: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert SciPy's squareform upper-triangle condensed index."""
    k = np.asarray(indices, dtype=np.int64)
    i = n - 2 - np.floor(
        np.sqrt(-8.0 * k + 4.0 * n * (n - 1) - 7.0) / 2.0 - 0.5
    ).astype(np.int64)
    j = (
        k
        + i
        + 1
        - n * (n - 1) // 2
        + (n - i) * (n - i - 1) // 2
    )
    forward = n * i - i * (i + 1) // 2 + j - i - 1
    if not np.array_equal(forward, k):
        raise AssertionError("Condensed-index inversion failed")
    if np.any(i < 0) or np.any(j <= i) or np.any(j >= n):
        raise AssertionError("Invalid upper-triangle coordinates")
    return i.astype(np.uint64), j.astype(np.uint64)


def calibrate_radius(condensed_d2: np.ndarray) -> dict[str, int | float]:
    target_directed = SUBSET_ROWS * TARGET_RESULTS_PER_POINT
    if (target_directed - SUBSET_ROWS) % 2:
        raise AssertionError("Directed target is incompatible with symmetric self-join")
    target_upper_nonself = (target_directed - SUBSET_ROWS) // 2
    rank = target_upper_nonself - 1
    lower_d2 = float(np.partition(condensed_d2, rank)[rank])
    greater = condensed_d2[condensed_d2 > lower_d2]
    if greater.size == 0:
        raise ValueError("No strict upper radius bracket")
    upper_d2 = float(np.min(greater))
    lower = float(np.sqrt(lower_d2))
    upper = float(np.sqrt(upper_d2))
    epsilon = lower + (upper - lower) / 2.0
    effective_d2 = float(epsilon * epsilon)
    if not lower_d2 < effective_d2 < upper_d2:
        raise AssertionError(
            f"Rounded epsilon escaped mid-gap: {lower_d2}, {effective_d2}, {upper_d2}"
        )
    upper_count = int(np.count_nonzero(condensed_d2 <= effective_d2))
    directed_count = SUBSET_ROWS + 2 * upper_count
    return {
        "target_directed_pairs": target_directed,
        "target_results_per_point": TARGET_RESULTS_PER_POINT,
        "target_upper_nonself_pairs": target_upper_nonself,
        "lower_distance_d2": lower_d2,
        "upper_distance_d2": upper_d2,
        "gap_d2": upper_d2 - lower_d2,
        "epsilon": epsilon,
        "effective_epsilon_d2": effective_d2,
        "upper_nonself_pairs": upper_count,
        "directed_pairs_including_self": directed_count,
        "actual_results_per_point": directed_count / SUBSET_ROWS,
        "tie_expansion_directed_pairs": directed_count - target_directed,
    }


def build_canonical_pairs(
    condensed_d2: np.ndarray, effective_d2: float
) -> tuple[np.ndarray, np.ndarray]:
    condensed_ids = np.flatnonzero(condensed_d2 <= effective_d2).astype(np.int64)
    i, j = condensed_to_ij(condensed_ids, SUBSET_ROWS)
    diagonal = np.arange(SUBSET_ROWS, dtype=np.uint64)
    pairs = np.concatenate(
        (
            diagonal * SUBSET_ROWS + diagonal,
            i * SUBSET_ROWS + j,
            j * SUBSET_ROWS + i,
        )
    )
    pairs.sort()
    if pairs.size and np.any(pairs[1:] == pairs[:-1]):
        raise AssertionError("Duplicate canonical pair IDs")
    if pairs.size and int(pairs[-1]) >= SUBSET_ROWS * SUBSET_ROWS:
        raise AssertionError("Out-of-range canonical pair ID")
    counts = np.bincount(
        (pairs // np.uint64(SUBSET_ROWS)).astype(np.int64), minlength=SUBSET_ROWS
    ).astype(np.uint32)
    if int(counts.sum(dtype=np.uint64)) != pairs.size:
        raise AssertionError("Neighbor-count reconstruction failed")
    return pairs, counts


def main() -> int:
    for path in (OUTPUT_DIR, RESULT, RECEIPT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite G2A evidence: {path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    started = time.time()
    source_hash = sha256_file(SOURCE)
    script_hash = sha256_file(Path(__file__).resolve())
    run = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "script_sha256": script_hash,
    }
    print("RUN_START " + json.dumps(run, sort_keys=True), flush=True)

    vectors = load_fvecs(SOURCE)
    rng = np.random.default_rng(SEED)
    source_ids = np.sort(
        rng.choice(SOURCE_ROWS, size=SUBSET_ROWS, replace=False).astype(np.uint32)
    )
    subset_f32 = np.ascontiguousarray(vectors[source_ids], dtype=np.float32)
    subset_f64 = np.ascontiguousarray(subset_f32, dtype=np.float64)
    del vectors

    oracle_started = time.time()
    condensed_d2 = pdist(subset_f64, metric="sqeuclidean")
    if condensed_d2.shape != (SUBSET_ROWS * (SUBSET_ROWS - 1) // 2,):
        raise AssertionError(condensed_d2.shape)
    if not np.isfinite(condensed_d2).all() or np.any(condensed_d2 < 0):
        raise ValueError("Invalid direct-difference FP64 distances")
    radius = calibrate_radius(condensed_d2)
    pairs, neighbor_counts = build_canonical_pairs(
        condensed_d2, float(radius["effective_epsilon_d2"])
    )
    if pairs.size != int(radius["directed_pairs_including_self"]):
        raise AssertionError("Oracle count disagrees with radius calibration")
    oracle_elapsed = time.time() - oracle_started
    print(
        "ORACLE "
        + json.dumps({**radius, "elapsed_s": oracle_elapsed}, sort_keys=True),
        flush=True,
    )

    temporary = OUTPUT_DIR.with_name(f".{OUTPUT_DIR.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        np.save(temporary / "source_row_ids_u32.npy", source_ids, allow_pickle=False)
        np.save(temporary / "vectors_f32.npy", subset_f32, allow_pickle=False)
        subset_f64.astype("<f8", copy=False).tofile(temporary / "vectors_f64_le.bin")
        np.savetxt(
            temporary / "vectors_f64_exact.csv",
            subset_f64,
            fmt="%.17g",
            delimiter=",",
        )
        np.save(temporary / "oracle_pairs_u64.npy", pairs, allow_pickle=False)
        pairs.astype("<u8", copy=False).tofile(temporary / "oracle_pairs_u64_le.bin")
        np.save(
            temporary / "oracle_neighbor_counts_u32.npy",
            neighbor_counts,
            allow_pickle=False,
        )
        metadata = {
            **run,
            "subset_rows": SUBSET_ROWS,
            "dimension": DIMENSION,
            "source_id_min": int(source_ids[0]),
            "source_id_max": int(source_ids[-1]),
            "source_id_unique": int(np.unique(source_ids).size),
            "input_min": float(subset_f32.min()),
            "input_max": float(subset_f32.max()),
            "radius": radius,
            "oracle_method": "scipy_pdist_fp64_direct_difference_sqeuclidean",
            "oracle_pair_encoding": "uint64(i) * 4096 + uint64(j)",
            "oracle_includes_self": True,
            "oracle_is_directed": True,
            "oracle_pair_count": int(pairs.size),
            "neighbor_count_min": int(neighbor_counts.min()),
            "neighbor_count_median": float(np.median(neighbor_counts)),
            "neighbor_count_max": int(neighbor_counts.max()),
            "oracle_elapsed_s": oracle_elapsed,
        }
        with (temporary / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        file_records: dict[str, dict[str, int | str]] = {}
        for path in sorted(temporary.iterdir()):
            if path.is_file():
                file_records[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        os.replace(temporary, OUTPUT_DIR)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    completed = {
        **metadata,
        "artifact_dir": str(OUTPUT_DIR),
        "artifact_files": file_records,
        "wall_elapsed_s": time.time() - started,
    }
    atomic_json(RESULT, completed)
    atomic_json(
        RECEIPT,
        {
            "experiment_id": EXPERIMENT_ID,
            "source": {"path": str(SOURCE), "sha256": source_hash},
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": script_hash,
            },
            "artifacts": file_records,
            "result": {
                "path": str(RESULT),
                "sha256": sha256_file(RESULT),
            },
        },
    )
    print(
        "RUN_COMPLETE "
        + json.dumps(
            {
                "result": str(RESULT),
                "result_sha256": sha256_file(RESULT),
                "receipt": str(RECEIPT),
                "receipt_sha256": sha256_file(RECEIPT),
                "pair_count": int(pairs.size),
                "pair_hash": file_records["oracle_pairs_u64.npy"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
