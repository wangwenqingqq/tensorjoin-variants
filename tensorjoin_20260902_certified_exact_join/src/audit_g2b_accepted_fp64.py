#!/usr/bin/env python3
"""Audit every accepted G2B upper pair with independent CPU FP64 arithmetic."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import platform
from pathlib import Path

import numpy as np


N = 60_000
D = 512
CHUNK = 8_192
KEEP_CLOSEST = 64
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/g2b_cifar60000"
PAIR_FILE = PROJECT / "artifacts/g2b/tensorjoin_pairs_u64_le.bin"
RESULT = PROJECT / "results/g2b_accepted_cpu_fp64_audit.json"


def sha256(path: Path, block_size: int = 16 << 20) -> str:
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


def scalar_records(pair_id: int, vectors: np.ndarray, threshold_d2: float) -> dict[str, object]:
    row = pair_id // N
    column = pair_id - row * N
    left = vectors[row]
    right = vectors[column]
    sequential = np.float64(0.0)
    products: list[float] = []
    longdouble = np.longdouble(0.0)
    for a, b in zip(left, right, strict=True):
        delta = np.float64(a) - np.float64(b)
        product = np.float64(delta * delta)
        sequential = np.float64(sequential + product)
        products.append(float(product))
        longdouble += np.longdouble(delta) * np.longdouble(delta)
    fsum = math.fsum(products)
    sqrt_sequential = float(np.sqrt(sequential))
    epsilon = float(np.sqrt(np.float64(threshold_d2)))
    return {
        "pair_id": pair_id,
        "row": row,
        "column": column,
        "sequential_float64_d2": float(sequential),
        "sequential_minus_threshold": float(sequential) - threshold_d2,
        "sequential_inside_squared_contract": bool(sequential <= threshold_d2),
        "sqrt_sequential_float64": sqrt_sequential,
        "sqrt_inside_epsilon_contract": sqrt_sequential <= epsilon,
        "math_fsum_d2": fsum,
        "math_fsum_minus_threshold": fsum - threshold_d2,
        "longdouble_d2_decimal": np.format_float_scientific(
            longdouble, precision=25, unique=False, trim="k"
        ),
        "longdouble_minus_threshold_decimal": np.format_float_scientific(
            longdouble - np.longdouble(threshold_d2),
            precision=25,
            unique=False,
            trim="k",
        ),
    }


def main() -> int:
    for path in (DATA_DIR / "vectors_f32.npy", DATA_DIR / "metadata.json", PAIR_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULT}")
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    threshold_d2 = float(metadata["effective_epsilon_d2"])
    vectors = np.load(DATA_DIR / "vectors_f32.npy", mmap_mode="r", allow_pickle=False)
    pairs = np.memmap(PAIR_FILE, dtype="<u8", mode="r")
    rows = pairs // np.uint64(N)
    columns = pairs - rows * np.uint64(N)
    upper_mask = rows <= columns
    upper = np.asarray(pairs[upper_mask], dtype=np.uint64)
    expected_upper = (pairs.size + N) // 2
    if upper.size != expected_upper:
        raise ValueError((upper.size, expected_upper))

    outside: list[tuple[int, float]] = []
    closest: list[tuple[float, int, float]] = []
    for start in range(0, upper.size, CHUNK):
        current = upper[start : start + CHUNK]
        current_rows = (current // np.uint64(N)).astype(np.int64)
        current_columns = (current % np.uint64(N)).astype(np.int64)
        delta = (
            vectors[current_rows].astype(np.float64)
            - vectors[current_columns].astype(np.float64)
        )
        distances = np.sum(delta * delta, axis=1, dtype=np.float64)
        del delta
        for pair_id, distance in zip(current.tolist(), distances.tolist(), strict=True):
            gap = float(distance - threshold_d2)
            if gap > 0.0:
                outside.append((int(pair_id), gap))
            item = (-abs(gap), int(pair_id), float(distance))
            if len(closest) < KEEP_CLOSEST:
                heapq.heappush(closest, item)
            elif item > closest[0]:
                heapq.heapreplace(closest, item)
        print(
            f"AUDIT_PROGRESS stop={min(start + CHUNK, upper.size)} "
            f"upper={upper.size} outside={len(outside)}",
            flush=True,
        )

    closest_sorted = sorted(
        ((-negative_gap, pair_id, distance) for negative_gap, pair_id, distance in closest),
        key=lambda value: value[0],
    )
    probe_ids = sorted(
        {pair_id for pair_id, _ in outside}
        | {pair_id for _, pair_id, _ in closest_sorted[:KEEP_CLOSEST]}
    )
    probes = [scalar_records(pair_id, vectors, threshold_d2) for pair_id in probe_ids]
    result = {
        "experiment_id": "tensorjoin_20260903_g2b_accepted_cpu_fp64_audit",
        "host": platform.node(),
        "shape": [N, D],
        "threshold_d2": threshold_d2,
        "input_sha256": sha256(DATA_DIR / "vectors_f32.npy"),
        "pair_file_sha256": sha256(PAIR_FILE),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "directed_pair_count": int(pairs.size),
        "unique_upper_pair_count": int(upper.size),
        "vectorized_float64_outside_accepted_count": len(outside),
        "vectorized_float64_outside_accepted": [
            {"pair_id": pair_id, "distance_d2_minus_threshold": gap}
            for pair_id, gap in outside
        ],
        "closest_accepted_by_vectorized_float64": [
            {
                "pair_id": pair_id,
                "absolute_d2_gap": gap,
                "distance_d2": distance,
                "distance_d2_minus_threshold": distance - threshold_d2,
            }
            for gap, pair_id, distance in closest_sorted
        ],
        "scalar_cross_checks": probes,
        "accepted_set_passes_sequential_squared_contract": all(
            bool(record["sequential_inside_squared_contract"]) for record in probes
        ) and not outside,
        "measurement_status": "correctness_diagnostic_no_performance_claim",
    }
    atomic_json(RESULT, result)
    print("AUDIT_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["accepted_set_passes_sequential_squared_contract"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
