#!/usr/bin/env python3
"""Prepare immutable full Cifar60K artifacts for the G2B exact self-join smoke."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_cifar60000_prepare_g2b"
N = 60_000
D = 512
EPSILON = np.float64(0.62890625)
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/cifar60k/cifar60k_base.fvecs"
OUTPUT = PROJECT / "data/g2b_cifar60000"
RESULT = PROJECT / "results/g2b_prepare.json"
RECEIPT = PROJECT / "receipts/g2b_artifacts_sha256.json"
FASTED_ACCURACY = (
    PROJECT
    / "external/fasted/results/accuracy_results/global_averaged_real_world_accuracies.json"
)


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


def main() -> int:
    for path in (SOURCE, FASTED_ACCURACY):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (OUTPUT, RESULT, RECEIPT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    words = np.memmap(SOURCE, dtype="<i4", mode="r")
    if words.size % (D + 1):
        raise ValueError(f"Malformed fvec word count: {words.size}")
    rows = words.reshape(-1, D + 1)
    dimensions = rows[:, 0]
    if rows.shape[0] != N or not np.all(dimensions == D):
        raise ValueError(
            f"Expected {N} rows with D={D}; got {rows.shape[0]} rows and "
            f"dimension range [{dimensions.min()}, {dimensions.max()}]"
        )
    vectors = np.ascontiguousarray(rows[:, 1:].view("<f4"), dtype=np.float32)
    if vectors.shape != (N, D) or not np.isfinite(vectors).all():
        raise ValueError("Invalid vector shape or non-finite coordinate")

    accuracy = json.loads(FASTED_ACCURACY.read_text(encoding="utf-8"))
    cifar_records = {
        key: value for key, value in accuracy.items() if key.lower().startswith("cifar60k_")
    }
    if len(cifar_records) != 1:
        raise ValueError(f"Expected one Cifar accuracy record, got {len(cifar_records)}")
    accuracy_key, accuracy_record = next(iter(cifar_records.items()))
    if int(accuracy_record["total_truth_pairs"]) != 3_926_074:
        raise ValueError("Pinned FaSTED FP64-GDS truth count changed")

    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        np.save(temporary / "vectors_f32.npy", vectors, allow_pickle=False)
        vectors.astype("<f4", copy=False).tofile(temporary / "vectors_f32.raw")
        vectors.astype("<f8").tofile(temporary / "vectors_f64_exact_widening.raw")

        threshold_d2 = float(EPSILON * EPSILON)
        metadata = {
            "experiment_id": EXPERIMENT_ID,
            "host": platform.node(),
            "source": str(SOURCE.relative_to(PROJECT)),
            "source_sha256": sha256(SOURCE),
            "source_format": "fvecs: little-endian int32 dimension prefix plus float32 coordinates",
            "shape": [N, D],
            "source_coordinate_dtype": "little-endian float32",
            "finite": True,
            "epsilon": float(EPSILON),
            "effective_epsilon_d2": threshold_d2,
            "canonical_pair_encoding": "uint64(i) * 60000 + uint64(j), directed including self",
            "dataset_redistribution": "internal experiment only; no license found on source page",
            "coordinate_stats": {
                "minimum": float(vectors.min()),
                "maximum": float(vectors.max()),
                "mean": float(vectors.mean(dtype=np.float64)),
                "stddev": float(vectors.std(dtype=np.float64)),
            },
            "diagnostic_expected_exact_count": {
                "value": int(accuracy_record["total_truth_pairs"]),
                "role": "capacity/smoke diagnostic only; not the sole full-output oracle",
                "source_record": accuracy_key,
                "source_file": str(FASTED_ACCURACY.relative_to(PROJECT)),
                "source_file_sha256": sha256(FASTED_ACCURACY),
            },
            "diagnostic_fasted_count": int(accuracy_record["total_test_pairs"]),
            "diagnostic_fasted_test_only": int(accuracy_record["left_only"]),
            "diagnostic_fasted_truth_only": int(accuracy_record["right_only"]),
        }
        atomic_json(temporary / "metadata.json", metadata)
        artifacts = {}
        for path in sorted(temporary.iterdir()):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        atomic_json(temporary / "artifacts_sha256.json", artifacts)
        os.replace(temporary, OUTPUT)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "output_directory": str(OUTPUT.relative_to(PROJECT)),
        "artifacts": json.loads(
            (OUTPUT / "artifacts_sha256.json").read_text(encoding="utf-8")
        ),
        "preparation_source": str(Path(__file__).resolve().relative_to(PROJECT)),
        "preparation_source_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(RECEIPT, receipt)
    result = {
        **json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8")),
        "status": "prepared_no_gpu_execution",
        "artifact_receipt": str(RECEIPT.relative_to(PROJECT)),
        "artifact_receipt_sha256": sha256(RECEIPT),
        "preparation_source_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
