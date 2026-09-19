#!/usr/bin/env python3
"""Prepare deterministic nested G4B subsets from three public vector sources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT / "data/g4b_public"
SEED = 20_260_903
MAX_ROWS = 4_096

CIFAR_SOURCE = PROJECT / "data/cifar60k/cifar60k_base.fvecs"
FASHION_SOURCE = (
    PROJECT / "raw/g4b_sources/fashion-train-images-idx3-ubyte.gz"
)
DEFAULT_SIFT_SOURCE = Path(
    "@TENSORJOIN_ROOT@/project/GTS/Datasets/sift1m/sift_base.fvecs"
)

EXPECTED = {
    "sift128": {
        "rows": 1_000_000,
        "dimension": 128,
        "bytes": 516_000_000,
        "sha256": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    },
    "cifar_gist512": {
        "rows": 60_000,
        "dimension": 512,
        "bytes": 123_120_000,
        "sha256": "a7170faaa80a072cd603ed472104049ead87fbaff224e94a529021d161f8aea4",
    },
    "fashion784": {
        "rows": 60_000,
        "dimension": 784,
        "bytes": 26_421_880,
        "sha256": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sift-source", type=Path, default=DEFAULT_SIFT_SOURCE)
    return parser.parse_args()


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            value.update(block)
    return value.hexdigest()


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


def load_fvec_rows(
    path: Path, rows: int, dimension: int, row_ids: np.ndarray
) -> np.ndarray:
    expected_words = rows * (dimension + 1)
    if path.stat().st_size != expected_words * 4:
        raise RuntimeError(f"Unexpected fvecs byte count: {path}")
    words = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dimension + 1))
    if not np.all(words[:, 0] == dimension):
        raise RuntimeError(f"Inconsistent fvecs dimension headers: {path}")
    floats = np.memmap(path, dtype="<f4", mode="r", shape=(rows, dimension + 1))
    return np.ascontiguousarray(floats[row_ids, 1:], dtype=np.float32)


def load_fashion_rows(path: Path, row_ids: np.ndarray) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise RuntimeError("Truncated Fashion-MNIST image header")
        magic, rows, height, width = struct.unpack(">IIII", header)
        payload = handle.read()
    if (magic, rows, height, width) != (2051, 60_000, 28, 28):
        raise RuntimeError(
            f"Unexpected Fashion-MNIST header: {(magic, rows, height, width)}"
        )
    expected = rows * height * width
    if len(payload) != expected:
        raise RuntimeError("Unexpected Fashion-MNIST payload length")
    images = np.frombuffer(payload, dtype=np.uint8).reshape(rows, height * width)
    return np.ascontiguousarray(images[row_ids], dtype=np.float32)


def validate_source(dataset: str, path: Path) -> None:
    expected = EXPECTED[dataset]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"Source byte count mismatch for {dataset}")
    if digest(path) != expected["sha256"]:
        raise RuntimeError(f"Source SHA-256 mismatch for {dataset}")
    if "md5" in expected and digest(path, "md5") != expected["md5"]:
        raise RuntimeError(f"Source MD5 mismatch for {dataset}")


def main() -> int:
    args = parse_args()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    sources = {
        "sift128": args.sift_source.resolve(),
        "cifar_gist512": CIFAR_SOURCE.resolve(),
        "fashion784": FASHION_SOURCE.resolve(),
    }
    manifest: dict[str, object] = {
        "experiment_id": "tensorjoin_20260903_g4b_public_data_prepare",
        "seed": SEED,
        "rng": "numpy.random.PCG64",
        "sampling": (
            "first 4096 indices of one full-source seeded permutation; later "
            "scale cells are nested prefixes"
        ),
        "numpy": np.__version__,
        "preparer_sha256": digest(Path(__file__).resolve()),
        "datasets": {},
    }

    for dataset, source in sources.items():
        validate_source(dataset, source)
        expected = EXPECTED[dataset]
        rng = np.random.Generator(np.random.PCG64(SEED))
        row_ids = np.ascontiguousarray(
            rng.permutation(expected["rows"])[:MAX_ROWS], dtype=np.uint32
        )
        if dataset == "fashion784":
            vectors = load_fashion_rows(source, row_ids)
            loader = "official IDX gzip; uint8 pixels widened exactly to float32"
        else:
            vectors = load_fvec_rows(
                source, expected["rows"], expected["dimension"], row_ids
            )
            loader = "little-endian fvecs; native float32 coordinates"

        if vectors.shape != (MAX_ROWS, expected["dimension"]):
            raise RuntimeError(f"Prepared shape mismatch for {dataset}")
        if not np.isfinite(vectors).all():
            raise RuntimeError(f"Non-finite coordinate in {dataset}")
        accumulator_limit = int(expected["dimension"] * 127 * 127)
        if accumulator_limit > 2**24:
            raise RuntimeError(f"Exact INT32-to-FP32 precondition fails for {dataset}")

        output = OUTPUT_ROOT / dataset
        output.mkdir()
        vector_path = output / "vectors_f32.npy"
        rows_path = output / "source_row_ids_u32.npy"
        metadata_path = output / "metadata.json"
        atomic_npy(vector_path, vectors)
        atomic_npy(rows_path, row_ids)
        record = {
            "dataset_id": dataset,
            "source_path": str(source),
            "source_rows": expected["rows"],
            "source_dimension": expected["dimension"],
            "source_bytes": expected["bytes"],
            "source_sha256": expected["sha256"],
            "source_md5": expected.get("md5"),
            "loader": loader,
            "prepared_shape": list(vectors.shape),
            "prepared_dtype": str(vectors.dtype),
            "finite": bool(np.isfinite(vectors).all()),
            "coordinate_min": float(vectors.min()),
            "coordinate_max": float(vectors.max()),
            "source_row_ids_min": int(row_ids.min()),
            "source_row_ids_max": int(row_ids.max()),
            "source_row_ids_unique": int(np.unique(row_ids).size),
            "source_row_ids_file_sha256": digest(rows_path),
            "vectors_file_sha256": digest(vector_path),
            "int32_accumulator_absolute_limit": accumulator_limit,
            "exact_int32_to_fp32_precondition": accumulator_limit <= 2**24,
        }
        atomic_json(metadata_path, record)
        manifest["datasets"][dataset] = record
        print(json.dumps(record, sort_keys=True), flush=True)

    atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
