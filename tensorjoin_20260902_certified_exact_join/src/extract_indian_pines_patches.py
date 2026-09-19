#!/usr/bin/env python3
"""Extract standardized normalized 3x3 HSI patch tensors from Indian Pines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat


EXPERIMENT_ID = "tensorjoin_20260903_indian_pines_embedding_d3"
EXPECTED_MD5 = "cea396f8a7bdf947f26a7a36c7b7c81a"
PATCH = 3
RAW_DIMENSION = 3 * 3 * 220
PADDED_DIMENSION = 1984


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path, algorithm: str, block_size: int = 8 << 20) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            value.update(block)
    return value.hexdigest()


def load_cube(path: Path) -> tuple[np.ndarray, str, list[dict[str, object]]]:
    content = loadmat(path)
    inventory: list[dict[str, object]] = []
    candidates: list[tuple[str, np.ndarray]] = []
    for name, value in content.items():
        if name.startswith("__") or not isinstance(value, np.ndarray):
            continue
        inventory.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        if value.ndim == 3 and np.issubdtype(value.dtype, np.number):
            candidates.append((name, value))
    if not candidates:
        raise ValueError(f"No numeric 3-D array: {inventory}")
    name, cube = max(candidates, key=lambda pair: pair[1].size)
    spectral_axis = int(np.argmax(cube.shape))
    cube = np.moveaxis(cube, spectral_axis, 2) if spectral_axis != 2 else cube
    cube = np.asarray(cube, dtype=np.float64, order="C")
    if cube.shape != (145, 145, 220):
        raise ValueError(cube.shape)
    if not np.isfinite(cube).all():
        raise ValueError("Non-finite cube")
    return cube, name, inventory


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Refusing to overwrite D3 extraction evidence")
    dataset = args.dataset.resolve()
    md5 = digest(dataset, "md5")  # nosec: published artifact identity
    if md5 != EXPECTED_MD5:
        raise ValueError(f"MD5 mismatch: {md5}")
    start = time.time()
    cube, variable, inventory = load_cube(dataset)
    pixels = cube.reshape(-1, cube.shape[2])
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    if np.any(std == 0):
        raise ValueError("Zero-variance spectral band")
    standardized = (cube - mean[None, None, :]) / std[None, None, :]
    view = np.lib.stride_tricks.sliding_window_view(
        standardized, window_shape=(PATCH, PATCH), axis=(0, 1)
    )
    # H' x W' x bands x patch x patch -> H' x W' x patch x patch x bands.
    patches = np.ascontiguousarray(view.transpose(0, 1, 3, 4, 2))
    h, w = patches.shape[:2]
    flat = patches.reshape(h * w, RAW_DIMENSION)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-norm standardized patch")
    normalized = flat / norms[:, None]
    features = np.zeros((len(normalized), PADDED_DIMENSION), dtype=np.float32)
    features[:, :RAW_DIMENSION] = normalized.astype(np.float32)
    rows, cols = np.unravel_index(np.arange(h * w), (h, w))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez(
            handle,
            features=features,
            row=np.asarray(rows, dtype=np.int16),
            col=np.asarray(cols, dtype=np.int16),
            band_mean=mean,
            band_std=std,
        )
    feature_norms = np.linalg.norm(features, axis=1)
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "dataset": str(dataset),
        "dataset_md5": md5,
        "dataset_sha256": digest(dataset, "sha256"),
        "selected_variable": variable,
        "mat_inventory": inventory,
        "cube_shape": list(cube.shape),
        "patch_shape": [PATCH, PATCH, cube.shape[2]],
        "raw_dimension": RAW_DIMENSION,
        "padded_dimension": PADDED_DIMENSION,
        "shape": list(features.shape),
        "finite": bool(np.isfinite(features).all()),
        "zero_norm_vectors": int(np.count_nonzero(feature_norms == 0)),
        "norm_min": float(feature_norms.min()),
        "norm_max": float(feature_norms.max()),
        "script_sha256": digest(Path(__file__), "sha256"),
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output, "sha256"),
        "elapsed_s": time.time() - start,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
