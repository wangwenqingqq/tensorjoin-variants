#!/usr/bin/env python3
"""Run the frozen CPU-only ESC-50 certificate-geometry experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


EXPERIMENT_ID = "tensorjoin_20260902_int8_certificate_esc50_a0"
SOURCE_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
SEED = 20260902
QUERY_COUNT = 512
BASE_COUNT = 4096
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
NPERSEG = 512
NOVERLAP = 256
POOL_F = 32
POOL_T = 32
DIMENSION = POOL_F * POOL_T
TARGET_RESULTS_PER_QUERY = (1, 8, 64)
CONTAINMENT_GUARD = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = dataset_root / "meta" / "esc50.csv"
    rows: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["filename"]] = row
    return rows


def to_float_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        denominator = float(max(abs(info.min), info.max))
        return samples.astype(np.float32) / denominator
    return samples.astype(np.float32, copy=False)


def mean_pool_axis(matrix: np.ndarray, target: int, axis: int) -> np.ndarray:
    chunks = np.array_split(np.arange(matrix.shape[axis]), target)
    if any(len(chunk) == 0 for chunk in chunks):
        raise ValueError(
            f"Cannot pool axis length {matrix.shape[axis]} into {target} bins"
        )
    pooled = [np.take(matrix, chunk, axis=axis).mean(axis=axis) for chunk in chunks]
    return np.stack(pooled, axis=axis)


def window_feature(samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
    _, _, spectrum = signal.stft(
        samples,
        fs=sample_rate,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        boundary=None,
        padded=False,
    )
    feature = np.log1p(np.abs(spectrum).astype(np.float64))
    feature = mean_pool_axis(feature, POOL_F, axis=0)
    feature = mean_pool_axis(feature, POOL_T, axis=1)
    vector = feature.reshape(-1)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm):
        raise ValueError("Encountered a non-finite feature")
    if norm == 0:
        return None
    return (vector / norm).astype(np.float32)


def build_feature_cache(dataset_root: Path, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    audio_paths = sorted((dataset_root / "audio").glob("*.wav"))
    if len(audio_paths) != 2000:
        raise ValueError(f"Expected 2000 ESC-50 WAV files, found {len(audio_paths)}")
    metadata = load_metadata(dataset_root)
    features: list[np.ndarray] = []
    clip_names: list[str] = []
    categories: list[str] = []
    starts: list[int] = []
    sample_rates: set[int] = set()
    excluded_zero_norm: list[tuple[str, int]] = []
    start_time = time.time()
    for index, audio_path in enumerate(audio_paths, start=1):
        sample_rate, raw_samples = wavfile.read(audio_path)
        sample_rates.add(int(sample_rate))
        samples = to_float_mono(raw_samples)
        window_size = int(round(WINDOW_SECONDS * sample_rate))
        hop_size = int(round(HOP_SECONDS * sample_rate))
        if len(samples) < window_size:
            raise ValueError(f"Clip shorter than one window: {audio_path}")
        for start in range(0, len(samples) - window_size + 1, hop_size):
            feature = window_feature(samples[start : start + window_size], sample_rate)
            if feature is None:
                excluded_zero_norm.append((audio_path.name, start))
                print(
                    f"EXCLUDE_ZERO_NORM clip={audio_path.name} start_sample={start}",
                    flush=True,
                )
                continue
            features.append(feature)
            clip_names.append(audio_path.name)
            categories.append(metadata[audio_path.name]["category"])
            starts.append(start)
        if index % 100 == 0:
            print(
                f"FEATURE_PROGRESS clips={index}/{len(audio_paths)} "
                f"windows={len(features)} elapsed_s={time.time()-start_time:.3f}",
                flush=True,
            )
    feature_matrix = np.stack(features)
    if feature_matrix.shape[1] != DIMENSION:
        raise AssertionError(feature_matrix.shape)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=feature_matrix,
        clip=np.asarray(clip_names),
        category=np.asarray(categories),
        start_sample=np.asarray(starts, dtype=np.int64),
        sample_rates=np.asarray(sorted(sample_rates), dtype=np.int64),
        excluded_zero_norm=np.asarray(excluded_zero_norm, dtype=str).reshape(-1, 2),
    )
    print(
        f"FEATURE_CACHE path={cache_path} sha256={sha256_file(cache_path)} "
        f"shape={feature_matrix.shape} sample_rates={sorted(sample_rates)} "
        f"excluded_zero_norm={len(excluded_zero_norm)}",
        flush=True,
    )
    return feature_matrix, np.asarray(clip_names)


def load_or_build_features(
    dataset_root: Path, cache_path: Path, rebuild: bool
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"]
            clips = data["clip"]
        print(
            f"FEATURE_CACHE_REUSED path={cache_path} sha256={sha256_file(cache_path)} "
            f"shape={features.shape}",
            flush=True,
        )
        return features, clips
    return build_feature_cache(dataset_root, cache_path)


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
        clip = permuted_clips[cursor]
        query_indices.extend(np.flatnonzero(clips == clip).tolist())
        cursor += 1
    while len(base_indices) < BASE_COUNT:
        clip = permuted_clips[cursor]
        base_indices.extend(np.flatnonzero(clips == clip).tolist())
        cursor += 1
    query_indices = rng.permutation(query_indices)[:QUERY_COUNT]
    base_indices = rng.permutation(base_indices)[:BASE_COUNT]
    query_clips = clips[query_indices]
    base_clips = clips[base_indices]
    overlap = np.intersect1d(np.unique(query_clips), np.unique(base_clips))
    if len(overlap):
        raise AssertionError(f"Clip leakage: {overlap[:5]}")
    return (
        np.ascontiguousarray(features[query_indices]),
        np.ascontiguousarray(features[base_indices]),
        query_clips,
        base_clips,
    )


def squared_distances_float64(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64)
    x = base.astype(np.float64)
    distances = (
        np.sum(q * q, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (q @ x.T)
    )
    np.maximum(distances, 0.0, out=distances)
    return distances


def quantize_per_vector(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    scales = max_abs / np.float32(127.0)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    codes = np.rint(vectors / scales[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    reconstruction = (codes.astype(np.float32) * scales[:, None]).astype(np.float32)
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    residual_norm = np.sqrt(np.sum(residual * residual, axis=1))
    residual_norm = np.nextafter(residual_norm, np.inf)
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


def calibrated_thresholds(exact_d2: np.ndarray) -> list[tuple[int, float]]:
    flat = exact_d2.reshape(-1)
    thresholds: list[tuple[int, float]] = []
    for target in TARGET_RESULTS_PER_QUERY:
        desired_count = min(len(flat), exact_d2.shape[0] * target)
        threshold = float(np.partition(flat, desired_count - 1)[desired_count - 1])
        thresholds.append((target, threshold))
    return thresholds


def evaluate_threshold(
    target: int,
    threshold_d2: float,
    exact_d2: np.ndarray,
    reconstructed_d: np.ndarray,
    q_error: np.ndarray,
    x_error: np.ndarray,
) -> dict[str, int | float]:
    radius = q_error[:, None] + x_error[None, :]
    lower = np.maximum(0.0, reconstructed_d - radius) ** 2
    upper = (reconstructed_d + radius) ** 2
    lower = np.nextafter(lower, -np.inf)
    upper = np.nextafter(upper, np.inf)
    exact_inside = exact_d2 <= threshold_d2
    accept = upper <= threshold_d2
    reject = lower > threshold_d2
    ambiguous = ~(accept | reject)
    final_inside = accept | (ambiguous & exact_inside)
    total = exact_d2.size
    exact_count = int(np.count_nonzero(exact_inside))
    ambiguous_count = int(np.count_nonzero(ambiguous))
    result = {
        "target_results_per_query": target,
        "threshold_d2": threshold_d2,
        "exact_pairs": exact_count,
        "exact_results_per_query": exact_count / exact_d2.shape[0],
        "direct_accept_pairs": int(np.count_nonzero(accept)),
        "direct_reject_pairs": int(np.count_nonzero(reject)),
        "ambiguous_pairs": ambiguous_count,
        "ambiguous_fraction": ambiguous_count / total,
        "ambiguous_to_output": ambiguous_count / max(exact_count, 1),
        "lower_containment_violations": int(
            np.count_nonzero(exact_d2 + CONTAINMENT_GUARD < lower)
        ),
        "upper_containment_violations": int(
            np.count_nonzero(exact_d2 - CONTAINMENT_GUARD > upper)
        ),
        "false_accept_pairs": int(np.count_nonzero(accept & ~exact_inside)),
        "false_reject_pairs": int(np.count_nonzero(reject & exact_inside)),
        "final_classification_mismatch": int(np.count_nonzero(final_inside != exact_inside)),
    }
    print("THRESHOLD " + json.dumps(result, sort_keys=True), flush=True)
    return result


def write_threshold_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    results_dir = project_root / "results"
    data_dir = project_root / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "esc50_windows_logstft_32x32.npz"
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_commit": SOURCE_COMMIT,
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "seed": SEED,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "dataset_root": str(dataset_root),
                "project_root": str(project_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(dataset_root / "audio")
    start = time.time()
    features, clips = load_or_build_features(dataset_root, cache_path, args.rebuild_features)
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
    accumulator_limit = DIMENSION * 127 * 127
    accumulator_pass = accumulator_limit <= np.iinfo(np.int32).max
    quant_start = time.time()
    reconstructed_d = reconstructed_distances(q_codes, q_scales, x_codes, x_scales)
    print(f"INT_DISTANCE elapsed_s={time.time()-quant_start:.6f}", flush=True)
    rows = [
        evaluate_threshold(target, threshold, exact_d2, reconstructed_d, q_error, x_error)
        for target, threshold in calibrated_thresholds(exact_d2)
    ]
    ambiguity = [float(row["ambiguous_fraction"]) for row in rows]
    exactness_pass = all(
        row["lower_containment_violations"] == 0
        and row["upper_containment_violations"] == 0
        and row["false_accept_pairs"] == 0
        and row["false_reject_pairs"] == 0
        and row["final_classification_mismatch"] == 0
        for row in rows
    )
    ambiguity_pass = max(ambiguity) <= 0.05 and float(np.median(ambiguity)) <= 0.01
    gate_pass = bool(exactness_pass and ambiguity_pass and accumulator_pass)
    source_bytes = int((query.nbytes + base.nbytes))
    hot_bytes = int(
        q_codes.nbytes
        + x_codes.nbytes
        + q_scales.nbytes
        + x_scales.nbytes
        + q_error.nbytes
        + x_error.nbytes
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": SOURCE_COMMIT,
        "host": platform.node(),
        "seed": SEED,
        "query_shape": list(query.shape),
        "base_shape": list(base.shape),
        "query_unique_clips": int(len(np.unique(query_clips))),
        "base_unique_clips": int(len(np.unique(base_clips))),
        "clip_overlap": int(len(np.intersect1d(np.unique(query_clips), np.unique(base_clips)))),
        "accumulator_worst_case": accumulator_limit,
        "accumulator_pass": bool(accumulator_pass),
        "max_ambiguity_fraction": max(ambiguity),
        "median_ambiguity_fraction": float(np.median(ambiguity)),
        "exactness_pass": bool(exactness_pass),
        "ambiguity_pass": bool(ambiguity_pass),
        "gate_pass": gate_pass,
        "hot_representation_bytes": hot_bytes,
        "fp32_source_bytes": source_bytes,
        "hot_to_fp32_ratio": hot_bytes / source_bytes,
        "retained_total_to_fp32_ratio": (source_bytes + hot_bytes) / source_bytes,
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "containment_guard": CONTAINMENT_GUARD,
        "thresholds": rows,
        "elapsed_s": time.time() - start,
    }
    summary_path = results_dir / "a0_summary.json"
    threshold_path = results_dir / "a0_thresholds.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_threshold_csv(threshold_path, rows)
    print("RUN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"RUN_END gate_pass={gate_pass} elapsed_s={summary['elapsed_s']:.3f}", flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
