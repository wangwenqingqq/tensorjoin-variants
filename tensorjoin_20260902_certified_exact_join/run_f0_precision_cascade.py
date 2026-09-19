#!/usr/bin/env python3
"""Audit an exact INT8-to-FP32-to-FP64 similarity-join cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "tensorjoin_20260903_precision_cascade_f0"
QUERY_COUNT = 512
BASE_COUNT = 4096
TARGETS = (1, 64)
BOUND_PAD = 1e-4
FP32_GUARD = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--hsi-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            value.update(block)
    return value.hexdigest()


def audio_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    features, clips = data["features"], data["clip"]
    rng = np.random.default_rng(20260902)
    units = rng.permutation(np.unique(clips))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(clips == units[cursor]).tolist()); cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(clips == units[cursor]).tolist()); cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    return np.ascontiguousarray(features[query_ids]), np.ascontiguousarray(features[base_ids])


def video_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    features, clips, groups = data["features"], data["clip"], data["group"]
    rng = np.random.default_rng(20260903)
    units = rng.permutation(np.unique(groups))
    query_ids: list[int] = []
    base_ids: list[int] = []
    cursor = 0
    while len(query_ids) < QUERY_COUNT:
        query_ids.extend(np.flatnonzero(groups == units[cursor]).tolist()); cursor += 1
    while len(base_ids) < BASE_COUNT:
        base_ids.extend(np.flatnonzero(groups == units[cursor]).tolist()); cursor += 1
    query_ids = rng.permutation(query_ids)[:QUERY_COUNT]
    base_ids = rng.permutation(base_ids)[:BASE_COUNT]
    if len(np.intersect1d(groups[query_ids], groups[base_ids])):
        raise AssertionError("Video group overlap")
    if len(np.intersect1d(clips[query_ids], clips[base_ids])):
        raise AssertionError("Video clip overlap")
    return np.ascontiguousarray(features[query_ids]), np.ascontiguousarray(features[base_ids])


def hsi_split(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    features, rows = data["features"], data["row"]
    rng = np.random.default_rng(20260903)
    query_ids = rng.choice(np.flatnonzero(rows <= 47), QUERY_COUNT, replace=False)
    base_ids = rng.choice(np.flatnonzero(rows >= 51), BASE_COUNT, replace=False)
    if int(rows[query_ids].max()) + 2 >= int(rows[base_ids].min()):
        raise AssertionError("HSI source-pixel overlap")
    return np.ascontiguousarray(features[query_ids]), np.ascontiguousarray(features[base_ids])


def direct_d2(query: np.ndarray, base: np.ndarray, dtype: np.dtype) -> np.ndarray:
    q = query.astype(dtype)
    x = base.astype(dtype)
    output = np.empty((len(q), len(x)), dtype=dtype)
    block = max(1, 8192 // query.shape[1])
    for start in range(0, len(q), block):
        stop = min(start + block, len(q))
        delta = q[start:stop, None, :] - x[None, :, :]
        output[start:stop] = np.einsum(
            "ijk,ijk->ij", delta, delta, dtype=dtype, optimize=True
        )
    return output


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32)
    max_abs = np.max(np.abs(vectors), axis=1)
    scales = np.where(max_abs == 0, np.float32(1.0), max_abs / np.float32(127.0)).astype(np.float32)
    codes = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    reconstruction = codes.astype(np.float32) * scales[:, None]
    residual = vectors.astype(np.float64) - reconstruction.astype(np.float64)
    errors = np.nextafter(np.linalg.norm(residual, axis=1), np.inf)
    return codes, scales, errors


def reconstructed_distance(
    qc: np.ndarray, qs: np.ndarray, xc: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    dot = qc.astype(np.int32) @ xc.astype(np.int32).T
    qn = np.sum(qc.astype(np.int64) ** 2, axis=1) * qs.astype(np.float64) ** 2
    xn = np.sum(xc.astype(np.int64) ** 2, axis=1) * xs.astype(np.float64) ** 2
    d2 = qn[:, None] + xn[None, :] - 2.0 * dot * qs.astype(np.float64)[:, None] * xs.astype(np.float64)[None, :]
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def midgap(exact: np.ndarray, target: int) -> dict[str, float]:
    flat = exact.reshape(-1)
    rank = QUERY_COUNT * target - 1
    lower = float(np.partition(flat, rank)[rank])
    upper = float(np.min(flat[flat > lower]))
    return {
        "lower_distance_d2": lower,
        "upper_distance_d2": upper,
        "gap_d2": upper - lower,
        "threshold_d2": lower + (upper - lower) / 2.0,
    }


def evaluate(name: str, query: np.ndarray, base: np.ndarray) -> dict[str, object]:
    exact = direct_d2(query, base, np.float64)
    fp32 = direct_d2(query, base, np.float32)
    qc, qs, qe = quantize(query)
    xc, xs, xe = quantize(base)
    reconstructed = reconstructed_distance(qc, qs, xc, xs)
    radius = qe[:, None] + xe[None, :] + BOUND_PAD
    lower = np.nextafter(np.maximum(reconstructed - radius, 0.0) ** 2, -np.inf)
    upper = np.nextafter((reconstructed + radius) ** 2, np.inf)
    targets: dict[str, object] = {}
    for target in TARGETS:
        info = midgap(exact, target)
        threshold = info["threshold_d2"]
        oracle = exact <= threshold
        int8_accept = upper <= threshold
        int8_reject = lower > threshold
        int8_ambiguous = ~(int8_accept | int8_reject)
        qid, xid = np.nonzero(int8_ambiguous)
        bitwise_equal = np.all(query[qid] == base[xid], axis=1)
        fp32_values = fp32[qid, xid]
        fp32_accept = bitwise_equal | (fp32_values <= threshold - FP32_GUARD)
        fp32_reject = (~bitwise_equal) & (fp32_values > threshold + FP32_GUARD)
        fp64_needed = ~(fp32_accept | fp32_reject)
        direct_accept_ids = qid[fp32_accept] * BASE_COUNT + xid[fp32_accept]
        direct_reject_ids = qid[fp32_reject] * BASE_COUNT + xid[fp32_reject]
        oracle_flat = oracle.reshape(-1)
        false_accept = int(np.count_nonzero(~oracle_flat[direct_accept_ids]))
        false_reject = int(np.count_nonzero(oracle_flat[direct_reject_ids]))
        final = int8_accept.copy()
        final[qid[fp32_accept], xid[fp32_accept]] = True
        refine_q = qid[fp64_needed]
        refine_x = xid[fp64_needed]
        final[refine_q, refine_x] = oracle[refine_q, refine_x]
        mismatch = int(np.count_nonzero(final != oracle))
        ambiguous_count = int(np.count_nonzero(int8_ambiguous))
        fp64_count = int(np.count_nonzero(fp64_needed))
        row = {
            "target_results_per_query": target,
            "actual_results_per_query": int(np.count_nonzero(oracle)) / QUERY_COUNT,
            **info,
            "int8_ambiguous_pairs": ambiguous_count,
            "bitwise_equal_pairs": int(np.count_nonzero(bitwise_equal)),
            "fp32_direct_accept_pairs": int(np.count_nonzero(fp32_accept)),
            "fp32_direct_reject_pairs": int(np.count_nonzero(fp32_reject)),
            "fp64_refine_pairs": fp64_count,
            "fp64_fraction_of_int8_ambiguous": fp64_count / max(ambiguous_count, 1),
            "fp64_work_reduction": ambiguous_count / max(fp64_count, 1),
            "fp32_false_accepts": false_accept,
            "fp32_false_rejects": false_reject,
            "final_mismatch": mismatch,
        }
        row["gate_pass"] = bool(
            false_accept == 0
            and false_reject == 0
            and mismatch == 0
            and row["fp64_fraction_of_int8_ambiguous"] <= 0.25
        )
        print("CELL " + json.dumps({"modality": name, **row}, sort_keys=True), flush=True)
        targets[str(target)] = row
    return {
        "shape": [QUERY_COUNT, BASE_COUNT, int(query.shape[1])],
        "targets": targets,
        "gate_pass": all(bool(row["gate_pass"]) for row in targets.values()),
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("F0 must be CPU-only")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite F0 evidence")
    start = time.time()
    caches = {
        "audio_panns_2048d": args.audio_cache,
        "video_r3d18_512d": args.video_cache,
        "hsi_indian_pines_1984d": args.hsi_cache,
    }
    print("RUN_START " + json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cache_sha256": {key: sha256_file(path) for key, path in caches.items()},
        "script_sha256": sha256_file(Path(__file__)),
        "bound_pad": BOUND_PAD,
        "fp32_guard": FP32_GUARD,
    }, sort_keys=True), flush=True)
    with np.load(args.audio_cache, allow_pickle=False) as data:
        audio = audio_split(data)
    with np.load(args.video_cache, allow_pickle=False) as data:
        video = video_split(data)
    with np.load(args.hsi_cache, allow_pickle=False) as data:
        hsi = hsi_split(data)
    modalities = {
        "audio_panns_2048d": evaluate("audio_panns_2048d", *audio),
        "video_r3d18_512d": evaluate("video_r3d18_512d", *video),
        "hsi_indian_pines_1984d": evaluate("hsi_indian_pines_1984d", *hsi),
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "host": platform.node(),
        "cache_sha256": {key: sha256_file(path) for key, path in caches.items()},
        "script_sha256": sha256_file(Path(__file__)),
        "bound_pad": BOUND_PAD,
        "fp32_guard": FP32_GUARD,
        "modalities": modalities,
        "gate_pass": all(bool(row["gate_pass"]) for row in modalities.values()),
        "elapsed_s": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
