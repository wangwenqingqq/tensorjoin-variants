#!/usr/bin/env python3
"""Run cuVS brute force under the frozen G6 exact self-join contract."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force


ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("@TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock")
PHYSICAL_GPU = 1
K = 4096
BATCH_ROWS = 256
DATASETS = {
    "g2a": {
        "n": 4096,
        "d": 512,
        "epsilon": 0.7541135250198396,
        "threshold_d2": 0.5686872086178483,
        "npy": ROOT / "data/g2a_cifar4096/vectors_f32.npy",
        "oracle": ROOT / "data/g2a_cifar4096/oracle_pairs_u64_le.bin",
        "expected_count": 262144,
        "expected_sha256": "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d",
    },
    "g2b": {
        "n": 60000,
        "d": 512,
        "epsilon": 0.62890625,
        "threshold_d2": 0.3955230712890625,
        "npy": ROOT / "data/g2b_cifar60000/vectors_f32.npy",
        "oracle": ROOT / "artifacts/g2b/mistic_pairs_u64_le.bin",
        "expected_count": 3926078,
        "expected_sha256": "13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, object]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    selected = [row for row in rows if int(row.split(",", 1)[0].strip()) == PHYSICAL_GPU]
    if len(selected) != 1:
        raise RuntimeError("physical GPU1 not found uniquely")
    fields = [field.strip() for field in selected[0].split(",")]
    uuid = fields[1]
    app_text = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, check=True, capture_output=True,
    ).stdout.strip()
    apps = []
    for row in app_text.splitlines() if app_text else []:
        parts = [field.strip() for field in row.split(",")]
        if parts[0] == uuid:
            apps.append({"gpu_uuid": parts[0], "pid": int(parts[1]),
                         "name": parts[2], "memory_mib": int(parts[3])})
    return {"index": PHYSICAL_GPU, "uuid": uuid, "name": fields[2],
            "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            "compute_apps": apps}


def package_versions() -> dict[str, str]:
    names = ["cuvs-cu13", "libcuvs-cu13", "pylibraft-cu13", "libraft-cu13",
             "rmm-cu13", "librmm-cu13", "cupy-cuda13x", "numpy",
             "nvidia-cublas", "nvidia-cuda-nvrtc"]
    return {name: importlib.metadata.version(name) for name in names}


def compare_pairs(observed: np.ndarray, oracle: Path,
                  expected_count: int, expected_hash: str,
                  output_path: Path) -> dict[str, object]:
    reference = np.fromfile(oracle, dtype="<u8")
    digest = sha256(output_path)
    extra = np.setdiff1d(observed, reference, assume_unique=True)
    missing = np.setdiff1d(reference, observed, assume_unique=True)
    duplicate_count = int(np.count_nonzero(observed[1:] == observed[:-1])) if observed.size > 1 else 0
    true_positive = int(reference.size - missing.size)
    precision = true_positive / int(observed.size) if observed.size else 0.0
    recall = true_positive / int(reference.size) if reference.size else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = bool(observed.size == expected_count and digest == expected_hash
                 and duplicate_count == 0 and extra.size == 0 and missing.size == 0)
    return {
        "pair_count": int(observed.size), "expected_pair_count": expected_count,
        "count_delta": int(observed.size) - expected_count,
        "raw_u64_sha256": digest, "expected_raw_u64_sha256": expected_hash,
        "duplicate_count": duplicate_count,
        "extra_count": int(extra.size), "missing_count": int(missing.size),
        "true_positive_count": true_positive,
        "pair_precision": precision, "pair_recall": recall, "pair_f1": f1,
        "first_extra_keys": [int(value) for value in extra[:20]],
        "first_missing_keys": [int(value) for value in missing[:20]],
        "exact": exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--attempt", default="a0")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError("launch with CUDA_VISIBLE_DEVICES=1")
    spec = DATASETS[args.dataset]
    host = np.load(Path(spec["npy"]), allow_pickle=False)
    if host.dtype != np.float32 or host.shape != (spec["n"], spec["d"]):
        raise RuntimeError(f"unexpected input contract: {host.dtype} {host.shape}")
    host = np.ascontiguousarray(host)
    oracle = Path(spec["oracle"])
    if not oracle.exists() or sha256(oracle) != spec["expected_sha256"]:
        raise RuntimeError("oracle missing or hash mismatch")

    (ROOT / "raw").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "artifacts/g6").mkdir(parents=True, exist_ok=True)
    stem = f"g6_cuvs_{args.dataset}_{args.attempt}"
    output_path = ROOT / f"artifacts/g6/{stem}_pairs_u64_le.bin"
    result_path = ROOT / f"results/{stem}.json"

    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU1 campaign lock is held: {LOCK}") from error
        pre = gpu_snapshot()
        if pre["compute_apps"]:
            raise RuntimeError(f"foreign process on physical GPU1: {pre['compute_apps']}")
        cp.cuda.runtime.free(0)  # CUDA context creation is excluded by contract.
        cp.cuda.runtime.deviceSynchronize()

        start = time.perf_counter()
        h2d_start = start
        device_data = cp.asarray(host, dtype=cp.float32, order="C")
        cp.cuda.runtime.deviceSynchronize()
        h2d_stop = time.perf_counter()
        index = brute_force.build(device_data, metric="sqeuclidean")
        build_stop = time.perf_counter()
        chunks: list[np.ndarray] = []
        max_accepted = 0
        saturated_rows = 0
        invalid_neighbor_ids = 0
        batch_count = 0
        for q0 in range(0, int(spec["n"]), BATCH_ROWS):
            q1 = min(q0 + BATCH_ROWS, int(spec["n"]))
            distances_dev, neighbors_dev = brute_force.search(
                index, device_data[q0:q1], k=min(K, int(spec["n"]))
            )
            distances = cp.asnumpy(cp.asarray(distances_dev))
            neighbors = cp.asnumpy(cp.asarray(neighbors_dev))
            invalid_neighbor_ids += int(np.count_nonzero((neighbors < 0) | (neighbors >= int(spec["n"]))))
            accepted = distances.astype(np.float64) <= float(spec["threshold_d2"])
            row_counts = accepted.sum(axis=1, dtype=np.int64)
            max_accepted = max(max_accepted, int(row_counts.max(initial=0)))
            saturated_rows += int(np.count_nonzero(row_counts == min(K, int(spec["n"]))))
            rows = np.arange(q0, q1, dtype=np.uint64)[:, None]
            encoded = rows * np.uint64(spec["n"]) + neighbors.astype(np.uint64, copy=False)
            chunks.append(encoded[accepted])
            batch_count += 1
        search_stop = time.perf_counter()
        observed = np.concatenate(chunks).astype("<u8", copy=False)
        observed.sort()
        stop = time.perf_counter()
        post = gpu_snapshot()

    # File output, hashing, and reference-set diagnostics are outside timing.
    observed.tofile(output_path)
    validation = compare_pairs(observed, oracle, int(spec["expected_count"]),
                               str(spec["expected_sha256"]), output_path)
    validation.update({
        "invalid_neighbor_ids": invalid_neighbor_ids,
        "max_accepted_neighbors_per_row": max_accepted,
        "saturated_rows": saturated_rows,
    })
    validation["exact"] = bool(validation["exact"] and invalid_neighbor_ids == 0
                                and saturated_rows == 0)
    result = {
        "experiment_id": "tensorjoin_20260904_g6_latest_baselines",
        "baseline": "NVIDIA cuVS brute_force sqeuclidean",
        "dataset": args.dataset, "shape": [int(spec["n"]), int(spec["d"])],
        "epsilon": float(spec["epsilon"]), "threshold_d2": float(spec["threshold_d2"]),
        "k": K, "query_batch_rows": BATCH_ROWS, "query_batches": batch_count,
        "physical_gpu": PHYSICAL_GPU, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "preflight": pre, "postflight": post,
        "package_versions": package_versions(),
        "public_metrics": {
            "elapsed_seconds": stop - start,
            "h2d_seconds": h2d_stop - h2d_start,
            "build_seconds": build_stop - h2d_stop,
            "search_filter_d2h_seconds": search_stop - build_stop,
            "concatenate_sort_seconds": stop - search_stop,
        },
        "validation": validation,
        "receipts": {
            "runner_sha256": sha256(Path(__file__)),
            "input_npy_sha256": sha256(Path(spec["npy"])),
            "oracle_sha256": sha256(oracle),
            "wheel_manifest_sha256": sha256(ROOT / "artifacts/g6/cuvs_268_linux_wheels.sha256"),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if validation["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
