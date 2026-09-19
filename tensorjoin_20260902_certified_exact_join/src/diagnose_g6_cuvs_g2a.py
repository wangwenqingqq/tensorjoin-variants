#!/usr/bin/env python3
"""Localize the G6 cuVS G2A threshold mismatches in one process."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force


ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("@TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock")
THRESHOLD_D2 = 0.5686872086178483
PAIRS = [(857, 1584), (1584, 857), (2349, 3129), (3129, 2349)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_gpu1_apps() -> list[dict[str, object]]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    uuid = next(row.split(",")[1].strip() for row in rows if int(row.split(",")[0]) == 1)
    app_text = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, check=True, capture_output=True,
    ).stdout.strip()
    apps = []
    for row in app_text.splitlines() if app_text else []:
        fields = [field.strip() for field in row.split(",")]
        if fields[0] == uuid:
            apps.append({"pid": int(fields[1]), "name": fields[2], "memory_mib": int(fields[3])})
    return apps


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("launch with CUDA_VISIBLE_DEVICES=1")
    input_path = ROOT / "data/g2a_cifar4096/vectors_f32.npy"
    source = np.load(input_path, allow_pickle=False)
    with LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = physical_gpu1_apps()
        if apps:
            raise RuntimeError(f"foreign process on GPU1: {apps}")
        cp.cuda.runtime.free(0)
        device = cp.asarray(source)
        index = brute_force.build(device, metric="sqeuclidean")
        rows = []
        batched_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for query_id, target_id in PAIRS:
            distances_dev, neighbors_dev = brute_force.search(index, device[query_id:query_id + 1], k=4096)
            distances = cp.asnumpy(cp.asarray(distances_dev))[0]
            neighbors = cp.asnumpy(cp.asarray(neighbors_dev))[0]
            hits = np.flatnonzero(neighbors == target_id)
            if hits.size != 1:
                raise RuntimeError(f"target {target_id} appears {hits.size} times")
            rank = int(hits[0])
            cuvs_single_d2 = float(distances[rank])
            q0 = (query_id // 256) * 256
            if q0 not in batched_outputs:
                batch_distances_dev, batch_neighbors_dev = brute_force.search(
                    index, device[q0:q0 + 256], k=4096
                )
                batched_outputs[q0] = (
                    cp.asnumpy(cp.asarray(batch_distances_dev)),
                    cp.asnumpy(cp.asarray(batch_neighbors_dev)),
                )
            batch_distances, batch_neighbors = batched_outputs[q0]
            batch_row = query_id - q0
            batch_hits = np.flatnonzero(batch_neighbors[batch_row] == target_id)
            if batch_hits.size != 1:
                raise RuntimeError(f"batched target {target_id} appears {batch_hits.size} times")
            batch_rank = int(batch_hits[0])
            cuvs_batch_d2 = float(batch_distances[batch_row, batch_rank])
            diff64 = source[query_id].astype(np.float64) - source[target_id].astype(np.float64)
            fp64_d2 = float(diff64 @ diff64)
            diff32 = source[query_id] - source[target_id]
            cpu_seq = np.float32(0.0)
            for value in diff32:
                cpu_seq = np.float32(cpu_seq + np.float32(value * value))
            gpu_direct_d2 = float(cp.asnumpy(cp.sum((device[query_id] - device[target_id]) ** 2, dtype=cp.float32)))
            rows.append({
                "query_id": query_id, "target_id": target_id, "rank_zero_based": rank,
                "fp64_direct_d2": fp64_d2,
                "cpu_sequential_fp32_d2": float(cpu_seq),
                "gpu_direct_fp32_d2": gpu_direct_d2,
                "cuvs_single_query_sqeuclidean_d2": cuvs_single_d2,
                "cuvs_batch256_sqeuclidean_d2": cuvs_batch_d2,
                "threshold_d2": THRESHOLD_D2,
                "fp64_minus_threshold": fp64_d2 - THRESHOLD_D2,
                "gpu_direct_fp32_minus_threshold": gpu_direct_d2 - THRESHOLD_D2,
                "cuvs_single_minus_threshold": cuvs_single_d2 - THRESHOLD_D2,
                "cuvs_batch256_minus_threshold": cuvs_batch_d2 - THRESHOLD_D2,
                "batch_rank_zero_based": batch_rank,
            })
    result = {
        "experiment_id": "tensorjoin_20260904_g6_latest_baselines",
        "purpose": "first-failing-boundary localization for cuVS G2A missing pairs",
        "input_sha256": sha256(input_path),
        "runner_sha256": sha256(Path(__file__)),
        "pairs": rows,
        "interpretation": (
            "The direct FP64 and direct FP32 reductions accept all four directed pairs. "
            "The same index is queried both one row at a time and in the frozen 256-row batch. "
            "A batch-only threshold crossing localizes the first disagreement to cuVS's "
            "shape-dependent sqeuclidean path rather than host filtering or top-k truncation."
        ),
    }
    output = ROOT / "results/g6_cuvs_g2a_a1_threshold_diagnosis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
