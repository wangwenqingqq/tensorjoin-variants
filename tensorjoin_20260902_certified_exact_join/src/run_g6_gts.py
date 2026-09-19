#!/usr/bin/env python3
"""Run and validate the G6 GTS adapter under the frozen GPU1 contract."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("@TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock")
PHYSICAL_GPU = 1
DATASETS = {
    "g2a": {
        "n": 4096,
        "d": 512,
        "epsilon": "0.7541135250198396",
        "npy": ROOT / "data/g2a_cifar4096/vectors_f32.npy",
        "raw": ROOT / "data/g2a_cifar4096/vectors_f32.raw",
        "oracle": ROOT / "data/g2a_cifar4096/oracle_pairs_u64_le.bin",
        "expected_count": 262144,
        "expected_sha256": "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d",
    },
    "g2b": {
        "n": 60000,
        "d": 512,
        "epsilon": "0.62890625",
        "npy": ROOT / "data/g2b_cifar60000/vectors_f32.npy",
        "raw": ROOT / "data/g2b_cifar60000/vectors_f32.raw",
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
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    selected = [row for row in rows if int(row.split(",", 1)[0].strip()) == PHYSICAL_GPU]
    if len(selected) != 1:
        raise RuntimeError("physical GPU1 not found uniquely")
    fields = [field.strip() for field in selected[0].split(",")]
    uuid = fields[1]
    apps_text = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    apps = []
    for row in apps_text.splitlines() if apps_text else []:
        parts = [field.strip() for field in row.split(",")]
        if parts[0] == uuid:
            apps.append(
                {"gpu_uuid": parts[0], "pid": int(parts[1]), "name": parts[2], "memory_mib": int(parts[3])}
            )
    return {
        "index": PHYSICAL_GPU,
        "uuid": uuid,
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "compute_apps": apps,
    }


def ensure_raw(spec: dict[str, object]) -> None:
    raw = Path(spec["raw"])
    expected_bytes = int(spec["n"]) * int(spec["d"]) * 4
    if raw.exists() and raw.stat().st_size == expected_bytes:
        return
    array = np.load(Path(spec["npy"]), allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (int(spec["n"]), int(spec["d"])):
        raise RuntimeError(f"unexpected array contract: {array.dtype} {array.shape}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(raw)
    if raw.stat().st_size != expected_bytes:
        raise RuntimeError("raw conversion byte count mismatch")


def validate_pairs(output: Path, oracle: Path, spec: dict[str, object]) -> dict[str, object]:
    if not output.exists() or output.stat().st_size % 8:
        return {"valid_file": False, "exact": False}
    observed = np.fromfile(output, dtype="<u8")
    reference = np.fromfile(oracle, dtype="<u8")
    n = int(spec["n"])
    sorted_ok = bool(observed.size < 2 or np.all(observed[1:] >= observed[:-1]))
    duplicate_count = int(np.count_nonzero(observed[1:] == observed[:-1])) if observed.size > 1 else 0
    invalid_count = int(np.count_nonzero(observed >= np.uint64(n) * np.uint64(n)))
    common = min(observed.size, reference.size)
    mismatch_indices = np.flatnonzero(observed[:common] != reference[:common])
    examples = []
    for index in mismatch_indices[:10]:
        examples.append(
            {"index": int(index), "observed": int(observed[index]), "reference": int(reference[index])}
        )
    digest = sha256(output)
    expected_digest = str(spec["expected_sha256"])
    exact = bool(
        observed.size == int(spec["expected_count"])
        and observed.size == reference.size
        and digest == expected_digest
        and sorted_ok
        and duplicate_count == 0
        and invalid_count == 0
        and mismatch_indices.size == 0
    )
    return {
        "valid_file": True,
        "pair_count": int(observed.size),
        "expected_pair_count": int(spec["expected_count"]),
        "raw_u64_sha256": digest,
        "expected_raw_u64_sha256": expected_digest,
        "sorted": sorted_ok,
        "duplicate_count": duplicate_count,
        "invalid_count": invalid_count,
        "prefix_mismatch_count": int(mismatch_indices.size),
        "count_delta": int(observed.size) - int(reference.size),
        "first_mismatches": examples,
        "exact": exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--attempt", default="a0")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    binary = ROOT / "adapters/gts_g6_r1/build_sm120/GTS_G6"
    if not binary.exists():
        raise RuntimeError(f"missing binary: {binary}")
    ensure_raw(spec)
    oracle = Path(spec["oracle"])
    if not oracle.exists() or sha256(oracle) != spec["expected_sha256"]:
        raise RuntimeError("oracle missing or hash mismatch")

    (ROOT / "raw").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "artifacts/g6").mkdir(parents=True, exist_ok=True)
    stem = f"g6_gts_{args.dataset}_{args.attempt}"
    pair_path = ROOT / f"artifacts/g6/{stem}_pairs_u64_le.bin"
    log_path = ROOT / f"raw/{stem}.log"
    result_path = ROOT / f"results/{stem}.json"

    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU1 campaign lock is held: {LOCK}") from error
        pre = gpu_snapshot()
        if pre["compute_apps"]:
            raise RuntimeError(f"foreign process on physical GPU1: {pre['compute_apps']}")
        command = [
            str(binary), str(spec["raw"]), str(spec["n"]), str(spec["d"]),
            str(spec["epsilon"]), str(pair_path),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
        wall_start = time.time()
        process = subprocess.run(command, env=environment, text=True, capture_output=True)
        wall_stop = time.time()
        post = gpu_snapshot()

    log_path.write_text(
        "COMMAND " + " ".join(command) + "\n\nSTDOUT\n" + process.stdout
        + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    metrics = {}
    for key, value in re.findall(r"^G6_METRIC ([a-z_]+)=(.+)$", process.stdout, flags=re.MULTILINE):
        metrics[key] = float(value) if key == "elapsed_seconds" else int(value)
    validation = validate_pairs(pair_path, oracle, spec) if process.returncode == 0 else {"exact": False}
    result = {
        "experiment_id": "tensorjoin_20260904_g6_latest_baselines",
        "baseline": "GTS upstream FP32 plus G6 export adapter",
        "upstream_commit": "3bac1b725e92e98e3b69d5cb79ad9c56ccb5e639",
        "dataset": args.dataset,
        "shape": [int(spec["n"]), int(spec["d"])],
        "epsilon": float(spec["epsilon"]),
        "physical_gpu": PHYSICAL_GPU,
        "cuda_visible_devices": str(PHYSICAL_GPU),
        "preflight": pre,
        "postflight": post,
        "command": command,
        "returncode": process.returncode,
        "outer_process_wall_seconds": wall_stop - wall_start,
        "public_metrics": metrics,
        "validation": validation,
        "receipts": {
            "binary_sha256": sha256(binary),
            "input_raw_sha256": sha256(Path(spec["raw"])),
            "oracle_sha256": sha256(oracle),
            "log_sha256": sha256(log_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if process.returncode == 0 and bool(validation.get("exact")) else 1


if __name__ == "__main__":
    sys.exit(main())
