#!/usr/bin/env python3
"""Run one frozen G4C public-anchor screen slot on isolated physical GPU1."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g2b_public_screen import (
    MONITOR_INTERVAL_SECONDS,
    PYTHON,
    compute_rows,
    gpu_rows,
    is_descendant_or_self,
    preflight_text,
    require_quiescence,
    terminate_own_group,
)
from run_g4b_r1_public_opportunity import DATASETS


RUNNER = PROJECT / "src/run_g4c_public_anchor_timing.py"
LOCK_PATH = Path("@TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock")
ORDERS = ("KC", "CK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--process-id", type=int, required=True, choices=range(2))
    parser.add_argument("--attempt", type=int, default=0, choices=range(3))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    order = ORDERS[args.process_id]
    label = (
        f"g4c_screen_{args.dataset}_n4096_k64_process_"
        f"{args.process_id}_{order.lower()}_a{args.attempt}"
    )
    result_path = PROJECT / f"results/{label}.json"
    manifest_path = PROJECT / f"results/{label}_manifest.json"
    raw_path = PROJECT / f"raw/{label}.log"
    preflight_path = PROJECT / f"raw/{label}_preflight.log"
    occupancy_path = PROJECT / f"raw/{label}_occupancy.jsonl"
    cache_path = PROJECT / f"artifacts/{label}_triton_cache"
    for path in (
        result_path,
        manifest_path,
        raw_path,
        preflight_path,
        occupancy_path,
        cache_path,
    ):
        if path.exists():
            raise FileExistsError(path)

    gpu1 = gpu_rows()[1]
    gpu_uuid = str(gpu1["uuid"])
    command = [
        PYTHON,
        str(RUNNER),
        "--dataset",
        args.dataset,
        "--phase",
        "screen",
        "--process-id",
        str(args.process_id),
        "--attempt",
        str(args.attempt),
        "--order",
        order,
        "--warmups",
        "10",
        "--observations",
        "50",
        "--sustained-launches",
        "200",
    ]
    preflight_path.write_text(
        preflight_text(gpu_uuid).replace(
            "experiment_id=tensorjoin_20260903_g2b_public_screen",
            "experiment_id=tensorjoin_20260903_g4c_public_anchor_screen",
        ),
        encoding="utf-8",
    )
    started = time.time()
    foreign_rows: list[dict[str, object]] = []
    target_pids: set[int] = set()
    returncode: int | None = None
    post_compute: list[dict[str, object]] = []
    launched = False
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with occupancy_path.open("x", encoding="utf-8") as monitor:
            try:
                require_quiescence(gpu_uuid, monitor)
            except RuntimeError:
                foreign_rows = compute_rows(gpu_uuid)
            else:
                cache_path.mkdir(parents=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "1"
                env["TRITON_CACHE_DIR"] = str(cache_path)
                with raw_path.open("x", encoding="utf-8") as output:
                    output.write("COMMAND " + json.dumps(command) + "\n")
                    output.flush()
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    launched = True
                    while process.poll() is None:
                        rows = compute_rows(gpu_uuid)
                        for row in rows:
                            pid = int(row["pid"])
                            if pid in target_pids or is_descendant_or_self(pid, process.pid):
                                target_pids.add(pid)
                            else:
                                foreign_rows.append(row)
                        monitor.write(
                            json.dumps(
                                {
                                    "phase": "run",
                                    "gpu": gpu_rows()[1],
                                    "compute_rows": rows,
                                    "target_root_pid": process.pid,
                                    "target_pids": sorted(target_pids),
                                    "foreign_rows": foreign_rows,
                                    "wall_time": time.time(),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        monitor.flush()
                        if foreign_rows:
                            terminate_own_group(process)
                            break
                        time.sleep(MONITOR_INTERVAL_SECONDS)
                    returncode = process.wait()
                post_compute = compute_rows(gpu_uuid)

    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None
    )
    admitted = bool(
        launched
        and returncode == 0
        and not foreign_rows
        and not post_compute
        and result is not None
        and result.get("correctness_pass")
        and result.get("process_measurement_pass")
    )
    manifest = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_screen",
        "status": "clean_slot_complete" if admitted else "failed_or_interrupted",
        "dataset_id": args.dataset,
        "process_id": args.process_id,
        "attempt": args.attempt,
        "order": order,
        "admitted": admitted,
        "runner_was_launched": launched,
        "returncode": returncode,
        "command": command,
        "gpu1": gpu1,
        "foreign_rows": foreign_rows,
        "postflight_compute_rows": post_compute,
        "target_gpu_pids": sorted(target_pids),
        "result": str(result_path.relative_to(PROJECT)) if result_path.is_file() else None,
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "raw_log": str(raw_path.relative_to(PROJECT)) if raw_path.is_file() else None,
        "raw_log_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "preflight_log": str(preflight_path.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_path),
        "occupancy_log": str(occupancy_path.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_path),
        "cache_path": str(cache_path.relative_to(PROJECT)) if cache_path.is_dir() else None,
        "runner_sha256": sha256_file(RUNNER),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4C_SCREEN.md"),
        "wall_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
