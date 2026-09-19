#!/usr/bin/env python3
"""Launch one G3B-R1 validation with GPU0 isolation and monitoring."""

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
    LOCK_PATH,
    MONITOR_INTERVAL_SECONDS,
    PYTHON,
    compute_rows,
    gpu_rows,
    is_descendant_or_self,
    preflight_text,
    require_quiescence,
    terminate_own_group,
)


RUNNER = PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True, choices=range(2))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    result_path = PROJECT / f"results/g3b_r1_gpu_analytic_process_{args.run_id}.json"
    manifest_path = PROJECT / f"results/g3b_r1_gpu_analytic_process_{args.run_id}_manifest.json"
    raw_path = PROJECT / f"raw/g3b_r1_gpu_analytic_process_{args.run_id}.log"
    preflight_path = PROJECT / f"raw/g3b_r1_gpu_analytic_process_{args.run_id}_preflight.log"
    occupancy_path = PROJECT / f"raw/g3b_r1_gpu_analytic_process_{args.run_id}_occupancy.jsonl"
    cache_path = PROJECT / f"artifacts/g3b_r1_triton_cache_run{args.run_id}"
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
    gpu0 = gpu_rows()[0]
    uuid = str(gpu0["uuid"])
    started = time.time()
    target_pids: set[int] = set()
    foreign_rows: list[dict[str, object]] = []
    max_memory_mib = 0
    max_utilization = 0
    command = [PYTHON, str(RUNNER), "--run-id", str(args.run_id)]
    cache_path.mkdir(parents=True)
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        preflight_path.write_text(preflight_text(uuid), encoding="utf-8")
        with occupancy_path.open("x", encoding="utf-8") as monitor:
            require_quiescence(uuid, monitor)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
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
                while process.poll() is None:
                    rows = compute_rows(uuid)
                    for row in rows:
                        pid = int(row["pid"])
                        if pid in target_pids or is_descendant_or_self(pid, process.pid):
                            target_pids.add(pid)
                        else:
                            foreign_rows.append(row)
                        try:
                            max_memory_mib = max(
                                max_memory_mib,
                                int(str(row["used_memory_mib"]).split()[0]),
                            )
                        except ValueError:
                            pass
                    current_gpu0 = gpu_rows()[0]
                    try:
                        max_utilization = max(
                            max_utilization, int(current_gpu0["utilization.gpu"])
                        )
                    except ValueError:
                        pass
                    monitor.write(
                        json.dumps(
                            {
                                "phase": "run",
                                "gpu": current_gpu0,
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
            post_compute = compute_rows(uuid)
            monitor.write(
                json.dumps(
                    {
                        "phase": "postflight",
                        "gpu": gpu_rows()[0],
                        "compute_rows": post_compute,
                        "wall_time": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None
    )
    admitted = bool(
        returncode == 0
        and not foreign_rows
        and not post_compute
        and result is not None
        and result.get("process_gate_pass")
    )
    manifest = {
        "experiment_id": "tensorjoin_20260903_gpu_analytic_certificate_g3b_r1",
        "run_id": args.run_id,
        "status": "clean_validation_process_complete" if admitted else "failed",
        "admitted": admitted,
        "host": platform.node(),
        "gpu0": gpu0,
        "command": command,
        "returncode": returncode,
        "target_gpu_pids": sorted(target_pids),
        "foreign_rows": foreign_rows,
        "postflight_compute_rows": post_compute,
        "external_monitor_max_process_memory_mib": max_memory_mib,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
        "lock_path": str(LOCK_PATH),
        "quiescence_seconds": 30.0,
        "raw_log_sha256": sha256_file(raw_path),
        "preflight_log_sha256": sha256_file(preflight_path),
        "occupancy_log_sha256": sha256_file(occupancy_path),
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "cache_path": str(cache_path.relative_to(PROJECT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "wall_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
