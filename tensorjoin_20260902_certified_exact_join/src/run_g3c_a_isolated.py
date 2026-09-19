#!/usr/bin/env python3
"""Run the frozen G3C-A opportunity gate under physical-GPU0 isolation."""

from __future__ import annotations

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


EXPERIMENT_ID = "tensorjoin_20260903_g3c_a_host_fp32_interval"
RUNNER = PROJECT / "src/run_g3c_a_host_fp32_interval.py"
RESULT = PROJECT / "results/g3c_a_host_fp32_interval.json"
MANIFEST = PROJECT / "results/g3c_a_host_fp32_interval_manifest.json"
RAW = PROJECT / "raw/g3c_a_host_fp32_interval.log"
PREFLIGHT = PROJECT / "raw/g3c_a_host_fp32_interval_preflight.log"
OCCUPANCY = PROJECT / "raw/g3c_a_host_fp32_interval_occupancy.jsonl"
CACHE = PROJECT / "artifacts/g3c_a_g3b_r1_triton_cache"


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    for path in (RESULT, MANIFEST, RAW, PREFLIGHT, OCCUPANCY, CACHE):
        if path.exists():
            raise FileExistsError(path)
    gpu0 = gpu_rows()[0]
    gpu_uuid = str(gpu0["uuid"])
    command = [PYTHON, str(RUNNER)]
    started = time.time()
    target_pids: set[int] = set()
    foreign_rows: list[dict[str, object]] = []
    max_memory_mib = 0
    max_utilization = 0
    CACHE.mkdir(parents=True)
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        PREFLIGHT.write_text(preflight_text(gpu_uuid), encoding="utf-8")
        with OCCUPANCY.open("x", encoding="utf-8") as monitor:
            require_quiescence(gpu_uuid, monitor)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["TRITON_CACHE_DIR"] = str(CACHE)
            with RAW.open("x", encoding="utf-8") as output:
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
                    rows = compute_rows(gpu_uuid)
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
            post_compute = compute_rows(gpu_uuid)
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
    result = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.is_file() else None
    admitted = bool(
        returncode == 0
        and not foreign_rows
        and not post_compute
        and result is not None
        and result.get("g3c_a_gate_pass")
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "isolated_opportunity_gate_not_performance",
        "status": "clean_gate_complete" if admitted else "failed",
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
        "raw_log_sha256": sha256_file(RAW),
        "preflight_log_sha256": sha256_file(PREFLIGHT),
        "occupancy_log_sha256": sha256_file(OCCUPANCY),
        "result_sha256": sha256_file(RESULT) if RESULT.is_file() else None,
        "cache_path": str(CACHE.relative_to(PROJECT)),
        "runner_sha256": sha256_file(RUNNER),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "wall_seconds": time.time() - started,
        "performance_claim_allowed": False,
    }
    atomic_json(MANIFEST, manifest)
    print("G3C_A_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
