#!/usr/bin/env python3
"""Run TensorJoin memcheck and 1,000-iteration stability gates safely."""

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


EXPERIMENT_ID = "tensorjoin_20260903_g2b_tensorjoin_safety"
RUNNER = PROJECT / "src/run_g2b_tensorjoin_safety.py"
SANITIZER = "/usr/local/bin/compute-sanitizer"
MANIFEST = PROJECT / "results/g2b_tensorjoin_safety_manifest.json"


def run_guarded(
    label: str,
    command: list[str],
    result_path: Path,
    gpu_uuid: str,
) -> dict[str, object]:
    raw_log = PROJECT / f"raw/g2b_tensorjoin_safety_{label}.log"
    preflight_log = PROJECT / f"raw/g2b_tensorjoin_safety_{label}_preflight.log"
    occupancy_log = PROJECT / f"raw/g2b_tensorjoin_safety_{label}_occupancy.jsonl"
    for path in (raw_log, preflight_log, occupancy_log, result_path):
        if path.exists():
            raise FileExistsError(path)
    preflight_log.write_text(preflight_text(gpu_uuid), encoding="utf-8")
    foreign_rows: list[dict[str, object]] = []
    target_gpu_pids: set[int] = set()
    max_memory_used_mib = 0
    max_utilization = 0
    with occupancy_log.open("x", encoding="utf-8") as monitor:
        require_quiescence(gpu_uuid, monitor)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        started = time.time()
        with raw_log.open("x", encoding="utf-8") as output:
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
                    if is_descendant_or_self(pid, process.pid):
                        target_gpu_pids.add(pid)
                    else:
                        foreign_rows.append(row)
                    try:
                        max_memory_used_mib = max(
                            max_memory_used_mib,
                            int(str(row["used_memory_mib"]).split()[0]),
                        )
                    except ValueError:
                        pass
                gpu0 = gpu_rows()[0]
                try:
                    max_utilization = max(max_utilization, int(gpu0["utilization.gpu"]))
                except ValueError:
                    pass
                monitor.write(
                    json.dumps(
                        {
                            "phase": "run",
                            "elapsed_s": time.time() - started,
                            "gpu": gpu0,
                            "compute_rows": rows,
                            "target_root_pid": process.pid,
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
    record: dict[str, object] = {
        "label": label,
        "command": command,
        "returncode": returncode,
        "foreign_rows": foreign_rows,
        "target_gpu_pids": sorted(target_gpu_pids),
        "external_monitor_max_process_memory_mib": max_memory_used_mib,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
        "raw_log": str(raw_log.relative_to(PROJECT)),
        "raw_log_sha256": sha256_file(raw_log),
        "preflight_log": str(preflight_log.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_log),
        "occupancy_log": str(occupancy_log.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_log),
        "postflight_compute_rows": post_compute,
    }
    if returncode == 0 and not foreign_rows and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record.update(
            {
                "result": str(result_path.relative_to(PROJECT)),
                "result_sha256": sha256_file(result_path),
                "safety_pass": bool(result.get("safety_pass")),
            }
        )
    else:
        record["safety_pass"] = False
    record["admitted"] = bool(record["safety_pass"] and not foreign_rows)
    return record


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Expected live gpu-host-8 host gpu-host-8, got {platform.node()}")
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    for path in (RUNNER, Path(SANITIZER)):
        if not path.is_file():
            raise FileNotFoundError(path)
    gpu0 = gpu_rows()[0]
    gpu_uuid = str(gpu0["uuid"])
    memcheck_result = PROJECT / "results/g2b_tensorjoin_safety_memcheck.json"
    stress_result = PROJECT / "results/g2b_tensorjoin_safety_stress1000.json"
    records: list[dict[str, object]] = []
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        memcheck_command = [
            SANITIZER,
            "--tool",
            "memcheck",
            "--target-processes",
            "all",
            "--error-exitcode",
            "99",
            "--leak-check",
            "full",
            PYTHON,
            str(RUNNER),
            "--record-id",
            "memcheck",
            "--iterations",
            "2",
        ]
        memcheck = run_guarded(
            "memcheck", memcheck_command, memcheck_result, gpu_uuid
        )
        memcheck_text = (PROJECT / memcheck["raw_log"]).read_text(
            encoding="utf-8", errors="replace"
        )
        memcheck["sanitizer_zero_error_summary"] = "ERROR SUMMARY: 0 errors" in memcheck_text
        memcheck["admitted"] = bool(
            memcheck["admitted"] and memcheck["sanitizer_zero_error_summary"]
        )
        records.append(memcheck)
        print("SAFETY_PROCESS_COMPLETE " + json.dumps(memcheck, sort_keys=True), flush=True)
        if memcheck["admitted"]:
            stress_command = [
                PYTHON,
                str(RUNNER),
                "--record-id",
                "stress1000",
                "--iterations",
                "1000",
            ]
            stress = run_guarded(
                "stress1000", stress_command, stress_result, gpu_uuid
            )
            records.append(stress)
            print("SAFETY_PROCESS_COMPLETE " + json.dumps(stress, sort_keys=True), flush=True)
    safety_gate_pass = len(records) == 2 and all(record["admitted"] for record in records)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "host": platform.node(),
        "gpu0": gpu0,
        "lock_path": str(LOCK_PATH),
        "quiescence_seconds_before_each_process": 30.0,
        "records": records,
        "safety_gate_pass": safety_gate_pass,
        "wall_seconds": time.time() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(MANIFEST, manifest)
    print("SAFETY_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if safety_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

