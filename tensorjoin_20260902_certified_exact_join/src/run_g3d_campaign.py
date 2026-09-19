#!/usr/bin/env python3
"""Run the frozen G3D screen or formal campaign under GPU0 isolation."""

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


EXPERIMENT_ID = "tensorjoin_20260903_g3d_complete_pipeline_timing"
RUNNER = PROJECT / "src/run_g3d_complete_pipeline_timing.py"
ORDERS = {
    "screen": ("KC", "CK"),
    "formal": ("KC", "CK", "KC", "CK", "CK", "KC", "CK", "KC"),
}
SETTINGS = {
    "screen": {"warmups": 5, "observations": 20, "sustained_launches": 100},
    "formal": {"warmups": 20, "observations": 100, "sustained_launches": 1_000},
}
MAX_CONTAMINATION_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    return parser.parse_args()


def run_one(
    phase: str,
    process_id: int,
    order: str,
    attempt: int,
    gpu_uuid: str,
) -> dict[str, object]:
    label = f"g3d_{phase}_process_{process_id}_{order.lower()}_a{attempt}"
    result_path = PROJECT / f"results/{label}.json"
    raw_path = PROJECT / f"raw/{label}.log"
    preflight_path = PROJECT / f"raw/{label}_preflight.log"
    occupancy_path = PROJECT / f"raw/{label}_occupancy.jsonl"
    cache_path = PROJECT / f"artifacts/{label}_triton_cache"
    for path in (result_path, raw_path, preflight_path, occupancy_path, cache_path):
        if path.exists():
            raise FileExistsError(path)

    settings = SETTINGS[phase]
    command = [
        PYTHON,
        str(RUNNER),
        "--phase",
        phase,
        "--process-id",
        str(process_id),
        "--attempt",
        str(attempt),
        "--order",
        order,
        "--warmups",
        str(settings["warmups"]),
        "--observations",
        str(settings["observations"]),
        "--sustained-launches",
        str(settings["sustained_launches"]),
    ]
    preflight_path.write_text(preflight_text(gpu_uuid), encoding="utf-8")
    cache_path.mkdir(parents=True)
    foreign_rows: list[dict[str, object]] = []
    target_gpu_pids: set[int] = set()
    max_memory_mib = 0
    max_utilization = 0
    started = time.time()
    with occupancy_path.open("x", encoding="utf-8") as monitor:
        require_quiescence(gpu_uuid, monitor)
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
                rows = compute_rows(gpu_uuid)
                for row in rows:
                    pid = int(row["pid"])
                    if pid in target_gpu_pids or is_descendant_or_self(pid, process.pid):
                        target_gpu_pids.add(pid)
                    else:
                        foreign_rows.append(row)
                    try:
                        max_memory_mib = max(
                            max_memory_mib,
                            int(str(row["used_memory_mib"]).split()[0]),
                        )
                    except ValueError:
                        pass
                gpu0 = gpu_rows()[0]
                try:
                    max_utilization = max(
                        max_utilization, int(gpu0["utilization.gpu"])
                    )
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
                            "target_pids": sorted(target_gpu_pids),
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
        and result.get("correctness_pass")
        and result.get("process_measurement_pass")
    )
    return {
        "phase": phase,
        "process_id": process_id,
        "order": order,
        "attempt": attempt,
        "admitted": admitted,
        "returncode": returncode,
        "command": command,
        "foreign_rows": foreign_rows,
        "contaminated": bool(foreign_rows),
        "postflight_compute_rows": post_compute,
        "target_gpu_pids": sorted(target_gpu_pids),
        "external_monitor_max_process_memory_mib": max_memory_mib,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
        "result": str(result_path.relative_to(PROJECT)) if result_path.is_file() else None,
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "raw_log": str(raw_path.relative_to(PROJECT)),
        "raw_log_sha256": sha256_file(raw_path),
        "preflight_log": str(preflight_path.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_path),
        "occupancy_log": str(occupancy_path.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_path),
        "cache_path": str(cache_path.relative_to(PROJECT)),
        "wall_seconds": time.time() - started,
    }


def write_manifest(
    path: Path,
    phase: str,
    status: str,
    gpu0: dict[str, object],
    records: list[dict[str, object]],
    started: float,
) -> None:
    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "status": status,
            "orders": ORDERS[phase],
            "settings": SETTINGS[phase],
            "host": platform.node(),
            "gpu0": gpu0,
            "lock_path": str(LOCK_PATH),
            "quiescence_seconds_before_each_process": 30.0,
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "max_contamination_attempts_per_slot": MAX_CONTAMINATION_ATTEMPTS,
            "records": records,
            "runner_sha256": sha256_file(RUNNER),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3D.md"),
            "wall_seconds": time.time() - started,
        },
    )


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    manifest_path = PROJECT / f"results/g3d_{args.phase}_campaign_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if args.phase == "formal":
        screen_summary = PROJECT / "results/g3d_screen_summary.json"
        screen_audit = PROJECT / "results/g3d_screen_generated_code_audit.json"
        if not screen_summary.is_file():
            raise FileNotFoundError(screen_summary)
        if not screen_audit.is_file():
            raise FileNotFoundError(screen_audit)
        screen = json.loads(screen_summary.read_text(encoding="utf-8"))
        audit = json.loads(screen_audit.read_text(encoding="utf-8"))
        if not screen.get("screen_gate_pass"):
            raise RuntimeError("G3D screen did not admit formal timing")
        if not audit.get("screen_runtime_binary_gate_pass"):
            raise RuntimeError("G3D screen runtime binaries do not match accepted cubins")

    gpu0 = gpu_rows()[0]
    gpu_uuid = str(gpu0["uuid"])
    records: list[dict[str, object]] = []
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for process_id, order in enumerate(ORDERS[args.phase]):
            admitted = False
            for attempt in range(MAX_CONTAMINATION_ATTEMPTS):
                record = run_one(args.phase, process_id, order, attempt, gpu_uuid)
                records.append(record)
                print("PROCESS_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
                if record["admitted"]:
                    admitted = True
                    break
                if not record["contaminated"]:
                    write_manifest(
                        manifest_path,
                        args.phase,
                        "stopped_on_non_contamination_failure",
                        gpu0,
                        records,
                        started,
                    )
                    return 2
            if not admitted:
                write_manifest(
                    manifest_path,
                    args.phase,
                    "stopped_after_contamination_retry_limit",
                    gpu0,
                    records,
                    started,
                )
                return 2

    write_manifest(
        manifest_path,
        args.phase,
        f"{len(ORDERS[args.phase])}_clean_paired_processes_complete",
        gpu0,
        records,
        started,
    )
    print(f"CAMPAIGN_COMPLETE manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
