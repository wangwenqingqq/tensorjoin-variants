#!/usr/bin/env python3
"""Run one frozen G4A dynamic-router timing slot under GPU0 isolation."""

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


EXPERIMENT_ID = "tensorjoin_20260903_g4a_dynamic_count_router"
RUNNER = PROJECT / "src/run_g4a_dynamic_router_timing.py"
ORDERS = {
    "screen": ("KC", "CK"),
    "formal": ("KC", "CK", "KC", "CK", "CK", "KC", "CK", "KC"),
}
SETTINGS = {
    "screen": {"warmups": 10, "observations": 50, "sustained_launches": 200},
    "formal": {"warmups": 20, "observations": 200, "sustained_launches": 1_000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True, choices=range(3))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    if args.process_id not in range(len(ORDERS[args.phase])):
        raise RuntimeError("Process ID is outside the frozen schedule")
    if args.phase == "formal":
        summary = json.loads(
            (PROJECT / "results/g4a_screen_summary.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (PROJECT / "results/g4a_screen_runtime_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if not summary.get("screen_gate_pass") or not audit.get(
            "runtime_binary_gate_pass"
        ):
            raise RuntimeError("G4A screen has not admitted formal execution")

    order = ORDERS[args.phase][args.process_id]
    label = f"g4a_{args.phase}_process_{args.process_id}_{order.lower()}_a{args.attempt}"
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

    gpu0 = gpu_rows()[0]
    gpu_uuid = str(gpu0["uuid"])
    settings = SETTINGS[args.phase]
    command = [
        PYTHON,
        str(RUNNER),
        "--phase",
        args.phase,
        "--process-id",
        str(args.process_id),
        "--attempt",
        str(args.attempt),
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
    started = time.time()
    foreign_rows: list[dict[str, object]] = []
    target_pids: set[int] = set()
    max_memory_mib = 0
    max_utilization = 0
    returncode: int | None = None
    post_compute: list[dict[str, object]] = []
    runner_was_launched = False

    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with occupancy_path.open("x", encoding="utf-8") as monitor:
            try:
                require_quiescence(gpu_uuid, monitor)
            except RuntimeError:
                foreign_rows = compute_rows(gpu_uuid)
                monitor.write(
                    json.dumps(
                        {
                            "phase": "prelaunch_blocked",
                            "compute_rows": foreign_rows,
                            "wall_time": time.time(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:
                cache_path.mkdir(parents=True)
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
                    runner_was_launched = True
                    while process.poll() is None:
                        rows = compute_rows(gpu_uuid)
                        for row in rows:
                            pid = int(row["pid"])
                            if pid in target_pids or is_descendant_or_self(
                                pid, process.pid
                            ):
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
                                max_utilization,
                                int(current_gpu0["utilization.gpu"]),
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

    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None
    )
    admitted = bool(
        runner_was_launched
        and returncode == 0
        and not foreign_rows
        and not post_compute
        and result is not None
        and result.get("correctness_pass")
        and result.get("process_measurement_pass")
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "clean_slot_complete"
            if admitted
            else "prelaunch_blocked"
            if not runner_was_launched and foreign_rows
            else "slot_failed"
        ),
        "phase": args.phase,
        "process_id": args.process_id,
        "attempt": args.attempt,
        "order": order,
        "settings": settings,
        "admitted": admitted,
        "runner_was_launched": runner_was_launched,
        "returncode": returncode,
        "command": command,
        "gpu0": gpu0,
        "foreign_rows": foreign_rows,
        "postflight_compute_rows": post_compute,
        "target_gpu_pids": sorted(target_pids),
        "external_monitor_max_process_memory_mib": max_memory_mib,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
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
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4A.md"),
        "wall_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    print("SLOT_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
