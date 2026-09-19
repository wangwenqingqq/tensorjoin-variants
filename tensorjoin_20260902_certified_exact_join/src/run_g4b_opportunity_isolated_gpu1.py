#!/usr/bin/env python3
"""Run one G4B dataset opportunity matrix under physical GPU1 isolation."""

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
    terminate_own_group,
)
from run_g4b_public_opportunity import DATASETS


EXPERIMENT_ID = "tensorjoin_20260903_g4b_public_breadth_opportunity"
RUNNER = PROJECT / "src/run_g4b_public_opportunity.py"
LOCK_PATH = Path("@TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock")
QUIESCENCE_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return completed.stdout + completed.stderr


def preflight_text(gpu_uuid: str) -> str:
    commands = {
        "date": ["date", "--iso-8601=seconds"],
        "hostname": ["hostname"],
        "who": ["who"],
        "uptime": ["uptime"],
        "nvidia_smi": ["nvidia-smi"],
        "compute_apps": [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        "top_cpu": [
            "bash",
            "-lc",
            "ps -eo user,pid,ppid,pcpu,pmem,etime,comm --sort=-pcpu | head -25",
        ],
        "nvcc": ["/usr/local/cuda-13.1/bin/nvcc", "--version"],
    }
    sections = [
        f"experiment_id={EXPERIMENT_ID}",
        f"project={PROJECT}",
        f"physical_gpu=1",
        f"gpu_uuid={gpu_uuid}",
        f"python={PYTHON}",
    ]
    for label, command in commands.items():
        sections.append(f"\n--- {label} ---\n{run_text(command)}")
    return "\n".join(sections)


def require_quiescence(gpu_uuid: str, monitor) -> None:
    started = time.monotonic()
    while True:
        rows = compute_rows(gpu_uuid)
        elapsed = time.monotonic() - started
        monitor.write(
            json.dumps(
                {
                    "phase": "quiescence",
                    "elapsed_s": elapsed,
                    "compute_rows": rows,
                    "wall_time": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        monitor.flush()
        if rows:
            raise RuntimeError(f"Physical GPU1 is occupied: {rows}")
        if elapsed >= QUIESCENCE_SECONDS:
            return
        time.sleep(1.0)


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    label = f"g4b_opportunity_{args.dataset}"
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
    command = [PYTHON, str(RUNNER), "--dataset", args.dataset]
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
                        current_gpu1 = gpu_rows()[1]
                        try:
                            max_utilization = max(
                                max_utilization,
                                int(current_gpu1["utilization.gpu"]),
                            )
                        except ValueError:
                            pass
                        monitor.write(
                            json.dumps(
                                {
                                    "phase": "run",
                                    "gpu": current_gpu1,
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
                            "gpu": gpu_rows()[1],
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
        and result.get("process_gate_pass")
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "clean_dataset_complete"
            if admitted
            else "prelaunch_blocked"
            if not runner_was_launched and foreign_rows
            else "dataset_failed"
        ),
        "dataset_id": args.dataset,
        "admitted": admitted,
        "runner_was_launched": runner_was_launched,
        "returncode": returncode,
        "command": command,
        "gpu1": gpu1,
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
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4B_OPPORTUNITY.md"),
        "wall_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    print("DATASET_SLOT_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
