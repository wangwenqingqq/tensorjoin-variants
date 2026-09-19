#!/usr/bin/env python3
"""Run the frozen two-round G2B public-denominator screen under one GPU lock."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


EXPERIMENT_ID = "tensorjoin_20260903_g2b_public_screen"
PYTHON = "@TENSORJOIN_ROOT@/isaacsim6/env/bin/python"
LOCK_PATH = Path("@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock")
QUIESCENCE_SECONDS = 30.0
MONITOR_INTERVAL_SECONDS = 0.25
ORDERS = (
    ("gds", "mistic", "tensorjoin"),
    ("tensorjoin", "mistic", "gds"),
)
RUNNERS = {
    "gds": PROJECT / "src/run_g2b_gds_public.py",
    "mistic": PROJECT / "src/run_g2b_mistic_public.py",
    "tensorjoin": PROJECT / "src/run_g2b_tensorjoin_public.py",
}
MANIFEST = PROJECT / "results/g2b_public_screen_manifest.json"


def run_text(command: list[str], check: bool = True) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed {command}: {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def gpu_rows() -> list[dict[str, object]]:
    fields = (
        "index,uuid,name,driver_version,pstate,power.draw,power.limit,"
        "memory.used,utilization.gpu,clocks.sm,clocks.mem"
    )
    output = run_text(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    keys = fields.split(",")
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(keys):
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {line}")
        rows.append(dict(zip(keys, values)))
    return rows


def compute_rows(gpu_uuid: str) -> list[dict[str, object]]:
    fields = "gpu_uuid,pid,process_name,used_memory"
    output = run_text(
        [
            "nvidia-smi",
            f"--query-compute-apps={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4 or values[0] != gpu_uuid:
            continue
        try:
            pid = int(values[1])
        except ValueError:
            continue
        rows.append(
            {
                "gpu_uuid": values[0],
                "pid": pid,
                "process_name": values[2],
                "used_memory_mib": values[3],
            }
        )
    return rows


def parent_pid(pid: int) -> int | None:
    try:
        # /proc/<pid>/stat field 4 is PPID; comm may contain spaces in parentheses.
        remainder = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1]
        return int(remainder.split()[1])
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return None


def is_descendant_or_self(pid: int, root_pid: int) -> bool:
    current = pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        next_pid = parent_pid(current)
        if next_pid is None:
            return False
        current = next_pid
    return current == root_pid


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
        f"gpu_uuid={gpu_uuid}",
        f"python={PYTHON}",
        f"thread_env={json.dumps({key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, sort_keys=True)}",
    ]
    for label, command in commands.items():
        sections.append(f"\n--- {label} ---\n{run_text(command, check=False)}")
    return "\n".join(sections)


def require_quiescence(gpu_uuid: str, log_handle) -> None:
    started = time.monotonic()
    while True:
        rows = compute_rows(gpu_uuid)
        elapsed = time.monotonic() - started
        log_handle.write(
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
        log_handle.flush()
        if rows:
            raise RuntimeError(f"Physical GPU0 not empty during quiescence: {rows}")
        if elapsed >= QUIESCENCE_SECONDS:
            return
        time.sleep(1.0)


def terminate_own_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def result_path(method: str, record_id: str) -> Path:
    return PROJECT / f"results/g2b_public_screen_{method}_{record_id}.json"


def run_one(round_index: int, position: int, method: str, gpu_uuid: str) -> dict[str, object]:
    record_id = f"r{round_index}_p{position}"
    stem = f"g2b_public_screen_r{round_index}_p{position}_{method}"
    raw_log = PROJECT / f"raw/{stem}.log"
    preflight_log = PROJECT / f"raw/{stem}_preflight.log"
    occupancy_log = PROJECT / f"raw/{stem}_occupancy.jsonl"
    expected_result = result_path(method, record_id)
    for path in (raw_log, preflight_log, occupancy_log, expected_result):
        if path.exists():
            raise FileExistsError(path)
    preflight_log.write_text(preflight_text(gpu_uuid), encoding="utf-8")

    with occupancy_log.open("x", encoding="utf-8") as monitor:
        require_quiescence(gpu_uuid, monitor)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        command = [
            PYTHON,
            str(RUNNERS[method]),
            "--record-id",
            record_id,
        ]
        started = time.time()
        foreign_rows: list[dict[str, object]] = []
        target_gpu_pids: set[int] = set()
        max_memory_used_mib = 0
        max_utilization = 0
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
        post_gpu = gpu_rows()[0]
        post_compute = compute_rows(gpu_uuid)
        monitor.write(
            json.dumps(
                {
                    "phase": "postflight",
                    "gpu": post_gpu,
                    "compute_rows": post_compute,
                    "wall_time": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )

    record: dict[str, object] = {
        "round": round_index,
        "position": position,
        "method": method,
        "record_id": record_id,
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
    if returncode != 0 or foreign_rows or not expected_result.is_file():
        record["admitted"] = False
        record["failure"] = "process failure, foreign contamination, or missing result"
        return record
    result = json.loads(expected_result.read_text(encoding="utf-8"))
    exact = bool(result.get("correctness", {}).get("exact_contract_pass"))
    record.update(
        {
            "admitted": exact,
            "result": str(expected_result.relative_to(PROJECT)),
            "result_sha256": sha256_file(expected_result),
            "public_seconds": result.get("public_seconds"),
            "exact_contract_pass": exact,
        }
    )
    if not exact:
        record["failure"] = "exact output contract failed"
    return record


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Expected live gpu-host-8 host gpu-host-8, got {platform.node()}")
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    for runner in RUNNERS.values():
        if not runner.is_file():
            raise FileNotFoundError(runner)
    gpu0 = gpu_rows()[0]
    if str(gpu0["index"]) != "0":
        raise RuntimeError(f"Unexpected GPU0 row: {gpu0}")
    gpu_uuid = str(gpu0["uuid"])
    records: list[dict[str, object]] = []
    campaign_started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for round_index, order in enumerate(ORDERS):
            for position, method in enumerate(order):
                record = run_one(round_index, position, method, gpu_uuid)
                records.append(record)
                print("PROCESS_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
                if not record["admitted"]:
                    manifest = {
                        "experiment_id": EXPERIMENT_ID,
                        "status": "stopped_on_failed_or_contaminated_process",
                        "orders": ORDERS,
                        "host": platform.node(),
                        "gpu0": gpu0,
                        "lock_path": str(LOCK_PATH),
                        "quiescence_seconds": QUIESCENCE_SECONDS,
                        "records": records,
                        "wall_seconds": time.time() - campaign_started,
                        "runner_sha256": sha256_file(Path(__file__).resolve()),
                    }
                    atomic_json(MANIFEST, manifest)
                    return 2
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "six_clean_exact_processes_complete",
        "orders": ORDERS,
        "host": platform.node(),
        "gpu0": gpu0,
        "lock_path": str(LOCK_PATH),
        "quiescence_seconds": QUIESCENCE_SECONDS,
        "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "records": records,
        "wall_seconds": time.time() - campaign_started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(MANIFEST, manifest)
    print("SCREEN_PROCESSES_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
