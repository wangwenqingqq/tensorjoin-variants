#!/usr/bin/env python3
"""Run one G5 process on an isolated physical GPU and preserve its full trace."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


EXPERIMENT_ID = "tensorjoin_20260903_g5_unified_public_cifar60k"
QUIESCENCE_SECONDS = 30.0
MONITOR_INTERVAL_SECONDS = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-result", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    safe = args.label.replace("_", "").replace("-", "").isalnum()
    if not safe or not args.command:
        raise ValueError("Unsafe label or empty command")
    expected = (PROJECT / args.expected_result).resolve()
    try:
        expected.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError("Expected result must be inside the project") from error
    args.expected_result = expected
    return args


def run_text(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return completed.stdout + completed.stderr


def gpu_row(physical_gpu: int) -> dict[str, str]:
    fields = (
        "index,uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "memory.used,memory.total,utilization.gpu"
    )
    text = run_text(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--id={physical_gpu}",
        ]
    ).strip()
    values = [value.strip() for value in text.split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        raise RuntimeError(f"Unexpected nvidia-smi row: {text}")
    return dict(zip(names, values))


def compute_rows(gpu_uuid: str) -> list[dict[str, str]]:
    fields = "gpu_uuid,pid,process_name,used_memory"
    text = run_text(
        [
            "nvidia-smi",
            f"--query-compute-apps={fields}",
            "--format=csv,noheader",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4 or values[0] != gpu_uuid:
            continue
        rows.append(
            {
                "gpu_uuid": values[0],
                "pid": values[1],
                "process_name": values[2],
                "used_memory": values[3],
            }
        )
    return rows


def parent_pid(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[3])
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return None


def is_descendant_or_self(pid: int, root_pid: int) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        parent = parent_pid(current)
        if parent is None:
            return False
        current = parent
    return current == root_pid


def preflight_text(gpu_uuid: str) -> str:
    return (
        f"timestamp={time.time()}\n"
        + run_text(["nvidia-smi", "-L"])
        + "\n"
        + run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pstate,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        + "\n"
        + run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        )
        + f"\ntarget_gpu_uuid={gpu_uuid}\n"
    )


def require_quiescence(gpu_uuid: str, physical_gpu: int, monitor) -> None:
    started = time.monotonic()
    while True:
        rows = compute_rows(gpu_uuid)
        state = gpu_row(physical_gpu)
        monitor.write(
            json.dumps(
                {
                    "phase": "quiescence",
                    "elapsed_s": time.monotonic() - started,
                    "gpu": state,
                    "compute_rows": rows,
                    "wall_time": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        monitor.flush()
        if rows:
            raise RuntimeError(
                f"Physical GPU{physical_gpu} is occupied before G5: {rows}"
            )
        if time.monotonic() - started >= QUIESCENCE_SECONDS:
            return
        time.sleep(MONITOR_INTERVAL_SECONDS)


def terminate_own_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Expected gpu-host-8 live host, got {platform.node()}")
    label = args.label
    physical_gpu = args.physical_gpu
    lock_path = Path(f"/tmp/tensorjoin_g5_gpu{physical_gpu}.lock")
    raw_log = PROJECT / f"raw/g5_{label}.log"
    preflight_log = PROJECT / f"raw/g5_{label}_preflight.log"
    occupancy_log = PROJECT / f"raw/g5_{label}_occupancy.jsonl"
    guard_result = PROJECT / f"results/g5_guard_{label}.json"
    cache = PROJECT / f"artifacts/g5_{label}_triton_cache"
    for path in (
        raw_log,
        preflight_log,
        occupancy_log,
        guard_result,
        args.expected_result,
        cache,
    ):
        if path.exists():
            raise FileExistsError(path)

    state_before_lock = gpu_row(physical_gpu)
    gpu_uuid = state_before_lock["uuid"]
    foreign_rows: list[dict[str, str]] = []
    target_gpu_pids: set[int] = set()
    max_memory_mib = 0
    max_utilization = 0
    started = time.time()
    lock_path.touch(exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        preflight_log.write_text(preflight_text(gpu_uuid), encoding="utf-8")
        with occupancy_log.open("x", encoding="utf-8") as monitor:
            require_quiescence(gpu_uuid, physical_gpu, monitor)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            env["G5_PHYSICAL_GPU"] = str(physical_gpu)
            env["TRITON_CACHE_DIR"] = str(cache)
            with raw_log.open("x", encoding="utf-8") as output:
                output.write("COMMAND " + json.dumps(args.command) + "\n")
                output.write(
                    "ENV "
                    + json.dumps(
                        {
                            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                            "G5_PHYSICAL_GPU": env["G5_PHYSICAL_GPU"],
                            "TRITON_CACHE_DIR": env["TRITON_CACHE_DIR"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
                process = subprocess.Popen(
                    args.command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    rows = compute_rows(gpu_uuid)
                    for row in rows:
                        pid = int(row["pid"])
                        if pid in target_gpu_pids or is_descendant_or_self(
                            pid, process.pid
                        ):
                            target_gpu_pids.add(pid)
                        else:
                            foreign_rows.append(row)
                        try:
                            max_memory_mib = max(
                                max_memory_mib,
                                int(row["used_memory"].split()[0]),
                            )
                        except (ValueError, IndexError):
                            pass
                    state = gpu_row(physical_gpu)
                    try:
                        max_utilization = max(
                            max_utilization, int(state["utilization.gpu"])
                        )
                    except ValueError:
                        pass
                    monitor.write(
                        json.dumps(
                            {
                                "phase": "run",
                                "elapsed_s": time.time() - started,
                                "gpu": state,
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
            state_after = gpu_row(physical_gpu)
            monitor.write(
                json.dumps(
                    {
                        "phase": "postflight",
                        "gpu": state_after,
                        "compute_rows": post_compute,
                        "wall_time": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    record: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "guard_and_isolation_evidence_not_performance",
        "label": label,
        "host": platform.node(),
        "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid,
        "gpu_state_before_lock": state_before_lock,
        "gpu_state_after": state_after,
        "lock_path": str(lock_path),
        "quiescence_seconds": QUIESCENCE_SECONDS,
        "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "command": args.command,
        "returncode": returncode,
        "foreign_rows": foreign_rows,
        "target_gpu_pids": sorted(target_gpu_pids),
        "external_monitor_max_process_memory_mib": max_memory_mib,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
        "postflight_compute_rows": post_compute,
        "raw_log": str(raw_log.relative_to(PROJECT)),
        "raw_log_sha256": sha256_file(raw_log),
        "preflight_log": str(preflight_log.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_log),
        "occupancy_log": str(occupancy_log.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_log),
        "triton_cache": str(cache.relative_to(PROJECT)) if cache.exists() else None,
        "expected_result": str(args.expected_result.relative_to(PROJECT)),
        "wall_seconds": time.time() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(
            PROJECT / "PROTOCOL_G5_COMPATIBILITY_SCREEN.md"
        ),
    }
    if returncode == 0 and not foreign_rows and args.expected_result.is_file():
        child = json.loads(args.expected_result.read_text(encoding="utf-8"))
        exact = bool(child.get("correctness", {}).get("exact_contract_pass"))
        record.update(
            {
                "expected_result_sha256": sha256_file(args.expected_result),
                "child_exact_contract_pass": exact,
                "admitted": exact,
            }
        )
    else:
        record.update({"child_exact_contract_pass": False, "admitted": False})
    atomic_json(guard_result, record)
    print("GUARD_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
