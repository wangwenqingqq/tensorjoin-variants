#!/usr/bin/env python3
"""Run FaSTED validation or the frozen eight-round G2B public campaign."""

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


EXPERIMENT_ID = "tensorjoin_20260903_g2b_public_formal"
ORDERS = (
    ("gds", "mistic", "fasted", "tensorjoin"),
    ("mistic", "fasted", "tensorjoin", "gds"),
    ("fasted", "tensorjoin", "gds", "mistic"),
    ("tensorjoin", "gds", "mistic", "fasted"),
    ("tensorjoin", "fasted", "mistic", "gds"),
    ("fasted", "mistic", "gds", "tensorjoin"),
    ("mistic", "gds", "tensorjoin", "fasted"),
    ("gds", "tensorjoin", "fasted", "mistic"),
)
RUNNERS = {
    "gds": PROJECT / "src/run_g2b_gds_public.py",
    "mistic": PROJECT / "src/run_g2b_mistic_public.py",
    "fasted": PROJECT / "src/run_g2b_fasted_public.py",
    "tensorjoin": PROJECT / "src/run_g2b_tensorjoin_public.py",
}
FORMAL_MANIFEST = PROJECT / "results/g2b_public_formal_manifest.json"
VALIDATION_MANIFEST = PROJECT / "results/g2b_public_fasted_validation_manifest.json"
MAX_CONTAMINATION_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fasted-validation", "formal"), required=True)
    return parser.parse_args()


def expected_result(method: str, phase: str, record_id: str) -> Path:
    return PROJECT / f"results/g2b_public_{phase}_{method}_{record_id}.json"


def command_for(method: str, phase: str, record_id: str) -> list[str]:
    return [
        PYTHON,
        str(RUNNERS[method]),
        "--record-id",
        record_id,
        "--phase",
        phase,
    ]


def run_one(
    *,
    label: str,
    method: str,
    phase: str,
    record_id: str,
    gpu_uuid: str,
    round_index: int | None,
    position: int | None,
    attempt: int,
) -> dict[str, object]:
    raw_log = PROJECT / f"raw/{label}.log"
    preflight_log = PROJECT / f"raw/{label}_preflight.log"
    occupancy_log = PROJECT / f"raw/{label}_occupancy.jsonl"
    result_path = expected_result(method, phase, record_id)
    for path in (raw_log, preflight_log, occupancy_log, result_path):
        if path.exists():
            raise FileExistsError(path)
    preflight_log.write_text(preflight_text(gpu_uuid), encoding="utf-8")
    command = command_for(method, phase, record_id)
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
                    # nvidia-smi can retain a row briefly after /proc/<pid>
                    # disappears.  Once a PID has been attributed to this
                    # process tree, keep that attribution for the rest of the
                    # run instead of reclassifying the stale row as foreign.
                    if pid in target_gpu_pids or is_descendant_or_self(pid, process.pid):
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
        "round": round_index,
        "position": position,
        "attempt": attempt,
        "phase": phase,
        "method": method,
        "record_id": record_id,
        "command": command,
        "returncode": returncode,
        "foreign_rows": foreign_rows,
        "contaminated": bool(foreign_rows),
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
    if returncode != 0 or foreign_rows or not result_path.is_file():
        record["admitted"] = False
        record["failure"] = "process failure, contamination, or missing result"
        return record
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if method == "fasted":
        contract_pass = bool(result.get("structural_context_pass"))
    else:
        contract_pass = bool(result.get("correctness", {}).get("exact_contract_pass"))
    record.update(
        {
            "admitted": contract_pass,
            "contract_pass": contract_pass,
            "result": str(result_path.relative_to(PROJECT)),
            "result_sha256": sha256_file(result_path),
            "public_seconds": result.get("public_seconds"),
        }
    )
    if method == "fasted":
        record["quality"] = {
            key: result[key]
            for key in (
                "output_pairs",
                "canonical_raw_u64_sha256",
                "exact_intersection_pairs",
                "exact_only_pairs",
                "approximate_only_pairs",
                "precision",
                "recall",
                "f1",
            )
        }
    if not contract_pass:
        record["failure"] = "output/capacity contract failed"
    return record


def write_manifest(path: Path, mode: str, status: str, gpu0, records, started) -> None:
    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": mode,
            "status": status,
            "orders": ORDERS if mode == "formal" else None,
            "host": platform.node(),
            "gpu0": gpu0,
            "lock_path": str(LOCK_PATH),
            "quiescence_seconds_before_each_process": 30.0,
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "max_contamination_attempts_per_slot": MAX_CONTAMINATION_ATTEMPTS,
            "records": records,
            "wall_seconds": time.time() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G2_FORMAL.md"),
        },
    )


def main() -> int:
    args = parse_args()
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Expected live gpu-host-8 host gpu-host-8, got {platform.node()}")
    manifest_path = VALIDATION_MANIFEST if args.mode == "fasted-validation" else FORMAL_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    for runner in RUNNERS.values():
        if not runner.is_file():
            raise FileNotFoundError(runner)
    safety = json.loads(
        (PROJECT / "results/g2b_tensorjoin_safety_manifest.json").read_text(encoding="utf-8")
    )
    if not safety.get("safety_gate_pass"):
        raise RuntimeError("TensorJoin safety gate is not accepted")
    gpu0 = gpu_rows()[0]
    gpu_uuid = str(gpu0["uuid"])
    records: list[dict[str, object]] = []
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if args.mode == "fasted-validation":
            record = run_one(
                label="g2b_public_fasted_validation",
                method="fasted",
                phase="validation",
                record_id="adaptercheck",
                gpu_uuid=gpu_uuid,
                round_index=None,
                position=None,
                attempt=0,
            )
            records.append(record)
            print("PROCESS_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
            status = "clean_context_validation_complete" if record["admitted"] else "validation_failed"
            write_manifest(manifest_path, args.mode, status, gpu0, records, started)
            return 0 if record["admitted"] else 2

        for round_index, order in enumerate(ORDERS):
            for position, method in enumerate(order):
                admitted = False
                for attempt in range(MAX_CONTAMINATION_ATTEMPTS):
                    record_id = f"r{round_index}_p{position}_a{attempt}"
                    label = f"g2b_public_formal_r{round_index}_p{position}_{method}_a{attempt}"
                    record = run_one(
                        label=label,
                        method=method,
                        phase="formal",
                        record_id=record_id,
                        gpu_uuid=gpu_uuid,
                        round_index=round_index,
                        position=position,
                        attempt=attempt,
                    )
                    records.append(record)
                    print("PROCESS_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
                    if record["admitted"]:
                        admitted = True
                        break
                    if not record["contaminated"]:
                        write_manifest(
                            manifest_path,
                            args.mode,
                            "stopped_on_non_contamination_failure",
                            gpu0,
                            records,
                            started,
                        )
                        return 2
                if not admitted:
                    write_manifest(
                        manifest_path,
                        args.mode,
                        "stopped_after_contamination_retry_limit",
                        gpu0,
                        records,
                        started,
                    )
                    return 2
    write_manifest(
        manifest_path,
        args.mode,
        "eight_clean_four_method_rounds_complete",
        gpu0,
        records,
        started,
    )
    print(f"FORMAL_PROCESSES_COMPLETE manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
