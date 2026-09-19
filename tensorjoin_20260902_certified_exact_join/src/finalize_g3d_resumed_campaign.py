#!/usr/bin/env python3
"""Assemble immutable G3D slot evidence into the frozen campaign manifest."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g3d_campaign import (
    MAX_CONTAMINATION_ATTEMPTS,
    MONITOR_INTERVAL_SECONDS,
    ORDERS,
    RUNNER,
    SETTINGS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    return parser.parse_args()


def recovered_screen_process_zero() -> dict[str, object]:
    label = "g3d_screen_process_0_kc_a0"
    result_path = PROJECT / f"results/{label}.json"
    raw_path = PROJECT / f"raw/{label}.log"
    preflight_path = PROJECT / f"raw/{label}_preflight.log"
    occupancy_path = PROJECT / f"raw/{label}_occupancy.jsonl"
    cache_path = PROJECT / f"artifacts/{label}_triton_cache"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("correctness_pass") or not result.get("process_measurement_pass"):
        raise RuntimeError("Recovered screen process 0 is not admissible")
    return {
        "phase": "screen",
        "process_id": 0,
        "order": "KC",
        "attempt": 0,
        "admitted": True,
        "returncode": 0,
        "command": None,
        "foreign_rows": [],
        "contaminated": False,
        "postflight_compute_rows": [],
        "target_gpu_pids": [],
        "external_monitor_max_process_memory_mib": None,
        "external_monitor_max_gpu_utilization_percent": None,
        "result": str(result_path.relative_to(PROJECT)),
        "result_sha256": sha256_file(result_path),
        "raw_log": str(raw_path.relative_to(PROJECT)),
        "raw_log_sha256": sha256_file(raw_path),
        "preflight_log": str(preflight_path.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_path),
        "occupancy_log": str(occupancy_path.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_path),
        "cache_path": str(cache_path.relative_to(PROJECT)),
        "wall_seconds": None,
        "recovered_from_interrupted_original_orchestrator": True,
    }


def recovered_screen_prelaunch_block() -> dict[str, object]:
    label = "g3d_screen_process_1_ck_a0"
    preflight_path = PROJECT / f"raw/{label}_preflight.log"
    occupancy_path = PROJECT / f"raw/{label}_occupancy.jsonl"
    cache_path = PROJECT / f"artifacts/{label}_triton_cache"
    lines = [
        json.loads(line)
        for line in occupancy_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    foreign_rows = [row for line in lines for row in line.get("compute_rows", [])]
    if not foreign_rows:
        raise RuntimeError("Expected the preserved prelaunch foreign-GPU blockage")
    return {
        "phase": "screen",
        "process_id": 1,
        "order": "CK",
        "attempt": 0,
        "admitted": False,
        "returncode": None,
        "command": None,
        "foreign_rows": foreign_rows,
        "contaminated": True,
        "contamination_phase": "quiescence_before_runner_launch",
        "runner_was_launched": False,
        "result": None,
        "preflight_log": str(preflight_path.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(preflight_path),
        "occupancy_log": str(occupancy_path.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(occupancy_path),
        "cache_path": str(cache_path.relative_to(PROJECT)),
    }


def resumed_records(phase: str) -> list[dict[str, object]]:
    manifests = sorted((PROJECT / "results").glob(f"g3d_{phase}_slot_*_manifest.json"))
    return [
        json.loads(path.read_text(encoding="utf-8"))["record"] for path in manifests
    ]


def main() -> int:
    args = parse_args()
    output_path = PROJECT / f"results/g3d_{args.phase}_campaign_manifest.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    records = resumed_records(args.phase)
    if args.phase == "screen":
        records = [
            recovered_screen_process_zero(),
            recovered_screen_prelaunch_block(),
            *records,
        ]
    admitted = [record for record in records if record.get("admitted")]
    expected = len(ORDERS[args.phase])
    admitted_by_slot = {int(record["process_id"]): record for record in admitted}
    if len(admitted) != expected or sorted(admitted_by_slot) != list(range(expected)):
        raise RuntimeError("Exactly one clean admitted record is required for every slot")
    for process_id, order in enumerate(ORDERS[args.phase]):
        record = admitted_by_slot[process_id]
        if record["order"] != order:
            raise RuntimeError("Admitted order does not match the frozen schedule")
    attempts_by_slot: dict[int, set[int]] = {}
    for record in records:
        attempts_by_slot.setdefault(int(record["process_id"]), set()).add(
            int(record["attempt"])
        )
    if any(len(attempts) > MAX_CONTAMINATION_ATTEMPTS for attempts in attempts_by_slot.values()):
        raise RuntimeError("A slot exceeds the frozen attempt limit")

    slot_manifest_paths = sorted(
        (PROJECT / "results").glob(f"g3d_{args.phase}_slot_*_manifest.json")
    )
    result = {
        "experiment_id": "tensorjoin_20260903_g3d_complete_pipeline_timing",
        "phase": args.phase,
        "status": f"{expected}_clean_paired_processes_complete_via_resume1",
        "orders": ORDERS[args.phase],
        "settings": SETTINGS[args.phase],
        "host": platform.node(),
        "records": records,
        "admitted_process_ids": sorted(admitted_by_slot),
        "runner_sha256": sha256_file(RUNNER),
        "original_orchestrator_sha256": sha256_file(
            PROJECT / "src/run_g3d_campaign.py"
        ),
        "resume_orchestrator_sha256": sha256_file(
            PROJECT / "src/run_g3d_resumed_slot.py"
        ),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "original_protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3D.md"),
        "resume_protocol_sha256": sha256_file(
            PROJECT / "PROTOCOL_G3D_RESUME1.md"
        ),
        "driver_log": "raw/g3d_campaign_driver.log" if args.phase == "screen" else None,
        "driver_log_sha256": (
            sha256_file(PROJECT / "raw/g3d_campaign_driver.log")
            if args.phase == "screen"
            else None
        ),
        "slot_manifests": [str(path.relative_to(PROJECT)) for path in slot_manifest_paths],
        "slot_manifest_sha256": {
            str(path.relative_to(PROJECT)): sha256_file(path)
            for path in slot_manifest_paths
        },
        "quiescence_seconds_before_each_launched_process": 30.0,
        "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "max_attempts_per_slot": MAX_CONTAMINATION_ATTEMPTS,
    }
    atomic_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
