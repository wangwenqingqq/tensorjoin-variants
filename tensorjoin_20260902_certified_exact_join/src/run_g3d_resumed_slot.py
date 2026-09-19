#!/usr/bin/env python3
"""Run one frozen G3D slot after a prelaunch foreign-GPU blockage."""

from __future__ import annotations

import argparse
import fcntl
import json
import platform
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g2b_public_screen import LOCK_PATH, gpu_rows
from run_g3d_campaign import ORDERS, RUNNER, SETTINGS, run_one


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
        raise RuntimeError("Process ID is outside the frozen phase schedule")
    if args.phase == "formal":
        screen_summary = json.loads(
            (PROJECT / "results/g3d_screen_summary.json").read_text(encoding="utf-8")
        )
        screen_audit = json.loads(
            (PROJECT / "results/g3d_screen_generated_code_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if not screen_summary.get("screen_gate_pass") or not screen_audit.get(
            "screen_runtime_binary_gate_pass"
        ):
            raise RuntimeError("G3D screen has not admitted formal execution")

    order = ORDERS[args.phase][args.process_id]
    slot_manifest = PROJECT / (
        f"results/g3d_{args.phase}_slot_{args.process_id}_{order.lower()}_"
        f"a{args.attempt}_manifest.json"
    )
    if slot_manifest.exists():
        raise FileExistsError(slot_manifest)
    gpu0 = gpu_rows()[0]
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        record = run_one(
            args.phase,
            args.process_id,
            order,
            args.attempt,
            str(gpu0["uuid"]),
        )
    manifest = {
        "experiment_id": "tensorjoin_20260903_g3d_complete_pipeline_timing",
        "status": "clean_slot_complete" if record["admitted"] else "slot_failed",
        "phase": args.phase,
        "process_id": args.process_id,
        "order": order,
        "attempt": args.attempt,
        "settings": SETTINGS[args.phase],
        "record": record,
        "gpu0": gpu0,
        "runner_sha256": sha256_file(RUNNER),
        "original_orchestrator_sha256": sha256_file(
            PROJECT / "src/run_g3d_campaign.py"
        ),
        "resume_orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "original_protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3D.md"),
        "resume_protocol_sha256": sha256_file(
            PROJECT / "PROTOCOL_G3D_RESUME1.md"
        ),
        "wall_seconds": time.time() - started,
    }
    atomic_json(slot_manifest, manifest)
    print("SLOT_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if record["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
