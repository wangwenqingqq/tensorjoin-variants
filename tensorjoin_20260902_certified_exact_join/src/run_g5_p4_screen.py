#!/usr/bin/env python3
"""Run the frozen two-round G5 external-system cheap screen."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


EXPERIMENT_ID = "tensorjoin_20260903_g5_unified_public_cifar60k"
PYTHON = "@TENSORJOIN_ROOT@/isaacsim6/env/bin/python"
PHYSICAL_GPU = 2
ORDERS = (
    ("gds", "mistic", "tensorjoin"),
    ("tensorjoin", "mistic", "gds"),
)
RUNNERS = {
    "gds": PROJECT / "src/run_g5_gds_public.py",
    "mistic": PROJECT / "src/run_g5_mistic_public.py",
    "tensorjoin": PROJECT / "src/run_g5_tensorjoin_public.py",
}
EXPECTED_RUNNER_HASHES = {
    "gds": "786053df96026bc0daeda146324de0f26464e2cc629f1c220cee8113238f4a96",
    "mistic": "83f4141ec87c74a1f288ce82e706bd40327367a849118278e046bdc8fe75339e",
    "tensorjoin": "a8653469a0e5bae82b64cd3d9a82c476df741783735c1b9d699e2f27b2db13a5",
}
GUARD = PROJECT / "src/run_g5_guarded_process.py"
EXPECTED_GUARD_SHA256 = "8d7de27698313a31777d08b315d7fbafde9fd8a0cca5f50c8fd54eced3009126"
SAFETY = PROJECT / "results/g5_p3_safety_manifest.json"
EXPECTED_SAFETY_SHA256 = "6c6b0dfc52c33f91c6bd2d95331254838de5379c24299c3c88b87fafc81895d0"
MANIFEST = PROJECT / "results/g5_p4_screen_manifest.json"


def write_manifest(status: str, records: list[dict[str, object]], started: float) -> None:
    atomic_json(
        MANIFEST,
        {
            "experiment_id": EXPERIMENT_ID,
            "measurement_status": "diagnostic_public_denominator_cheap_screen",
            "status": status,
            "host": platform.node(),
            "physical_gpu": PHYSICAL_GPU,
            "orders": ORDERS,
            "records": records,
            "runner_hashes": {
                method: sha256_file(path) for method, path in RUNNERS.items()
            },
            "guard_sha256": sha256_file(GUARD),
            "safety_manifest_sha256": sha256_file(SAFETY),
            "wall_seconds": time.time() - started,
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G5_P4_SCREEN.md"),
        },
    )


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Expected live gpu-host-8 host, got {platform.node()}")
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    observed_runners = {
        method: sha256_file(path) for method, path in RUNNERS.items()
    }
    if observed_runners != EXPECTED_RUNNER_HASHES:
        raise RuntimeError("Frozen G5 P4 runner hash mismatch")
    if sha256_file(GUARD) != EXPECTED_GUARD_SHA256:
        raise RuntimeError("Frozen G5 guard hash mismatch")
    if sha256_file(SAFETY) != EXPECTED_SAFETY_SHA256:
        raise RuntimeError("G5 P3 safety manifest hash mismatch")
    safety = json.loads(SAFETY.read_text(encoding="utf-8"))
    if not safety.get("safety_gate_pass"):
        raise RuntimeError("G5 P3 did not admit P4")

    records: list[dict[str, object]] = []
    started = time.time()
    for round_index, order in enumerate(ORDERS):
        for position, method in enumerate(order):
            record_id = f"r{round_index}_p{position}_a0"
            label = f"p4_screen_r{round_index}_p{position}_{method}_a0"
            expected_result = PROJECT / f"results/g5_screen_{method}_{record_id}.json"
            command = [
                PYTHON,
                str(GUARD),
                "--label",
                label,
                "--expected-result",
                str(expected_result.relative_to(PROJECT)),
                "--physical-gpu",
                str(PHYSICAL_GPU),
                "--",
                PYTHON,
                str(RUNNERS[method]),
                "--record-id",
                record_id,
                "--phase",
                "screen",
            ]
            print("SCREEN_SLOT_START " + json.dumps(command), flush=True)
            completed = subprocess.run(command, check=False)
            guard_path = PROJECT / f"results/g5_guard_{label}.json"
            record: dict[str, object] = {
                "round": round_index,
                "position": position,
                "method": method,
                "record_id": record_id,
                "label": label,
                "command": command,
                "returncode": completed.returncode,
                "guard_result": (
                    str(guard_path.relative_to(PROJECT)) if guard_path.is_file() else None
                ),
                "child_result": (
                    str(expected_result.relative_to(PROJECT))
                    if expected_result.is_file()
                    else None
                ),
            }
            if guard_path.is_file():
                guard = json.loads(guard_path.read_text(encoding="utf-8"))
                record.update(
                    {
                        "guard_result_sha256": sha256_file(guard_path),
                        "guard_admitted": bool(guard.get("admitted")),
                        "foreign_rows": guard.get("foreign_rows"),
                        "postflight_compute_rows": guard.get(
                            "postflight_compute_rows"
                        ),
                    }
                )
            if expected_result.is_file():
                child = json.loads(expected_result.read_text(encoding="utf-8"))
                record.update(
                    {
                        "child_result_sha256": sha256_file(expected_result),
                        "public_seconds": child.get("public_seconds"),
                        "exact_contract_pass": bool(
                            child.get("correctness", {}).get("exact_contract_pass")
                        ),
                    }
                )
                if method == "tensorjoin":
                    record["candidate_runner_sha256"] = child.get("runner_sha256")
                    record["candidate_stage_counts"] = {
                        key: child.get(key)
                        for key in (
                            "g3b_direct_accept_upper_pairs",
                            "g3b_ambiguous_upper_pairs",
                            "fp32_direct_accept_upper_pairs",
                            "fp32_direct_reject_upper_pairs",
                            "fp32_bitwise_equal_upper_pairs",
                            "fp64_refined_upper_pairs",
                            "accepted_upper_pairs",
                            "overflow_events",
                        )
                    }
                    record["candidate_cache_artifacts"] = child.get(
                        "selected_cache_artifacts_after_timer"
                    )
                elif method == "gds":
                    record["library_sha256"] = child.get("library_sha256")
                elif method == "mistic":
                    record["binary_sha256"] = child.get("binary_sha256")
            records.append(record)
            print("SCREEN_SLOT_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
            if not (
                completed.returncode == 0
                and record.get("guard_admitted")
                and record.get("exact_contract_pass")
            ):
                write_manifest("stopped_on_failed_or_contaminated_slot", records, started)
                return 2

    write_manifest("two_clean_three_method_rounds_complete", records, started)
    print(f"SCREEN_COMPLETE manifest={MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

