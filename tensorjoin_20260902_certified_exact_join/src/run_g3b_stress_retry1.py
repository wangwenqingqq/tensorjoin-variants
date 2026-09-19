#!/usr/bin/env python3
"""Retry only G3B stress1000 after the preserved contamination stop."""

from __future__ import annotations

import fcntl
import json
import platform
import time

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g2b_public_screen import LOCK_PATH, PYTHON, gpu_rows
from run_g3b_safety_gates import CANDIDATE, RUNNER, run_guarded


EXPERIMENT_ID = "tensorjoin_20260903_gpu_analytic_certificate_g3b"
ORIGINAL_MANIFEST = PROJECT / "results/g3b_safety_manifest.json"
MANIFEST = PROJECT / "results/g3b_stress_retry1_manifest.json"
RESULT = PROJECT / "results/g3b_safety_stress1000.json"


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    if MANIFEST.exists() or RESULT.exists():
        raise FileExistsError(MANIFEST if MANIFEST.exists() else RESULT)
    original = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    if original.get("safety_gate_pass") is not False:
        raise RuntimeError("Retry is allowed only after the preserved failed first attempt")
    failed = original.get("records", [])[-1]
    if failed.get("label") != "stress1000" or not failed.get("foreign_rows"):
        raise RuntimeError("Original failure is not the expected contamination stop")

    gpu0 = gpu_rows()[0]
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        record = run_guarded(
            "stress1000_retry1",
            [PYTHON, str(RUNNER), "--mode", "stress1000"],
            RESULT,
            str(gpu0["uuid"]),
        )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "retry_reason": "attempt0 stopped on foreign GPU0 process; candidate had 600 clean iterations",
        "original_manifest": str(ORIGINAL_MANIFEST.relative_to(PROJECT)),
        "original_manifest_sha256": sha256_file(ORIGINAL_MANIFEST),
        "candidate_source_sha256": sha256_file(CANDIDATE),
        "safety_runner_sha256": sha256_file(RUNNER),
        "retry_orchestrator_sha256": sha256_file(__file__),
        "gpu0": gpu0,
        "record": record,
        "retry_pass": bool(record["admitted"]),
        "wall_seconds": time.time() - started,
    }
    atomic_json(MANIFEST, manifest)
    print("STRESS_RETRY_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if manifest["retry_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
