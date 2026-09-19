#!/usr/bin/env python3
"""Run G3B-A under the existing physical-GPU isolation contract."""

from __future__ import annotations

import fcntl
import json
import platform
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g2b_public_screen import LOCK_PATH, PYTHON, gpu_rows
from run_g3b_safety_gates import run_guarded


EXPERIMENT_ID = "tensorjoin_20260903_gpu_analytic_certificate_g3b_adversarial"
RUNNER = PROJECT / "src/run_g3b_adversarial.py"
RESULT = PROJECT / "results/g3b_adversarial.json"
MANIFEST = PROJECT / "results/g3b_adversarial_manifest.json"


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    if RESULT.exists() or MANIFEST.exists():
        raise FileExistsError(RESULT if RESULT.exists() else MANIFEST)
    gpu0 = gpu_rows()[0]
    started = time.time()
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        record = run_guarded(
            "adversarial",
            [PYTHON, str(RUNNER)],
            RESULT,
            str(gpu0["uuid"]),
        )
    cubins = list((PROJECT / "artifacts/g3b_triton_cache_adversarial").rglob(
        "analytic_certificate_compact_i64.cubin"
    ))
    cubin_hash = sha256_file(cubins[0]) if len(cubins) == 1 else None
    expected_cubin = "c80141730aecd650b57be14ecee6422be1477fc704d9a09e0ecc6949a3ad06af"
    record["candidate_cubin"] = str(cubins[0].relative_to(PROJECT)) if len(cubins) == 1 else None
    record["candidate_cubin_sha256"] = cubin_hash
    record["candidate_cubin_matches_frozen"] = cubin_hash == expected_cubin
    record["admitted"] = bool(record["admitted"] and record["candidate_cubin_matches_frozen"])
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "adversarial_correctness_only_no_performance_claim",
        "gpu0": gpu0,
        "record": record,
        "gate_pass": bool(record["admitted"]),
        "runner_sha256": sha256_file(RUNNER),
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "wall_seconds": time.time() - started,
        "performance_claim_allowed": False,
    }
    atomic_json(MANIFEST, manifest)
    print("ADVERSARIAL_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if manifest["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
