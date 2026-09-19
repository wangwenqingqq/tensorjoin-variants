#!/usr/bin/env python3
"""Run the G3C-B adversarial gate under physical-GPU0 isolation."""

from __future__ import annotations

import fcntl
import json
import platform
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file
from run_g2b_public_screen import LOCK_PATH, PYTHON, gpu_rows
from run_g3b_safety_gates import run_guarded


EXPERIMENT_ID = "tensorjoin_20260903_g3c_b_gpu_cascade_adversarial"
RUNNER = PROJECT / "src/run_g3c_b_adversarial.py"
RESULT = PROJECT / "results/g3c_b_adversarial.json"
MANIFEST = PROJECT / "results/g3c_b_adversarial_manifest.json"
EXPECTED_G3B_CUBIN = (
    "dd32e97936d8e0c6abd3924d8fdeeb17088e7a6e1e4d8ac2f1fc492bbc2114ac"
)
EXPECTED_G3C_CUBIN = (
    "94b52dfcc8af53deaa9d70cdfdfe5ae01be930df0330f8929be958088454f78b"
)


def one_hash(root: Path, name: str) -> tuple[str | None, str | None]:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        return None, None
    return str(matches[0].relative_to(PROJECT)), sha256_file(matches[0])


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    if RESULT.exists() or MANIFEST.exists():
        raise FileExistsError(RESULT if RESULT.exists() else MANIFEST)
    gpu0 = gpu_rows()[0]
    started = time.time()
    label = "g3c_b_adversarial"
    with LOCK_PATH.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        record = run_guarded(
            label,
            [PYTHON, str(RUNNER)],
            RESULT,
            str(gpu0["uuid"]),
        )
    cache = PROJECT / f"artifacts/g3b_triton_cache_{label}"
    g3b_path, g3b_hash = one_hash(cache, "analytic_certificate_compact_i64.cubin")
    g3c_path, g3c_hash = one_hash(cache, "certified_fp32_filter_i64.cubin")
    record.update(
        {
            "g3b_cubin": g3b_path,
            "g3b_cubin_sha256": g3b_hash,
            "g3b_cubin_match": g3b_hash == EXPECTED_G3B_CUBIN,
            "g3c_cubin": g3c_path,
            "g3c_cubin_sha256": g3c_hash,
            "g3c_cubin_match": g3c_hash == EXPECTED_G3C_CUBIN,
        }
    )
    record["admitted"] = bool(
        record["admitted"]
        and record["g3b_cubin_match"]
        and record["g3c_cubin_match"]
    )
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
