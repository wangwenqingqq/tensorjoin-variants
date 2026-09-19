#!/usr/bin/env python3
"""Rebuild the G3B retry manifest from immutable completed-run artifacts."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


EXPERIMENT_ID = "tensorjoin_20260903_gpu_analytic_certificate_g3b"
ORIGINAL_MANIFEST = PROJECT / "results/g3b_safety_manifest.json"
RESULT = PROJECT / "results/g3b_safety_stress1000.json"
RAW = PROJECT / "raw/g3b_safety_stress1000_retry1.log"
PREFLIGHT = PROJECT / "raw/g3b_safety_stress1000_retry1_preflight.log"
OCCUPANCY = PROJECT / "raw/g3b_safety_stress1000_retry1_occupancy.jsonl"
CACHE = PROJECT / "artifacts/g3b_triton_cache_stress1000_retry1"
MANIFEST = PROJECT / "results/g3b_stress_retry1_manifest.json"
DIAGNOSIS = PROJECT / "results/g3b_stress_retry1_manifest_write_failure_diagnosis.json"
CANDIDATE = PROJECT / "src/run_g3b_gpu_analytic_certificate.py"
RUNNER = PROJECT / "src/run_g3b_safety.py"
FAILED_ORCHESTRATOR = PROJECT / "src/run_g3b_stress_retry1.py"


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected live host: {platform.node()}")
    if MANIFEST.exists() or DIAGNOSIS.exists():
        raise FileExistsError(MANIFEST if MANIFEST.exists() else DIAGNOSIS)
    for path in (ORIGINAL_MANIFEST, RESULT, RAW, PREFLIGHT, OCCUPANCY, CACHE):
        if not path.exists():
            raise FileNotFoundError(path)
    original = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    raw_text = RAW.read_text(encoding="utf-8", errors="replace")
    rows = [json.loads(line) for line in OCCUPANCY.read_text(encoding="utf-8").splitlines()]
    run_rows = [row for row in rows if row.get("phase") == "run"]
    post_rows = [row for row in rows if row.get("phase") == "postflight"]
    foreign_rows = [entry for row in run_rows for entry in row.get("foreign_rows", [])]
    target_pids = sorted({pid for row in run_rows for pid in row.get("target_pids", [])})
    max_memory = max(
        [int(str(proc["used_memory_mib"]).split()[0]) for row in run_rows for proc in row.get("compute_rows", [])]
        or [0]
    )
    max_utilization = max([int(row["gpu"]["utilization.gpu"]) for row in run_rows] or [0])
    post_compute = post_rows[-1].get("compute_rows", []) if post_rows else None
    complete_marker = "RUN_COMPLETE " in raw_text
    progress_1000 = '"completed_iterations": 1000' in raw_text
    invariant_pass = bool(
        result.get("safety_pass")
        and result.get("completed_iterations") == 1000
        and result.get("certificate_kernel_launches") == 1000
        and result.get("count_mismatches") == 0
        and result.get("hash_mismatches") == 0
        and result.get("overflow_events") == 0
        and result.get("torch_memory_allocated_after_cleanup_bytes") == 0
        and result.get("torch_memory_reserved_after_cleanup_bytes") == 0
    )
    admitted = bool(
        complete_marker
        and progress_1000
        and invariant_pass
        and not foreign_rows
        and post_compute == []
    )
    diagnosis = {
        "experiment_id": EXPERIMENT_ID,
        "status": "outer_manifest_write_failed_after_completed_stress_run",
        "failed_orchestrator": str(FAILED_ORCHESTRATOR.relative_to(PROJECT)),
        "failed_orchestrator_sha256": sha256_file(FAILED_ORCHESTRATOR),
        "exception_type": "AttributeError",
        "exception": "sha256_file(__file__) received str; sha256_file requires pathlib.Path",
        "candidate_or_runner_failure": False,
        "completed_result_preserved": str(RESULT.relative_to(PROJECT)),
        "completed_result_sha256": sha256_file(RESULT),
        "repair": "read-only finalizer reconstructs manifest; no GPU rerun and no evidence overwrite",
    }
    atomic_json(DIAGNOSIS, diagnosis)
    record = {
        "label": "stress1000_retry1",
        "command": [
            "@TENSORJOIN_ROOT@/isaacsim6/env/bin/python",
            str(RUNNER),
            "--mode",
            "stress1000",
        ],
        "returncode": 0,
        "foreign_rows": foreign_rows,
        "target_gpu_pids": target_pids,
        "external_monitor_max_process_memory_mib": max_memory,
        "external_monitor_max_gpu_utilization_percent": max_utilization,
        "raw_log": str(RAW.relative_to(PROJECT)),
        "raw_log_sha256": sha256_file(RAW),
        "preflight_log": str(PREFLIGHT.relative_to(PROJECT)),
        "preflight_log_sha256": sha256_file(PREFLIGHT),
        "occupancy_log": str(OCCUPANCY.relative_to(PROJECT)),
        "occupancy_log_sha256": sha256_file(OCCUPANCY),
        "postflight_compute_rows": post_compute,
        "cache_path": str(CACHE.relative_to(PROJECT)),
        "result": str(RESULT.relative_to(PROJECT)),
        "result_sha256": sha256_file(RESULT),
        "safety_pass": invariant_pass,
        "admitted": admitted,
        "manifest_reconstructed_after_outer_bookkeeping_failure": True,
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "retry_reason": "attempt0 stopped on foreign GPU0 process after 600 clean iterations",
        "original_manifest": str(ORIGINAL_MANIFEST.relative_to(PROJECT)),
        "original_manifest_sha256": sha256_file(ORIGINAL_MANIFEST),
        "manifest_write_failure_diagnosis": str(DIAGNOSIS.relative_to(PROJECT)),
        "manifest_write_failure_diagnosis_sha256": sha256_file(DIAGNOSIS),
        "candidate_source_sha256": sha256_file(CANDIDATE),
        "safety_runner_sha256": sha256_file(RUNNER),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "record": record,
        "retry_pass": admitted,
        "finalized_wall_time": time.time(),
        "performance_claim_allowed": False,
    }
    atomic_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
