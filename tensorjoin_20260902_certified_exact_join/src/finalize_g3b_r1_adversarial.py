#!/usr/bin/env python3
"""Finalize completed G3B-R1 adversarial evidence after a cache-path wrapper bug."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


RESULT = PROJECT / "results/g3b_r1_adversarial.json"
FAILED = PROJECT / "results/g3b_r1_adversarial_manifest.json"
FINAL = PROJECT / "results/g3b_r1_adversarial_final_manifest.json"
DIAGNOSIS = PROJECT / "results/g3b_r1_adversarial_wrapper_path_diagnosis.json"
CACHE = PROJECT / "artifacts/g3b_triton_cache_adversarial_r1"


def main() -> int:
    if platform.node() != "gpu-host-8":
        raise RuntimeError(f"Unexpected host: {platform.node()}")
    if FINAL.exists() or DIAGNOSIS.exists():
        raise FileExistsError(FINAL if FINAL.exists() else DIAGNOSIS)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    failed = json.loads(FAILED.read_text(encoding="utf-8"))
    cubins = list(CACHE.rglob("analytic_certificate_compact_i64.cubin"))
    if len(cubins) != 1:
        raise RuntimeError(f"Expected one candidate cubin, found {cubins}")
    record = failed["record"]
    workload_pass = bool(
        result.get("safety_pass")
        and result.get("case_count") == 7
        and all(case.get("case_pass") for case in result.get("cases", []))
        and record.get("returncode") == 0
        and not record.get("foreign_rows")
        and record.get("postflight_compute_rows") == []
    )
    diagnosis = {
        "experiment_id": failed["experiment_id"],
        "status": "outer_cache_path_lookup_failed_after_completed_workload",
        "candidate_or_numeric_failure": False,
        "failed_manifest": str(FAILED.relative_to(PROJECT)),
        "failed_manifest_sha256": sha256_file(FAILED),
        "result": str(RESULT.relative_to(PROJECT)),
        "result_sha256": sha256_file(RESULT),
        "incorrect_lookup": "artifacts/g3b_r1_triton_cache_adversarial_r1",
        "actual_cache": str(CACHE.relative_to(PROJECT)),
        "cause": "R1 wrapper imported the original run_guarded helper, whose cache prefix is g3b_triton_cache_",
        "repair": "read-only finalizer locates the immutable actual cache; no GPU rerun or evidence overwrite",
    }
    atomic_json(DIAGNOSIS, diagnosis)
    final = {
        "experiment_id": failed["experiment_id"],
        "measurement_status": failed["measurement_status"],
        "workload_pass": workload_pass,
        "case_count": result["case_count"],
        "all_case_pass": all(case["case_pass"] for case in result["cases"]),
        "unsafe_direct_accepts_total": sum(case["unsafe_direct_accepts"] for case in result["cases"]),
        "unsafe_direct_rejects_total": sum(case["unsafe_direct_rejects"] for case in result["cases"]),
        "candidate_cubin": str(cubins[0].relative_to(PROJECT)),
        "candidate_cubin_sha256": sha256_file(cubins[0]),
        "result": str(RESULT.relative_to(PROJECT)),
        "result_sha256": sha256_file(RESULT),
        "failed_wrapper_manifest": str(FAILED.relative_to(PROJECT)),
        "failed_wrapper_manifest_sha256": sha256_file(FAILED),
        "diagnosis": str(DIAGNOSIS.relative_to(PROJECT)),
        "diagnosis_sha256": sha256_file(DIAGNOSIS),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "finalized_wall_time": time.time(),
        "gate_pass": workload_pass,
        "performance_claim_allowed": False,
    }
    atomic_json(FINAL, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if workload_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
