#!/usr/bin/env python3
"""Finalize the frozen G5 same-specialization safety gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results/g5_p3_safety_manifest.json"
EXPECTED = {
    "results/g5_safety_memcheck_a0.json": "5624f86251248186a1d21c3737d830f9ef490a142bb83411c5a2a47e3bbc9c19",
    "results/g5_safety_stress1000_a0.json": "ae15cef74f5b571e70cc89a66d05b4d08759d788dd3157e23a57aa6aaa9d56da",
    "results/g5_guard_p3_memcheck_gpu2_a0.json": "a3ae1f767df1f79dcb7ed8bb878a3c1c7f996260f6a90cb22d9e8130ddd6be64",
    "results/g5_guard_p3_stress1000_gpu2_a0.json": "ae3e48aef6f3904a2442e0df3dcb274629939a2b2c0e77773d50d38f8d66899d",
    "raw/g5_p3_memcheck_gpu2_a0.log": "2f4f7c614285a794d1b7d56ab2b383714253485680d05ba76215637e6fa1d3b1",
    "raw/g5_p3_stress1000_gpu2_a0.log": "dfb6f3439af790755c219fce3ecfb3b2468f68fb0f96e6e7cb514160f5a82b50",
}
EXPECTED_SIGNATURE = [74, 41, 16, 17, 8, 0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    observed = {name: sha256_file(PROJECT / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError(
            "G5 P3 evidence hash mismatch: "
            + json.dumps({"expected": EXPECTED, "observed": observed}, sort_keys=True)
        )
    memcheck = json.loads(
        (PROJECT / "results/g5_safety_memcheck_a0.json").read_text(encoding="utf-8")
    )
    stress = json.loads(
        (PROJECT / "results/g5_safety_stress1000_a0.json").read_text(encoding="utf-8")
    )
    mem_guard = json.loads(
        (PROJECT / "results/g5_guard_p3_memcheck_gpu2_a0.json").read_text(
            encoding="utf-8"
        )
    )
    stress_guard = json.loads(
        (PROJECT / "results/g5_guard_p3_stress1000_gpu2_a0.json").read_text(
            encoding="utf-8"
        )
    )
    mem_text = (PROJECT / "raw/g5_p3_memcheck_gpu2_a0.log").read_text(
        encoding="utf-8", errors="replace"
    )
    sanitizer_zero = "ERROR SUMMARY: 0 errors" in mem_text
    leak_zero = "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in mem_text
    signature_stable = bool(
        memcheck[
            "stage_counts_direct_ambiguous_fp32_accept_fp32_reject_fp64_equality"
        ]
        == EXPECTED_SIGNATURE
        == stress[
            "stage_counts_direct_ambiguous_fp32_accept_fp32_reject_fp64_equality"
        ]
    )
    binary_hashes = {
        name: {
            "memcheck": memcheck["selected_cache_artifacts_after"][name][
                "cubin_sha256"
            ],
            "stress": stress["selected_cache_artifacts_after"][name][
                "cubin_sha256"
            ],
        }
        for name in memcheck["selected_cache_artifacts_after"]
    }
    binary_stable = all(
        hashes["memcheck"] == hashes["stress"]
        for hashes in binary_hashes.values()
    )
    safety_gate = bool(
        memcheck["safety_pass"]
        and stress["safety_pass"]
        and memcheck["completed_iterations"] == 2
        and stress["completed_iterations"] == 1000
        and sanitizer_zero
        and leak_zero
        and signature_stable
        and binary_stable
        and mem_guard["admitted"]
        and stress_guard["admitted"]
        and not mem_guard["foreign_rows"]
        and not stress_guard["foreign_rows"]
        and not mem_guard["postflight_compute_rows"]
        and not stress_guard["postflight_compute_rows"]
    )
    result = {
        "experiment_id": "tensorjoin_20260903_g5_unified_public_cifar60k",
        "measurement_status": "correctness_safety_only_no_performance_claim",
        "evidence_hashes": observed,
        "sanitizer_zero_error_summary": sanitizer_zero,
        "sanitizer_zero_leak_summary": leak_zero,
        "stage_signature": EXPECTED_SIGNATURE,
        "stage_signature_stable": signature_stable,
        "binary_hashes": binary_hashes,
        "binary_stable_across_memcheck_and_stress": binary_stable,
        "completed_pipeline_invocations": 1002,
        "output_mismatches": memcheck["output_mismatches"]
        + stress["output_mismatches"],
        "stage_count_mismatches": memcheck["stage_count_mismatches"]
        + stress["stage_count_mismatches"],
        "overflow_events": memcheck["overflow_events"]
        + stress["overflow_events"],
        "safety_gate_pass": safety_gate,
        "performance_claim_allowed": False,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G5_P3_SAFETY_STRESS.md"),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if safety_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())

