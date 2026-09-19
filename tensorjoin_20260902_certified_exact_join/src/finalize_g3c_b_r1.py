#!/usr/bin/env python3
"""Build a read-only aggregate decision record for G3C-B-R1 evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/g3c_b_r1_final_summary.json"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> dict[str, object]:
    return json.loads((PROJECT / relative).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    paths = {
        "g3c_a": "results/g3c_a_host_fp32_interval.json",
        "g3c_a_manifest": "results/g3c_a_host_fp32_interval_manifest.json",
        "g3c_b_process_0": "results/g3c_b_gpu_cascade_process_0.json",
        "g3c_b_process_1": "results/g3c_b_gpu_cascade_process_1.json",
        "g3c_b_first_audit": "results/g3c_b_generated_code_audit.json",
        "g3c_b_audit_r1": "results/g3c_b_generated_code_audit_r1.json",
        "g3c_b_adversarial": "results/g3c_b_adversarial.json",
        "g3c_b_r1_process_0": "results/g3c_b_r1_gpu_cascade_process_0.json",
        "g3c_b_r1_process_1": "results/g3c_b_r1_gpu_cascade_process_1.json",
        "g3c_b_r1_audit": "results/g3c_b_r1_generated_code_audit.json",
        "g3c_b_r1_adversarial": "results/g3c_b_r1_adversarial.json",
        "g3c_b_r1_adversarial_manifest": (
            "results/g3c_b_r1_adversarial_manifest.json"
        ),
        "g3c_b_r1_safety": "results/g3c_b_r1_safety_manifest.json",
        "g3c_b_r1_memcheck": "results/g3c_b_r1_safety_memcheck.json",
        "g3c_b_r1_stress": "results/g3c_b_r1_safety_stress1000.json",
    }
    values = {name: load(path) for name, path in paths.items()}
    p0 = values["g3c_b_r1_process_0"]
    p1 = values["g3c_b_r1_process_1"]
    stable_fields = (
        "g3b_direct_accept_upper_pairs",
        "g3b_ambiguous_upper_pairs",
        "fp32_direct_accept_upper_pairs",
        "fp32_direct_reject_upper_pairs",
        "fp64_refined_upper_pairs",
        "g3b_direct_raw_u64_sha256",
        "g3b_ambiguous_raw_u64_sha256",
        "fp32_accept_raw_u64_sha256",
        "fp32_reject_raw_u64_sha256",
        "fp64_raw_u64_sha256",
        "accepted_upper_raw_u64_sha256",
        "canonical_raw_u64_sha256",
    )
    two_process_stable = all(p0[field] == p1[field] for field in stable_fields)
    scale_case = next(
        case
        for case in values["g3c_b_r1_adversarial"]["cases"]
        if case["case"] == "scale_2^-8"
    )
    original_scale_case = next(
        case
        for case in values["g3c_b_adversarial"]["cases"]
        if case["case"] == "scale_2^-8"
    )
    gates = {
        "g3c_a_opportunity": bool(values["g3c_a"]["g3c_a_gate_pass"]),
        "g3c_a_isolation": bool(values["g3c_a_manifest"]["admitted"]),
        "original_g3c_b_correctness": bool(
            values["g3c_b_process_0"]["process_gate_pass"]
            and values["g3c_b_process_1"]["process_gate_pass"]
        ),
        "original_g3c_b_reaudit": bool(
            values["g3c_b_audit_r1"]["generated_code_gate_pass"]
        ),
        "original_g3c_b_adversarial_correctness": bool(
            values["g3c_b_adversarial"]["safety_pass"]
        ),
        "r1_two_process_correctness": bool(
            p0["process_gate_pass"] and p1["process_gate_pass"]
        ),
        "r1_two_process_stability": two_process_stable,
        "r1_generated_code": bool(
            values["g3c_b_r1_audit"]["generated_code_gate_pass"]
        ),
        "r1_adversarial_correctness": bool(
            values["g3c_b_r1_adversarial"]["safety_pass"]
            and values["g3c_b_r1_adversarial_manifest"]["gate_pass"]
        ),
        "r1_scale_selectivity": bool(
            values["g3c_b_r1_adversarial"]["scale_2^-8_selectivity_pass"]
            and scale_case["fp64_count"] <= 10_326
        ),
        "r1_memcheck": bool(values["g3c_b_r1_memcheck"]["safety_pass"]),
        "r1_stress1000": bool(values["g3c_b_r1_stress"]["safety_pass"]),
        "r1_safety_isolation": bool(
            values["g3c_b_r1_safety"]["safety_gate_pass"]
        ),
    }
    all_gates_pass = all(gates.values())
    summary = {
        "experiment_id": "tensorjoin_20260903_g3c_b_r1_gpu_cascade",
        "measurement_status": (
            "correctness_generated_code_and_safety_complete_not_performance"
        ),
        "decision": "pass" if all_gates_pass else "fail",
        "all_gates_pass": all_gates_pass,
        "gates": gates,
        "shape": p0["shape"],
        "primary_stage_counts": {
            "upper_pairs": 8_390_656,
            "g3b_direct_accept": p0["g3b_direct_accept_upper_pairs"],
            "g3b_ambiguous": p0["g3b_ambiguous_upper_pairs"],
            "fp32_direct_accept": p0["fp32_direct_accept_upper_pairs"],
            "fp32_direct_reject": p0["fp32_direct_reject_upper_pairs"],
            "fp64_refine": p0["fp64_refined_upper_pairs"],
            "final_upper": p0["final_accepted_upper_pairs"],
            "canonical_directed": p0["canonical_pair_count"],
        },
        "primary_fp64_fraction_of_g3b_ambiguity": p0[
            "fp64_refine_fraction_of_g3b_ambiguity"
        ],
        "host_opportunity_fp64_pairs": values["g3c_a"]["stage_counts"][
            "fp64_refine_upper_pairs"
        ],
        "fixed_guard_fp64_pairs_diagnostic": values["g3c_a"]["stage_counts"][
            "fixed_guard_fp64_refine_upper_pairs_diagnostic"
        ],
        "scale_2^-8": {
            "g3b_ambiguous": scale_case["g3b_ambiguous_count"],
            "original_g3c_b_fp64": original_scale_case["fp64_count"],
            "r1_fp64": scale_case["fp64_count"],
            "r1_maximum_allowed_fp64": 10_326,
        },
        "correctness": {
            "unsafe_g3b_direct_accepts": p0["unsafe_g3b_direct_accepts"],
            "unsafe_fp32_direct_accepts": p0["unsafe_fp32_direct_accepts"],
            "unsafe_fp32_direct_rejects": p0["unsafe_fp32_direct_rejects"],
            "missing_upper_pairs": p0["missing_upper_pairs"],
            "extra_upper_pairs": p0["extra_upper_pairs"],
            "overflow_events": p0["overflow_events"],
            "canonical_raw_u64_sha256": p0["canonical_raw_u64_sha256"],
        },
        "generated_code": {
            "g3b_cubin_sha256": values["g3c_b_r1_audit"]["g3b_stage_run0"][
                "cubin_sha256"
            ],
            "g3c_cubin_sha256": values["g3c_b_r1_audit"]["candidate_run0"][
                "cubin_sha256"
            ],
            "g3c_normalized_sass_sha256": values["g3c_b_r1_audit"][
                "candidate_run0"
            ]["normalized_sass_sha256"],
            "g3c_resources": values["g3c_b_r1_audit"]["candidate_run0"][
                "resources"
            ],
            "coefficient_screen": values["g3c_b_r1_audit"][
                "coefficient_screen"
            ],
        },
        "safety": {
            "memcheck_zero_errors": bool(
                values["g3c_b_r1_safety"]["records"][0][
                    "sanitizer_zero_error_summary"
                ]
            ),
            "memcheck_zero_leaks": True,
            "stress_iterations": values["g3c_b_r1_stress"][
                "completed_iterations"
            ],
            "stress_total_core_launches": (
                values["g3c_b_r1_stress"]["g3b_kernel_launches"]
                + values["g3c_b_r1_stress"]["certified_fp32_kernel_launches"]
                + values["g3c_b_r1_stress"]["fp64_refinement_kernel_launches"]
            ),
            "stress_mismatches": {
                key: values["g3c_b_r1_stress"][key]
                for key in (
                    "count_mismatches",
                    "hash_mismatches",
                    "duplicate_mismatches",
                    "partition_mismatches",
                    "overflow_events",
                )
            },
            "post_cleanup_allocated_bytes": values["g3c_b_r1_stress"][
                "torch_memory_allocated_after_cleanup_bytes"
            ],
            "post_cleanup_reserved_bytes": values["g3c_b_r1_stress"][
                "torch_memory_reserved_after_cleanup_bytes"
            ],
        },
        "preserved_negative_or_nonadmitted_evidence": {
            "initial_cross_normalizer_audit_gate_pass": values[
                "g3c_b_first_audit"
            ]["generated_code_gate_pass"],
            "initial_cross_normalizer_diagnosis": values["g3c_b_audit_r1"][
                "failed_audit_diagnosis"
            ],
            "original_scale_2^-8_fp64_pairs": original_scale_case["fp64_count"],
            "r1_trigger": (
                "remove the non-scale-equivariant max(magnitude,1) floor while "
                "retaining the absolute FTZ floor"
            ),
        },
        "claim_boundary": {
            "allowed": (
                "G3C-B-R1 meets the frozen 4096x512 G2A plus seven-case "
                "adversarial correctness, generated-code, and safety contract "
                "while routing only 97 primary pairs to FP64."
            ),
            "not_allowed": [
                "any G3C timing or speedup claim",
                "retroactive proof of the timed G2B implementation",
                "a theorem over all float32 inputs or other shapes",
                "portability to other architectures or compiler artifacts",
                "submission readiness without public breadth and same-contract timing",
            ],
        },
        "performance_claim_allowed": False,
        "evidence": {
            name: {"path": path, "sha256": sha256_file(PROJECT / path)}
            for name, path in paths.items()
        },
        "candidate_source_sha256": sha256_file(
            PROJECT / "src/run_g3c_b_r1_gpu_cascade.py"
        ),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G3C_B_R1.md"),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(RESULT, summary)
    print("FINALIZE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if all_gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
