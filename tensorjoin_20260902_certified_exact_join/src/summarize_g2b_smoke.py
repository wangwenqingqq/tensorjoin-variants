#!/usr/bin/env python3
"""Adjudicate the G2B full-scale correctness/resource smoke."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    result_paths = {
        method: PROJECT / f"results/g2b_{method}_smoke.json"
        for method in ("gds", "tensorjoin", "mistic")
    }
    results = {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in result_paths.items()
    }
    audit_path = PROJECT / "results/g2b_accepted_cpu_fp64_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    counts = {result["canonical_pair_count"] for result in results.values()}
    hashes = {result["canonical_raw_u64_sha256"] for result in results.values()}
    pair_paths = {
        method: PROJECT / f"artifacts/g2b/{method}_pairs_u64_le.bin"
        for method in results
    }
    pair_hashes = {method: sha256(path) for method, path in pair_paths.items()}
    common_pair_hash = next(iter(hashes)) if len(hashes) == 1 else None
    structural_fields_pass = (
        all(result["invalid_neighbor_ids"] == 0 for result in [results["gds"]])
        and all(result["invalid_pairs"] == 0 for result in [results["mistic"]])
        and results["tensorjoin"]["invalid_or_lower_triangle_upper_ids"] == 0
        and all(result["duplicate_pairs"] == 0 for result in results.values())
        and all(result["self_pairs"] == 60_000 for result in results.values())
        and all(result["symmetry_missing_pairs"] == 0 for result in results.values())
        and results["tensorjoin"]["overflow_events"] == 0
    )
    exact_gate = (
        len(counts) == 1
        and len(hashes) == 1
        and len(set(pair_hashes.values())) == 1
        and structural_fields_pass
        and bool(audit["accepted_set_passes_sequential_squared_contract"])
        and bool(results["mistic"]["three_way_full_hash_exact_match"])
    )
    evidence = list(result_paths.values()) + list(pair_paths.values()) + [
        audit_path,
        PROJECT / "raw/g2b_accepted_cpu_fp64_audit.log",
        PROJECT / "src/run_g2b_gds_smoke.py",
        PROJECT / "src/run_g2b_tensorjoin_smoke.py",
        PROJECT / "src/run_g2b_mistic_smoke.py",
        PROJECT / "src/audit_g2b_accepted_fp64.py",
        PROJECT / "DESIGN_G2B.md",
        PROJECT / "PROTOCOL_G2.md",
    ]
    for method in results:
        evidence.extend(
            [
                PROJECT / f"raw/g2b_{method}_smoke.log",
                PROJECT / f"raw/g2b_{method}_smoke_preflight.log",
                PROJECT / f"raw/g2b_{method}_smoke_occupancy.log",
            ]
        )
    tensorjoin = results["tensorjoin"]
    summary = {
        "experiment_id": "tensorjoin_20260903_cifar60000_g2b_exact_smoke",
        "decision": "exact_smoke_accepted_public_timing_adapter_required"
        if exact_gate
        else "exact_smoke_rejected",
        "exact_correctness_resource_gate_pass": exact_gate,
        "formal_timing_result": "not_run",
        "performance_claim_allowed": False,
        "shape": [60_000, 512],
        "epsilon": tensorjoin["epsilon"],
        "effective_epsilon_d2": tensorjoin["threshold_d2"],
        "canonical_pair_count": next(iter(counts)) if len(counts) == 1 else None,
        "canonical_raw_u64_sha256": common_pair_hash,
        "methods": {
            method: {
                "pair_count": result["canonical_pair_count"],
                "pair_hash": result["canonical_raw_u64_sha256"],
                "zero_duplicates": result["duplicate_pairs"] == 0,
                "complete_self": result["self_pairs"] == 60_000,
                "complete_symmetry": result["symmetry_missing_pairs"] == 0,
                "diagnostic_only_no_performance_claim": True,
            }
            for method, result in results.items()
        },
        "tensorjoin_work": {
            "upper_comparisons": tensorjoin["analytical_upper_comparisons"],
            "scheduled_tiles": tensorjoin["scheduled_tiles"],
            "batch_count": tensorjoin["batch_count"],
            "direct_accept_upper_pairs": tensorjoin["direct_accept_upper_pairs"],
            "ambiguous_upper_pairs": tensorjoin["ambiguous_upper_pairs"],
            "ambiguity_fraction_of_upper": tensorjoin["ambiguous_upper_pairs"]
            / tensorjoin["analytical_upper_comparisons"],
            "fp32_accept_upper_pairs": tensorjoin["fp32_accept_upper_pairs"],
            "fp64_refine_upper_pairs": tensorjoin["fp64_refine_upper_pairs"],
            "fp64_fraction_of_ambiguity": tensorjoin["fp64_refine_upper_pairs"]
            / tensorjoin["ambiguous_upper_pairs"],
            "accepted_upper_pairs": tensorjoin["accepted_upper_pair_count"],
            "overflow_events": tensorjoin["overflow_events"],
            "torch_peak_memory_allocated_bytes": tensorjoin[
                "torch_peak_memory_allocated_bytes"
            ],
            "torch_peak_memory_reserved_bytes": tensorjoin[
                "torch_peak_memory_reserved_bytes"
            ],
        },
        "cpu_accepted_audit": {
            "audited_upper_pairs": audit["unique_upper_pair_count"],
            "outside_contract_count": audit[
                "vectorized_float64_outside_accepted_count"
            ],
            "closest_inside_d2_gap": audit[
                "closest_accepted_by_vectorized_float64"
            ][0]["distance_d2_minus_threshold"],
            "sequential_squared_contract_pass": audit[
                "accepted_set_passes_sequential_squared_contract"
            ],
        },
        "historical_count_discrepancy": {
            "pinned_fasted_fp64_gds_record": 3_926_074,
            "current_frozen_source_three_way_count": next(iter(counts))
            if len(counts) == 1
            else None,
            "difference": (next(iter(counts)) - 3_926_074)
            if len(counts) == 1
            else None,
            "status": "unresolved input/provenance contract mismatch; historical count is diagnostic only",
            "adjudication": "PROTOCOL_G2 forbids using FaSTED's published count as the oracle; current three-way canonical IDs govern this frozen source",
        },
        "runner_exit_note": "GDS and TensorJoin smoke runners returned nonzero because they incorrectly promoted the diagnostic historical count into a local pass condition; their complete outputs are retained and adjudicated here without rerunning or rewriting them.",
        "source_hashes": {
            str(path.relative_to(PROJECT)): sha256(path) for path in evidence
        },
    }
    output = PROJECT / "results/g2b_smoke_summary.json"
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if exact_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
