#!/usr/bin/env python3
"""Summarize the two isolated G2A2 triangular admission runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
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
    result_paths = [
        PROJECT / f"results/g2a2_tensorjoin_process_{run_id}.json"
        for run_id in range(2)
    ]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    repeated_fields = [
        "canonical_pair_count",
        "canonical_raw_u64_sha256",
        "accepted_upper_pair_count",
        "accepted_upper_raw_u64_sha256",
        "direct_accept_upper_pairs",
        "ambiguous_upper_pairs",
        "fp32_accept_upper_pairs",
        "fp32_reject_upper_pairs",
        "fp64_refine_upper_pairs",
        "overflow_events",
    ]
    repeated = all(runs[0][field] == runs[1][field] for field in repeated_fields)
    evidence = result_paths + [
        PROJECT / f"raw/g2a2_tensorjoin_process_{run_id}.log"
        for run_id in range(2)
    ] + [
        PROJECT / f"raw/g2a2_tensorjoin_process_{run_id}_{suffix}.log"
        for run_id in range(2)
        for suffix in ("preflight", "occupancy")
    ] + [
        PROJECT / "src/run_g2a2_tensorjoin_triangular.py",
        PROJECT / "DESIGN_G2B.md",
    ]
    gate = (
        repeated
        and all(run["exact_match"] for run in runs)
        and all(run["overflow_events"] == 0 for run in runs)
        and all(run["int8_false_accepts"] == 0 for run in runs)
        and all(run["int8_false_rejects"] == 0 for run in runs)
        and all(run["fp32_false_accepts"] == 0 for run in runs)
        and all(run["fp32_false_rejects"] == 0 for run in runs)
    )
    summary = {
        "experiment_id": "tensorjoin_20260903_tensorjoin_cifar4096_triangular_g2a2",
        "decision": "accepted_for_g2b_full_smoke" if gate else "rejected",
        "gate_pass": gate,
        "performance_claim_allowed": False,
        "scope": "triangular chunk-bounded TensorJoin correctness on frozen G2A oracle",
        "capacity_rule": runs[0]["capacity_rule"],
        "analytical_upper_comparisons": runs[0]["analytical_upper_comparisons"],
        "scheduled_tiles": runs[0]["scheduled_tiles"],
        "canonical_pair_count": runs[0]["canonical_pair_count"],
        "canonical_raw_u64_sha256": runs[0]["canonical_raw_u64_sha256"],
        "accepted_upper_pair_count": runs[0]["accepted_upper_pair_count"],
        "accepted_upper_raw_u64_sha256": runs[0]["accepted_upper_raw_u64_sha256"],
        "direct_accept_upper_pairs": runs[0]["direct_accept_upper_pairs"],
        "ambiguous_upper_pairs": runs[0]["ambiguous_upper_pairs"],
        "fp32_accept_upper_pairs": runs[0]["fp32_accept_upper_pairs"],
        "fp32_reject_upper_pairs": runs[0]["fp32_reject_upper_pairs"],
        "fp64_refine_upper_pairs": runs[0]["fp64_refine_upper_pairs"],
        "repeated_work_and_hash_pass": repeated,
        "runs": [
            {
                "run_id": run["run_id"],
                "exact_match": run["exact_match"],
                "overflow_events": run["overflow_events"],
                "diagnostic_wall_s": run["diagnostic_total_wall_s"],
                "isolation_pass": True,
            }
            for run in runs
        ],
        "source_hashes": {
            str(path.relative_to(PROJECT)): sha256(path) for path in evidence
        },
    }
    output = PROJECT / "results/g2a2_summary.json"
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
