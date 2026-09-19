#!/usr/bin/env python3
"""Apply the predeclared G2B public cheap-screen decision rule."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from g2b_public_common import PROJECT, atomic_json, sha256_file


MANIFEST = PROJECT / "results/g2b_public_screen_manifest.json"
SUMMARY = PROJECT / "results/g2b_public_screen_summary.json"


def main() -> int:
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest["records"]
    expected_methods = {"gds", "mistic", "tensorjoin"}
    by_round: dict[int, dict[str, float]] = {}
    exact_all = len(records) == 6 and all(
        record.get("admitted") and record.get("exact_contract_pass")
        for record in records
    )
    for record in records:
        if not record.get("admitted"):
            continue
        by_round.setdefault(int(record["round"]), {})[str(record["method"])] = float(
            record["public_seconds"]
        )
    complete_rounds = len(by_round) == 2 and all(
        set(methods) == expected_methods for methods in by_round.values()
    )
    method_samples = {
        method: [by_round[round_id][method] for round_id in sorted(by_round)]
        for method in sorted(expected_methods)
    } if complete_rounds else {}
    medians = {
        method: statistics.median(samples)
        for method, samples in method_samples.items()
    }
    faster_keeper = (
        min(("gds", "mistic"), key=lambda method: medians[method])
        if complete_rounds
        else None
    )
    speedups_by_keeper = {
        keeper: [
            by_round[round_id][keeper] / by_round[round_id]["tensorjoin"]
            for round_id in sorted(by_round)
        ]
        for keeper in ("gds", "mistic")
    } if complete_rounds else {}
    faster_speedups = speedups_by_keeper.get(faster_keeper, [])
    paired_geomean = (
        math.exp(sum(math.log(value) for value in faster_speedups) / len(faster_speedups))
        if faster_speedups
        else None
    )
    faster_than_both_each_round = complete_rounds and all(
        by_round[round_id]["tensorjoin"] < by_round[round_id][keeper]
        for round_id in by_round
        for keeper in ("gds", "mistic")
    )
    min_speedup_gate = bool(
        faster_speedups and min(faster_speedups) >= 1.50
    )
    geomean_gate = bool(paired_geomean is not None and paired_geomean >= 1.60)
    screen_pass = bool(
        exact_all
        and complete_rounds
        and faster_than_both_each_round
        and min_speedup_gate
        and geomean_gate
    )
    summary = {
        "experiment_id": "tensorjoin_20260903_g2b_public_screen",
        "measurement_status": "diagnostic_public_denominator_cheap_screen",
        "performance_claim_allowed": False,
        "manifest_sha256": sha256_file(MANIFEST),
        "orders": manifest.get("orders"),
        "exact_all_six_processes": exact_all,
        "complete_direction_balanced_rounds": complete_rounds,
        "samples_seconds": method_samples,
        "marginal_medians_seconds": medians,
        "faster_exact_keeper_by_predeclared_median": faster_keeper,
        "paired_speedups_tensorjoin_by_keeper": speedups_by_keeper,
        "paired_geomean_speedup_vs_faster_keeper": paired_geomean,
        "faster_than_both_each_round": faster_than_both_each_round,
        "minimum_round_speedup_vs_faster_keeper_gate_1_50": min_speedup_gate,
        "paired_geomean_gate_1_60": geomean_gate,
        "screen_pass": screen_pass,
        "next_action": (
            "run_candidate_sanitizer_and_stress_then_formal_eight_round_campaign"
            if screen_pass
            else "stop_current_public_path_and_record_negative_evidence"
        ),
        "screen_rule_source": "DESIGN_G2_TIMING.md",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if screen_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

