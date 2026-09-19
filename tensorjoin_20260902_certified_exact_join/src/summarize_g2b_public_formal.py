#!/usr/bin/env python3
"""Apply the frozen G2B formal paired estimators and acceptance gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from g2b_public_common import PROJECT, atomic_json, sha256_file


MANIFEST = PROJECT / "results/g2b_public_formal_manifest.json"
SAFETY = PROJECT / "results/g2b_tensorjoin_safety_manifest.json"
SUMMARY = PROJECT / "results/g2b_public_formal_summary.json"
SEED = 20260903
BOOTSTRAP_REPLICATES = 100_000
METHODS = ("gds", "mistic", "fasted", "tensorjoin")
EXACT_METHODS = ("gds", "mistic", "tensorjoin")


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10, method="linear")),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "arithmetic_mean": float(np.mean(array)),
        "geometric_mean": float(np.exp(np.mean(np.log(array)))),
    }


def bootstrap_speedup(speedups: list[float]) -> dict[str, object]:
    logs = np.log(np.asarray(speedups, dtype=np.float64))
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = rng.integers(0, logs.size, size=(BOOTSTRAP_REPLICATES, logs.size))
    replicates = np.exp(np.mean(logs[indices], axis=1))
    return {
        "seed": SEED,
        "generator": "NumPy PCG64",
        "replicates": BOOTSTRAP_REPLICATES,
        "paired_geometric_mean": float(np.exp(np.mean(logs))),
        "ci95_lower": float(np.quantile(replicates, 0.025, method="linear")),
        "ci95_upper": float(np.quantile(replicates, 0.975, method="linear")),
    }


def main() -> int:
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY.read_text(encoding="utf-8"))
    admitted = [record for record in manifest["records"] if record.get("admitted")]
    clean_count = len(admitted)
    by_round: dict[int, dict[str, dict[str, object]]] = {}
    for record in admitted:
        by_round.setdefault(int(record["round"]), {})[str(record["method"])] = record
    complete_rounds = len(by_round) == 8 and all(
        set(methods) == set(METHODS) for methods in by_round.values()
    )
    samples = {
        method: [
            float(by_round[round_index][method]["public_seconds"])
            for round_index in range(8)
        ]
        for method in METHODS
    } if complete_rounds else {}
    distributions = {method: distribution(values) for method, values in samples.items()}
    exact_all = complete_rounds and all(
        by_round[round_index][method].get("contract_pass")
        for round_index in range(8)
        for method in EXACT_METHODS
    )
    faster_keeper = (
        min(("gds", "mistic"), key=lambda method: distributions[method]["median"])
        if complete_rounds
        else None
    )
    paired: dict[str, object] = {}
    for keeper in ("gds", "mistic"):
        if not complete_rounds:
            continue
        speedups = [
            samples[keeper][round_index] / samples["tensorjoin"][round_index]
            for round_index in range(8)
        ]
        paired[keeper] = {
            "round_speedups": speedups,
            "tensorjoin_wins": int(sum(value > 1.0 for value in speedups)),
            "tensorjoin_losses": int(sum(value <= 1.0 for value in speedups)),
            **bootstrap_speedup(speedups),
        }
    exact_wins_gate = bool(
        complete_rounds
        and all(int(paired[keeper]["tensorjoin_wins"]) >= 7 for keeper in ("gds", "mistic"))
    )
    confidence_gate = bool(
        faster_keeper is not None
        and float(paired[faster_keeper]["ci95_lower"]) >= 1.50
    )
    fasted_records = [by_round[index]["fasted"] for index in range(8)] if complete_rounds else []
    fasted_quality = {
        "pair_counts": [int(record["quality"]["output_pairs"]) for record in fasted_records],
        "pair_hashes": [record["quality"]["canonical_raw_u64_sha256"] for record in fasted_records],
        "exact_intersection_pairs": [int(record["quality"]["exact_intersection_pairs"]) for record in fasted_records],
        "exact_only_pairs": [int(record["quality"]["exact_only_pairs"]) for record in fasted_records],
        "approximate_only_pairs": [int(record["quality"]["approximate_only_pairs"]) for record in fasted_records],
        "precision": [float(record["quality"]["precision"]) for record in fasted_records],
        "recall": [float(record["quality"]["recall"]) for record in fasted_records],
        "f1": [float(record["quality"]["f1"]) for record in fasted_records],
    }
    safety_gate = bool(safety.get("safety_gate_pass"))
    accepted = bool(
        clean_count == 32
        and complete_rounds
        and exact_all
        and safety_gate
        and exact_wins_gate
        and confidence_gate
    )
    summary = {
        "experiment_id": "tensorjoin_20260903_g2b_public_formal",
        "measurement_status": "formal_public_denominator",
        "scope": "CIFAR-10-GIST 60000x512, epsilon 0.62890625, gpu-host-8 physical GPU0",
        "manifest_sha256": sha256_file(MANIFEST),
        "safety_manifest_sha256": sha256_file(SAFETY),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G2_FORMAL.md"),
        "clean_admitted_processes": clean_count,
        "complete_eight_four_method_rounds": complete_rounds,
        "all_exact_outputs_pass": exact_all,
        "safety_gate_pass": safety_gate,
        "samples_seconds_by_round": samples,
        "marginal_distributions_seconds": distributions,
        "faster_exact_keeper_by_marginal_median": faster_keeper,
        "paired_tensorjoin_speedup_by_exact_keeper": paired,
        "seven_of_eight_win_gate_both_exact_keepers": exact_wins_gate,
        "bootstrap_lower_bound_gate_1_50_vs_faster_keeper": confidence_gate,
        "fasted_context_only": fasted_quality,
        "formal_acceptance_pass": accepted,
        "performance_claim_allowed_within_frozen_scope": accepted,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

