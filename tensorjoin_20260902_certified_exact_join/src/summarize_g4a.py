#!/usr/bin/env python3
"""Apply the frozen G4A dynamic-router screen or formal paired estimators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20_260_903
BOOTSTRAP_SAMPLES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_us": float(np.percentile(array, 10)),
        "median_us": float(np.median(array)),
        "p90_us": float(np.percentile(array, 90)),
    }


def bootstrap(speedups: list[float]) -> dict[str, float | int]:
    logs = np.log(np.asarray(speedups, dtype=np.float64))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(logs), size=(BOOTSTRAP_SAMPLES, len(logs)))
    samples = np.exp(np.mean(logs[indices], axis=1))
    return {
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "ci95_lower": float(np.percentile(samples, 2.5)),
        "ci95_upper": float(np.percentile(samples, 97.5)),
    }


def main() -> int:
    args = parse_args()
    expected_processes = 2 if args.phase == "screen" else 8
    audit_path = PROJECT / f"results/g4a_{args.phase}_runtime_audit.json"
    output_path = PROJECT / f"results/g4a_{args.phase}_summary.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("runtime_binary_gate_pass"):
        raise RuntimeError("Timed runtime cubins do not match the accepted artifacts")
    manifests = sorted(
        (PROJECT / "results").glob(f"g4a_{args.phase}_process_*_manifest.json")
    )
    all_records = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    admitted_records = [record for record in all_records if record["admitted"]]
    process_ids = [int(record["process_id"]) for record in admitted_records]
    if len(admitted_records) != expected_processes or sorted(process_ids) != list(
        range(expected_processes)
    ):
        raise RuntimeError("Manifest does not contain exactly one admitted record per slot")

    process_rows: list[dict[str, object]] = []
    raw = {"keeper_two_stage": [], "candidate_three_stage": []}
    position_raw = {
        "first": {"keeper_two_stage": [], "candidate_three_stage": []},
        "second": {"keeper_two_stage": [], "candidate_three_stage": []},
    }
    source_contracts: list[dict[str, str]] = []
    runner_hashes: list[str] = []
    protocol_hashes: list[str] = []
    dynamic_count_contracts: list[bool] = []
    for manifest_record in sorted(admitted_records, key=lambda value: value["process_id"]):
        result_path = PROJECT / str(manifest_record["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result["correctness_pass"] or not result["process_measurement_pass"]:
            raise RuntimeError(f"Unqualified admitted result: {result_path}")
        order = str(result["order"])
        first_name = "keeper_two_stage" if order[0] == "K" else "candidate_three_stage"
        for name in raw:
            values = [float(value) for value in result["metrics_us"][name]]
            raw[name].extend(values)
            position = "first" if name == first_name else "second"
            position_raw[position][name].extend(values)
        median_speedup = float(
            result["paired_process_median_speedup_keeper_over_candidate"]
        )
        sustained_speedup = float(
            result["sustained_speedup_keeper_over_candidate"]
        )
        expected_dynamic = {
            "keeper_two_stage": {
                "g3b_ambiguous": 102079,
                "fp64_refined": 102079,
                "final_upper": 133120,
            },
            "candidate_three_stage": {
                "g3b_ambiguous": 102079,
                "fp64_refined": 97,
                "final_upper": 133120,
            },
        }
        observed_samples = {name: [] for name in expected_dynamic}
        for sample in result["ordered_samples"]:
            observed_samples[str(sample["variant"])].append(
                sample["dynamic_stage_counts"]
            )
        dynamic_count_pass = bool(
            all(
                values
                and all(value == expected_dynamic[name] for value in values)
                and result["sustained"][name]["unique_dynamic_stage_counts"]
                == [expected_dynamic[name]]
                for name, values in observed_samples.items()
            )
        )
        dynamic_count_contracts.append(dynamic_count_pass)
        process_rows.append(
            {
                "process_id": int(result["process_id"]),
                "attempt": int(result["attempt"]),
                "order": order,
                "keeper_median_us": float(result["medians_us"]["keeper_two_stage"]),
                "candidate_median_us": float(
                    result["medians_us"]["candidate_three_stage"]
                ),
                "median_speedup_keeper_over_candidate": median_speedup,
                "keeper_sustained_mean_us": float(
                    result["sustained"]["keeper_two_stage"]["mean_us_per_pipeline"]
                ),
                "candidate_sustained_mean_us": float(
                    result["sustained"]["candidate_three_stage"][
                        "mean_us_per_pipeline"
                    ]
                ),
                "sustained_speedup_keeper_over_candidate": sustained_speedup,
                "dynamic_count_contract_pass": dynamic_count_pass,
                "result": manifest_record["result"],
                "result_sha256": manifest_record["result_sha256"],
            }
        )
        source_contracts.append(result["source_contract"])
        runner_hashes.append(str(result["runner_sha256"]))
        protocol_hashes.append(str(result["protocol_sha256"]))

    median_speedups = [
        float(row["median_speedup_keeper_over_candidate"]) for row in process_rows
    ]
    sustained_speedups = [
        float(row["sustained_speedup_keeper_over_candidate"]) for row in process_rows
    ]
    marginal = {name: percentiles(values) for name, values in raw.items()}
    order_position = {
        position: {
            name: percentiles(values) if values else None
            for name, values in methods.items()
        }
        for position, methods in position_raw.items()
    }
    common = {
        "all_expected_processes_admitted": True,
        "runtime_binary_gate_pass": True,
        "dynamic_count_contract_pass": all(dynamic_count_contracts),
        "all_source_contracts_equal": all(
            value == source_contracts[0] for value in source_contracts
        ),
        "single_runner_sha256": sorted(set(runner_hashes)),
        "single_protocol_sha256": sorted(set(protocol_hashes)),
        "process_median_wins": sum(value > 1.0 for value in median_speedups),
        "sustained_wins": sum(value > 1.0 for value in sustained_speedups),
        "paired_geometric_mean_median_speedup": geometric_mean(median_speedups),
        "sustained_geometric_mean_speedup": geometric_mean(sustained_speedups),
    }
    if args.phase == "screen":
        gate = bool(
            common["all_source_contracts_equal"]
            and common["dynamic_count_contract_pass"]
            and len(common["single_runner_sha256"]) == 1
            and len(common["single_protocol_sha256"]) == 1
            and all(value >= 1.20 for value in median_speedups)
            and all(value >= 1.20 for value in sustained_speedups)
        )
        decision = "admit_formal" if gate else "stop_preserve_negative"
        gate_record: dict[str, object] = {
            "screen_gate_pass": gate,
            "screen_thresholds": {
                "each_process_median_speedup_minimum": 1.20,
                "each_process_sustained_speedup_minimum": 1.20,
            },
        }
    else:
        confidence = bootstrap(median_speedups)
        gate = bool(
            common["all_source_contracts_equal"]
            and common["dynamic_count_contract_pass"]
            and len(common["single_runner_sha256"]) == 1
            and len(common["single_protocol_sha256"]) == 1
            and common["process_median_wins"] == 8
            and float(confidence["ci95_lower"]) >= 1.25
            and common["sustained_wins"] == 8
            and float(common["sustained_geometric_mean_speedup"]) >= 1.20
        )
        decision = "promote_shape_local_attribution" if gate else "stop_preserve_negative"
        gate_record = {
            "formal_gate_pass": gate,
            "formal_thresholds": {
                "process_median_wins_required": 8,
                "bootstrap_ci95_lower_minimum": 1.25,
                "sustained_wins_required": 8,
                "sustained_geometric_mean_minimum": 1.20,
            },
            "paired_process_bootstrap": confidence,
        }

    result = {
        "experiment_id": "tensorjoin_20260903_g4a_dynamic_count_router",
        "phase": args.phase,
        "decision": decision,
        "measurement_scope": "host-dispatched dynamic-count resident-GPU operator",
        "scope_is_not": "ingest-inclusive or host-output-materializing end-to-end latency",
        "processes": process_rows,
        "marginal_raw_sample_percentiles_us": marginal,
        "order_position_raw_sample_percentiles_us": order_position,
        **common,
        **gate_record,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
