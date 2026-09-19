#!/usr/bin/env python3
"""Apply the frozen per-dataset G4C formal timing estimators and gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/g4c_formal_summary.json"
DATASETS = ("sift128", "cifar_gist512", "fashion784")
ORDERS = ("kc", "ck", "kc", "ck", "ck", "kc", "ck", "kc")
BOOTSTRAP_SEED = 20_260_903
BOOTSTRAP_SAMPLES = 20_000


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
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


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_us": float(np.percentile(array, 10)),
        "median_us": float(np.median(array)),
        "p90_us": float(np.percentile(array, 90)),
    }


def bootstrap(speedups: list[float], dataset_index: int) -> dict[str, object]:
    logs = np.log(np.asarray(speedups, dtype=np.float64))
    seed = BOOTSTRAP_SEED + dataset_index
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(logs), size=(BOOTSTRAP_SAMPLES, len(logs)))
    samples = np.exp(np.mean(logs[indices], axis=1))
    return {
        "seed": seed,
        "samples": BOOTSTRAP_SAMPLES,
        "ci95_lower": float(np.percentile(samples, 2.5)),
        "ci95_upper": float(np.percentile(samples, 97.5)),
    }


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    audit_path = PROJECT / "results/g4c_formal_runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("runtime_binary_gate_pass"):
        raise RuntimeError("Formal runtime binary audit did not pass")
    datasets: dict[str, object] = {}
    evidence: list[dict[str, str]] = []
    all_dataset_gates = True
    for dataset_index, dataset in enumerate(DATASETS):
        process_rows: list[dict[str, object]] = []
        raw = {"keeper_two_stage": [], "candidate_three_stage": []}
        median_speedups: list[float] = []
        sustained_speedups: list[float] = []
        dynamic_contracts: list[bool] = []
        runner_hashes: set[str] = set()
        protocol_hashes: set[str] = set()
        for process_id, order in enumerate(ORDERS):
            stem = f"g4c_formal_{dataset}_n4096_k64_process_{process_id}_{order}_a0"
            result_path = PROJECT / f"results/{stem}.json"
            manifest_path = PROJECT / f"results/{stem}_manifest.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("admitted"):
                raise RuntimeError(f"Non-admitted formal slot: {manifest_path}")
            if manifest.get("result_sha256") != sha256_file(result_path):
                raise RuntimeError("Formal result hash mismatch")
            if not result.get("correctness_pass") or not result.get(
                "process_measurement_pass"
            ):
                raise RuntimeError(f"Unqualified formal result: {result_path}")
            expected = result["expected_dynamic_counts"]
            expected_by_variant = {
                "keeper_two_stage": {
                    "g3b_ambiguous": expected["g3b_ambiguous"],
                    "fp64_refined": expected["g3b_ambiguous"],
                    "final_upper": expected["final_upper"],
                },
                "candidate_three_stage": {
                    "g3b_ambiguous": expected["g3b_ambiguous"],
                    "fp64_refined": expected["fp64_candidate"],
                    "final_upper": expected["final_upper"],
                },
            }
            observed = {name: [] for name in expected_by_variant}
            for sample in result["ordered_samples"]:
                observed[sample["variant"]].append(sample["dynamic_stage_counts"])
            dynamic_pass = all(
                values
                and all(value == expected_by_variant[name] for value in values)
                and result["sustained"][name]["unique_dynamic_stage_counts"]
                == [expected_by_variant[name]]
                for name, values in observed.items()
            )
            dynamic_contracts.append(dynamic_pass)
            for name in raw:
                raw[name].extend(float(value) for value in result["metrics_us"][name])
            median_speedup = float(
                result["paired_process_median_speedup_keeper_over_candidate"]
            )
            sustained_speedup = float(
                result["sustained_speedup_keeper_over_candidate"]
            )
            median_speedups.append(median_speedup)
            sustained_speedups.append(sustained_speedup)
            runner_hashes.add(result["runner_sha256"])
            protocol_hashes.add(result["protocol_sha256"])
            process_rows.append(
                {
                    "process_id": process_id,
                    "order": order.upper(),
                    "keeper_median_us": result["medians_us"]["keeper_two_stage"],
                    "candidate_median_us": result["medians_us"]["candidate_three_stage"],
                    "median_speedup_keeper_over_candidate": median_speedup,
                    "keeper_sustained_mean_us": result["sustained"]["keeper_two_stage"]["mean_us_per_pipeline"],
                    "candidate_sustained_mean_us": result["sustained"]["candidate_three_stage"]["mean_us_per_pipeline"],
                    "sustained_speedup_keeper_over_candidate": sustained_speedup,
                    "dynamic_count_contract_pass": dynamic_pass,
                    "result": str(result_path.relative_to(PROJECT)),
                    "result_sha256": sha256_file(result_path),
                }
            )
            evidence.extend(
                [
                    {"path": str(result_path.relative_to(PROJECT)), "sha256": sha256_file(result_path)},
                    {"path": str(manifest_path.relative_to(PROJECT)), "sha256": sha256_file(manifest_path)},
                ]
            )
        confidence = bootstrap(median_speedups, dataset_index)
        median_geomean = geometric_mean(median_speedups)
        sustained_geomean = geometric_mean(sustained_speedups)
        dataset_gate = bool(
            all(dynamic_contracts)
            and len(runner_hashes) == 1
            and len(protocol_hashes) == 1
            and sum(value > 1.0 for value in median_speedups) == 8
            and float(confidence["ci95_lower"]) >= 1.10
            and sum(value > 1.0 for value in sustained_speedups) == 8
            and sustained_geomean >= 1.10
        )
        all_dataset_gates = all_dataset_gates and dataset_gate
        datasets[dataset] = {
            "processes": process_rows,
            "marginal_raw_sample_percentiles_us": {
                name: percentiles(values) for name, values in raw.items()
            },
            "paired_geometric_mean_median_speedup": median_geomean,
            "paired_process_bootstrap": confidence,
            "process_median_wins": sum(value > 1.0 for value in median_speedups),
            "sustained_geometric_mean_speedup": sustained_geomean,
            "sustained_wins": sum(value > 1.0 for value in sustained_speedups),
            "dynamic_count_contract_pass": all(dynamic_contracts),
            "single_runner_sha256": sorted(runner_hashes),
            "single_protocol_sha256": sorted(protocol_hashes),
            "dataset_formal_gate_pass": dataset_gate,
        }
    result = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_formal",
        "decision": (
            "promote_cross_dataset_shape_local_attribution"
            if all_dataset_gates
            else "retain_dataset_boundaries_no_cross_dataset_promotion"
        ),
        "measurement_scope": "host-dispatched dynamic-count resident-GPU operator",
        "scope_is_not": "ingest-inclusive or host-output-materializing end-to-end latency",
        "datasets": datasets,
        "formal_thresholds_per_dataset": {
            "process_median_wins_required": 8,
            "bootstrap_ci95_lower_minimum": 1.10,
            "sustained_wins_required": 8,
            "sustained_geometric_mean_minimum": 1.10,
        },
        "all_three_dataset_formal_gates_pass": all_dataset_gates,
        "formal_gate_pass": all_dataset_gates,
        "runtime_binary_gate_pass": True,
        "runtime_audit_sha256": sha256_file(audit_path),
        "evidence": evidence,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4C_FORMAL.md"),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if all_dataset_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
