#!/usr/bin/env python3
"""Summarize the frozen two-radius D1 learned-audio campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEED = 20260902
BOOTSTRAP_SAMPLES = 20_000
TARGETS = (1, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_us": float(np.percentile(array, 10)),
        "median_us": float(np.median(array)),
        "p90_us": float(np.percentile(array, 90)),
    }


def summarize_target(input_dir: Path, target: int) -> dict[str, object]:
    paths = sorted(input_dir.glob(f"d1_t{target}_process_*.json"))
    if len(paths) != 8:
        raise ValueError(f"Target {target}: expected 8 records, found {len(paths)}")
    records = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in paths),
        key=lambda row: int(row["process_id"]),
    )
    orders = [record["order"] for record in records]
    if orders != ["AB", "BA"] * 4:
        raise ValueError((target, orders))
    if any(int(record["target_results_per_query"]) != target for record in records):
        raise ValueError(f"Target label mismatch for {target}")
    ratios = []
    order_ratios: dict[str, list[float]] = {"AB": [], "BA": []}
    for record in records:
        baseline = float(np.median(record["metrics_us"]["baseline"]))
        candidate = float(np.median(record["metrics_us"]["candidate"]))
        ratio = baseline / candidate
        ratios.append(ratio)
        order_ratios[record["order"]].append(ratio)
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(SEED + target)
    bootstrap = np.exp(
        rng.choice(log_ratios, (BOOTSTRAP_SAMPLES, len(log_ratios)), replace=True).mean(axis=1)
    )
    baseline_samples = [value for row in records for value in row["metrics_us"]["baseline"]]
    candidate_samples = [value for row in records for value in row["metrics_us"]["candidate"]]
    result: dict[str, object] = {
        "target_results_per_query": target,
        "actual_results_per_query_values": sorted(
            {float(row["actual_results_per_query"]) for row in records}
        ),
        "orders": orders,
        "correctness_pass": all(row["correctness_pass"] for row in records),
        "per_process_ratios": ratios,
        "geomean_speedup": float(np.exp(log_ratios.mean())),
        "bootstrap_95_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_95_high": float(np.percentile(bootstrap, 97.5)),
        "process_wins": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        "order_split_geomean": {
            key: float(math.exp(np.log(values).mean())) for key, values in order_ratios.items()
        },
        "baseline_marginal": distribution(baseline_samples),
        "candidate_marginal": distribution(candidate_samples),
        "script_sha256_values": sorted({row["script_sha256"] for row in records}),
        "feature_cache_sha256_values": sorted(
            {row["feature_cache_sha256"] for row in records}
        ),
        "input_files": [str(path) for path in paths],
        "bootstrap_seed": SEED + target,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    result["gate_pass"] = bool(
        result["correctness_pass"]
        and float(result["bootstrap_95_low"]) >= 1.5
        and int(result["process_wins"]) >= 7
    )
    return result


def main() -> int:
    args = parse_args()
    targets = {str(target): summarize_target(args.input_dir, target) for target in TARGETS}
    result = {
        "experiment_id": "tensorjoin_20260902_panns_triton_d1",
        "targets": targets,
        "gate_pass": all(bool(row["gate_pass"]) for row in targets.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
