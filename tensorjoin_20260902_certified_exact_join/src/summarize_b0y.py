#!/usr/bin/env python3
"""Summarize the frozen eight-process B0Y guarded-FP32 campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEED = 20260902
BOOTSTRAP_SAMPLES = 20_000
TARGETS = (1, 8, 64)


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


def ratio_summary(
    records: list[dict], numerator: str, denominator: str
) -> dict[str, object]:
    ratios = []
    order_ratios: dict[str, list[float]] = {"AB": [], "BA": []}
    for record in records:
        num = float(np.median(record["metrics_us"][numerator]))
        den = float(np.median(record["metrics_us"][denominator]))
        ratio = num / den
        ratios.append(ratio)
        order_ratios[record["order"]].append(ratio)
    logs = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(SEED)
    samples = rng.choice(logs, size=(BOOTSTRAP_SAMPLES, len(logs)), replace=True)
    boot = np.exp(samples.mean(axis=1))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "per_process_ratios": ratios,
        "geomean_speedup": float(np.exp(logs.mean())),
        "bootstrap_95_low": float(np.percentile(boot, 2.5)),
        "bootstrap_95_high": float(np.percentile(boot, 97.5)),
        "process_wins": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        "order_split_geomean": {
            key: float(math.exp(np.log(values).mean())) for key, values in order_ratios.items()
        },
        "numerator_marginal": distribution(
            [sample for record in records for sample in record["metrics_us"][numerator]]
        ),
        "denominator_marginal": distribution(
            [sample for record in records for sample in record["metrics_us"][denominator]]
        ),
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("b0y_process_*.json"))
    if len(paths) != 8:
        raise ValueError(f"Expected 8 process records, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records.sort(key=lambda record: int(record["process_id"]))
    expected_orders = ["AB", "BA"] * 4
    observed_orders = [record["order"] for record in records]
    if observed_orders != expected_orders:
        raise ValueError((observed_orders, expected_orders))
    hashes = {record["script_sha256"] for record in records}
    caches = {record["feature_cache_sha256"] for record in records}
    correctness_pass = all(
        record["correctness"]["int_dot_mismatch"] == 0
        and all(
            row["keeper_mismatch"] == 0
            and row["candidate_mismatch"] == 0
            and row["candidate_duplicate_ids"] == 0
            for row in record["correctness"]["thresholds"].values()
        )
        for record in records
    )
    dense = ratio_summary(records, "dense_fp32", "dense_int8")
    end_to_end = {
        str(target): ratio_summary(
            records,
            f"baseline_fp32_certified_e2e_s{target}",
            f"candidate_e2e_s{target}",
        )
        for target in TARGETS
    }
    dense_pass = dense["bootstrap_95_low"] >= 1.5 and dense["process_wins"] >= 7
    e2e_pass = all(
        row["bootstrap_95_low"] >= 1.5 and row["process_wins"] >= 7
        for row in end_to_end.values()
    )
    component_medians = {
        metric: distribution(
            [sample for record in records for sample in record["metrics_us"][metric]]
        )
        for metric in records[0]["metrics_us"]
    }
    summary = {
        "experiment_id": records[0]["experiment_id"],
        "process_count": len(records),
        "orders": observed_orders,
        "script_sha256_values": sorted(hashes),
        "feature_cache_sha256_values": sorted(caches),
        "correctness_pass": correctness_pass,
        "dense_int8_vs_fp32": dense,
        "end_to_end_guarded_fp32_vs_candidate": end_to_end,
        "dense_gate_pass": bool(dense_pass),
        "end_to_end_gate_pass": bool(e2e_pass),
        "gate_pass": bool(correctness_pass and dense_pass and e2e_pass),
        "component_marginal_distributions": component_medians,
        "input_files": [str(path) for path in paths],
        "bootstrap_seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
