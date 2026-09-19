#!/usr/bin/env python3
"""Summarize the frozen eight-process C0B campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEED = 20260902
BOOTSTRAP_SAMPLES = 20_000


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


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("c0b_process_*.json"))
    if len(paths) != 8:
        raise ValueError(f"Expected 8 process records, found {len(paths)}")
    records = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in paths),
        key=lambda row: int(row["process_id"]),
    )
    orders = [record["order"] for record in records]
    if orders != ["AB", "BA"] * 4:
        raise ValueError(orders)
    correctness_pass = all(record["correctness_pass"] for record in records)
    ratios = []
    order_ratios: dict[str, list[float]] = {"AB": [], "BA": []}
    for record in records:
        baseline = float(np.median(record["metrics_us"]["baseline"]))
        candidate = float(np.median(record["metrics_us"]["candidate"]))
        ratio = baseline / candidate
        ratios.append(ratio)
        order_ratios[record["order"]].append(ratio)
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(SEED)
    bootstrap = np.exp(
        rng.choice(log_ratios, (BOOTSTRAP_SAMPLES, len(log_ratios)), replace=True).mean(axis=1)
    )
    baseline_samples = [v for r in records for v in r["metrics_us"]["baseline"]]
    candidate_samples = [v for r in records for v in r["metrics_us"]["candidate"]]
    result = {
        "experiment_id": records[0]["experiment_id"],
        "process_count": len(records),
        "orders": orders,
        "correctness_pass": correctness_pass,
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
        "script_sha256_values": sorted({r["script_sha256"] for r in records}),
        "feature_cache_sha256_values": sorted(
            {r["feature_cache_sha256"] for r in records}
        ),
        "input_files": [str(path) for path in paths],
        "bootstrap_seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    result["gate_pass"] = bool(
        correctness_pass
        and result["bootstrap_95_low"] >= 1.5
        and result["process_wins"] >= 7
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
