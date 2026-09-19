#!/usr/bin/env python3
"""Summarize the frozen three-modality F1 promotion campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEED = 20260903
BOOTSTRAP_SAMPLES = 20_000
MODALITIES = ("audio", "video", "hsi")
EXPECTED_ORDERS = ["GTC", "CTG", "TCG", "GCT", "CGT", "TGC", "GTC", "CTG"]
REQUIRED_GUARDED = {"audio": 1.50, "video": 1.35, "hsi": 1.35}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=list(MODALITIES),
        help="Summarize only completed modalities; the output is marked partial.",
    )
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_us": float(np.percentile(array, 10)),
        "median_us": float(np.median(array)),
        "p90_us": float(np.percentile(array, 90)),
    }


def comparison(
    records: list[dict[str, object]],
    numerator: str,
    denominator: str,
    required_low: float,
    seed_offset: int,
) -> dict[str, object]:
    ratios: list[float] = []
    by_denominator_position: dict[str, list[float]] = {"0": [], "1": [], "2": []}
    by_order: dict[str, list[float]] = {}
    for record in records:
        metrics = record["metrics_us"]
        ratio = float(np.median(metrics[numerator])) / float(np.median(metrics[denominator]))
        ratios.append(ratio)
        position = str(record["order"].index("C"))
        by_denominator_position[position].append(ratio)
        by_order.setdefault(record["order"], []).append(ratio)
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    rng = np.random.default_rng(SEED + seed_offset)
    bootstrap = np.exp(
        rng.choice(log_ratios, (BOOTSTRAP_SAMPLES, len(log_ratios)), replace=True).mean(axis=1)
    )
    result: dict[str, object] = {
        "numerator": numerator,
        "denominator": denominator,
        "required_95_low": required_low,
        "per_process_ratios": ratios,
        "geomean_speedup": float(np.exp(log_ratios.mean())),
        "bootstrap_95_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_95_high": float(np.percentile(bootstrap, 97.5)),
        "process_wins": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        "denominator_position_geomean": {
            key: float(math.exp(np.log(values).mean()))
            for key, values in by_denominator_position.items()
            if values
        },
        "order_geomean": {
            key: float(math.exp(np.log(values).mean())) for key, values in by_order.items()
        },
        "bootstrap_seed": SEED + seed_offset,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    result["gate_pass"] = bool(
        result["bootstrap_95_low"] >= required_low and result["process_wins"] == 8
    )
    return result


def summarize_modality(
    input_dir: Path, raw_dir: Path, modality: str, modality_index: int
) -> dict[str, object]:
    paths = []
    for process_id in range(8):
        suffix = "_replacement" if modality == "video" and process_id == 3 else ""
        paths.append(input_dir / f"f1_{modality}_process_{process_id}{suffix}.json")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"{modality}: missing formal records: {missing}")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    process_ids = [int(record["process_id"]) for record in records]
    if process_ids != list(range(8)):
        raise ValueError((modality, process_ids))
    orders = [record["order"] for record in records]
    if orders != EXPECTED_ORDERS:
        raise ValueError((modality, orders))
    if any(record["modality"] != modality for record in records):
        raise ValueError(f"{modality}: modality label mismatch")

    memcheck_path = input_dir / f"f1_{modality}_memcheck.json"
    stress_path = input_dir / f"f1_{modality}_stress1000.json"
    memcheck = json.loads(memcheck_path.read_text(encoding="utf-8"))
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    memcheck_log = (raw_dir / f"f1_{modality}_memcheck.log").read_text(
        encoding="utf-8", errors="replace"
    )
    memcheck_pass = bool(
        memcheck["correctness_pass"]
        and "ERROR SUMMARY: 0 errors" in memcheck_log
        and "ERROR SUMMARY: 1 errors" not in memcheck_log
    )
    stress_pass = bool(
        stress["correctness_pass"]
        and stress["stress_pass"]
        and int(stress["stress_launches"]) == 1000
        and len(stress["stress_hashes"]) == 1
    )
    formal_correctness = all(
        bool(record["correctness_pass"]) and bool(record["stress_pass"])
        for record in records
    )
    variants = ("guarded_fp32", "two_stage", "cascade")
    marginals = {
        variant: distribution(
            [value for record in records for value in record["metrics_us"][variant]]
        )
        for variant in variants
    }
    comparisons = {
        "two_stage_over_cascade": comparison(
            records, "two_stage", "cascade", 1.15, modality_index * 10 + 1
        ),
        "guarded_fp32_over_cascade": comparison(
            records,
            "guarded_fp32",
            "cascade",
            REQUIRED_GUARDED[modality],
            modality_index * 10 + 2,
        ),
    }
    result: dict[str, object] = {
        "modality": modality,
        "orders": orders,
        "formal_correctness_pass": formal_correctness,
        "memcheck_pass": memcheck_pass,
        "stress_pass": stress_pass,
        "comparisons": comparisons,
        "marginal_distributions": marginals,
        "actual_results_per_query_values": sorted(
            {float(row["correctness"]["actual_results_per_query"]) for row in records}
        ),
        "fp64_refine_pairs_values": sorted(
            {int(row["correctness"]["fp64_refine_pairs"]) for row in records}
        ),
        "script_sha256_values": sorted({row["script_sha256"] for row in records}),
        "stage1_source_sha256_values": sorted(
            {row["stage1_source_sha256"] for row in records}
        ),
        "feature_cache_sha256_values": sorted(
            {row["feature_cache_sha256"] for row in records}
        ),
        "input_files": [str(path) for path in paths],
        "excluded_contaminated_files": (
            [str(input_dir / "f1_video_process_3.json")] if modality == "video" else []
        ),
        "memcheck_file": str(memcheck_path),
        "stress_file": str(stress_path),
    }
    result["gate_pass"] = bool(
        formal_correctness
        and memcheck_pass
        and stress_pass
        and all(bool(row["gate_pass"]) for row in comparisons.values())
    )
    return result


def main() -> int:
    args = parse_args()
    selected_modalities = tuple(dict.fromkeys(args.modalities))
    modalities = {
        modality: summarize_modality(
            args.input_dir, args.raw_dir, modality, MODALITIES.index(modality)
        )
        for modality in selected_modalities
    }
    partial = set(selected_modalities) != set(MODALITIES)
    result = {
        "experiment_id": "tensorjoin_20260903_gpu_cascade_f1",
        "coverage": list(selected_modalities),
        "partial": partial,
        "modalities": modalities,
        "gate_pass": bool(
            not partial and all(bool(row["gate_pass"]) for row in modalities.values())
        ),
        "covered_modalities_gate_pass": all(
            bool(row["gate_pass"]) for row in modalities.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["covered_modalities_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
