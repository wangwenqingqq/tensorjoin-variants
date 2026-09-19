#!/usr/bin/env python3
"""Summarize the frozen six-slot G4C public-anchor screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("sift128", "cifar_gist512", "fashion784")
ORDERS = ("kc", "ck")
RESULT = PROJECT / "results/g4c_screen_summary.json"


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


def main() -> int:
    records: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in DATASETS}
    admissions: dict[str, list[bool]] = {dataset: [] for dataset in DATASETS}
    evidence: list[dict[str, str]] = []
    for dataset in DATASETS:
        for process_id, order in enumerate(ORDERS):
            stem = f"g4c_screen_{dataset}_n4096_k64_process_{process_id}_{order}_a0"
            result_path = PROJECT / f"results/{stem}.json"
            manifest_path = PROJECT / f"results/{stem}_manifest.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("result_sha256") != sha256_file(result_path):
                raise RuntimeError("Isolation result hash mismatch")
            if result.get("dataset_id") != dataset or manifest.get("dataset_id") != dataset:
                raise RuntimeError("Dataset mismatch")
            records[dataset].append(result)
            admissions[dataset].append(bool(manifest.get("admitted")))
            evidence.extend(
                [
                    {"path": str(result_path.relative_to(PROJECT)), "sha256": sha256_file(result_path)},
                    {"path": str(manifest_path.relative_to(PROJECT)), "sha256": sha256_file(manifest_path)},
                ]
            )
    dataset_summaries: dict[str, object] = {}
    passing_datasets = 0
    all_clean = True
    for dataset, values in records.items():
        median_speedups = [
            float(value["paired_process_median_speedup_keeper_over_candidate"])
            for value in values
        ]
        sustained_speedups = [
            float(value["sustained_speedup_keeper_over_candidate"])
            for value in values
        ]
        clean = all(
            value.get("correctness_pass") and value.get("process_measurement_pass")
            for value in values
        ) and all(admissions[dataset])
        median_geomean = geometric_mean(median_speedups)
        sustained_geomean = geometric_mean(sustained_speedups)
        performance_pass = bool(
            clean and median_geomean > 1.10 and sustained_geomean > 1.10
        )
        passing_datasets += int(performance_pass)
        all_clean = all_clean and clean
        dataset_summaries[dataset] = {
            "process_median_speedups": median_speedups,
            "process_sustained_speedups": sustained_speedups,
            "paired_geometric_mean_median_speedup": median_geomean,
            "paired_geometric_mean_sustained_speedup": sustained_geomean,
            "clean_correctness": clean,
            "screen_performance_pass": performance_pass,
        }
    screen_gate_pass = bool(all_clean and passing_datasets >= 2)
    summary = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_screen",
        "measurement_status": "screen_not_paper_claim",
        "selected_cells": [
            {"dataset_id": dataset, "n": 4096, "target_degree": 64}
            for dataset in DATASETS
        ],
        "dataset_summaries": dataset_summaries,
        "all_six_slots_clean_and_exact": all_clean,
        "datasets_above_both_1_10_thresholds": passing_datasets,
        "screen_gate_pass": screen_gate_pass,
        "formal_all_three_anchors_allowed": screen_gate_pass,
        "performance_claim_allowed": False,
        "evidence": evidence,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4C_SCREEN.md"),
    }
    atomic_json(RESULT, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if screen_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
