#!/usr/bin/env python3
"""Summarize the predeclared H2B-P4 early-stop rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/h2b_p4_process_0.json"
LOG = PROJECT / "raw/h2b_p4_process_0.log"
PREFLIGHT = PROJECT / "raw/h2b_p4_preflight.txt"
POSTFLIGHT = PROJECT / "raw/h2b_p4_postflight.txt"
OUTPUT = PROJECT / "results/h2b_p4_rejection.json"
TEXT = PROJECT / "raw/h2b_p4_rejection.txt"
EXPECTED = {
    RESULT: "c8ac7d4e8f69daa468545afddcc40097d02a9e50233fc09604c09073fd382c9c",
    LOG: "68db0aa926e3f5f02a86d10f4d1ef5e2ede88d5f64f337fed42936f80550ba96",
    PREFLIGHT: "4d30afee0e2774ea30029c90629012b7c3a851688999a8ebbea37f80f6c29684",
    POSTFLIGHT: "1c57e7fea908a79d684b02e890170d46c4d5a437813078bd9c7b01f2942c4506",
    PROJECT / "src/run_h2b_p4_timing.py": "b6ea11bc5948bf26cb29d49ae415cd0cb570bb8f98cf74123a5c33f40c6574e0",
}
MIN_SPEEDUP = 1.25
GPU_UUID = "GPU-863c06a5-9f33-0265-b098-013fa840d5db"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists() or TEXT.exists():
        raise FileExistsError("P4 rejection summary already exists")
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"P4 evidence hash mismatch: {path}")
    result = json.loads(RESULT.read_text())
    if result["slot"] != 0 or not result["all_exact"] or result["all_process_medians_won"]:
        raise RuntimeError("unexpected P4 slot-0 header")
    cells = []
    total_samples = 0
    for dataset in result["datasets"]:
        for cell in dataset["cells"]:
            samples = cell["samples"]
            total_samples += len(samples)
            if len(samples) != 200 or not all(row["exact"] for row in samples):
                raise RuntimeError("missing or inexact retained P4 samples")
            order_diagnostics = {}
            for order in ("BC", "CB"):
                order_diagnostics[order] = {
                    variant: {
                        "observations": len(values),
                        "median_us": float(np.median(values)),
                        "p90_us": float(np.percentile(values, 90)),
                    }
                    for variant in ("baseline", "candidate")
                    for values in [
                        [
                            row["wall_us"]
                            for row in samples
                            if row["order"] == order and row["variant"] == variant
                        ]
                    ]
                }
            paired = {}
            for row in samples:
                paired.setdefault(row["observation"], {})[row["variant"]] = row["wall_us"]
            paired_ratios = np.asarray(
                [values["baseline"] / values["candidate"] for values in paired.values()],
                dtype=np.float64,
            )
            speedup = float(cell["median_speedup"])
            cells.append(
                {
                    "dataset": cell["dataset"],
                    "dimension": cell["dimension"],
                    "target_results_per_query": cell["target_results_per_query"],
                    "retained_observations_per_variant": 100,
                    "exact_calls": 200,
                    "baseline_median_us": cell["wall"]["baseline"]["median_us"],
                    "candidate_median_us": cell["wall"]["candidate"]["median_us"],
                    "median_speedup": speedup,
                    "candidate_won_process_median": speedup > 1.0,
                    "passed_promotion_threshold": speedup > MIN_SPEEDUP,
                    "paired_observation_wins": int(np.count_nonzero(paired_ratios > 1.0)),
                    "paired_ratio_median": float(np.median(paired_ratios)),
                    "paired_ratio_p10": float(np.percentile(paired_ratios, 10)),
                    "paired_ratio_p90": float(np.percentile(paired_ratios, 90)),
                    "order_diagnostics": order_diagnostics,
                }
            )
    if total_samples != 800:
        raise RuntimeError("P4 retained sample count mismatch")
    post_lines = POSTFLIGHT.read_text().splitlines()
    apps_index = post_lines.index("COMPUTE_APPS")
    apps_end = post_lines.index("UNEXECUTED_ARTIFACT_CHECK")
    if any(GPU_UUID in line for line in post_lines[apps_index + 1 : apps_end]):
        raise RuntimeError("GPU4 was occupied at P4 postflight")
    not_created = post_lines[apps_end + 1 :]
    expected_not_created = [f"slot_{slot}=not_created" for slot in range(1, 8)] + [
        "sustained=not_created"
    ]
    if not_created != expected_not_created:
        raise RuntimeError("P4 early-stop artifact state mismatch")
    summary = {
        "experiment": "tensorjoin_20260903_h2b_p4_early_stop",
        "decision": "REJECT_H2B_PHASE_P",
        "reason": "slot 0 violates the predeclared every-process and 1.25x gates",
        "stop_rule_applied": True,
        "physical_gpu": 4,
        "gpu_uuid": GPU_UUID,
        "operator_scope": "complete host outer wall through dynamic repair, final ID transfer, and canonical sort",
        "cells": cells,
        "retained_exact_calls": total_samples,
        "unexecuted_by_design": {
            "fresh_process_slots": list(range(1, 8)),
            "sustained": True,
            "bootstrap_interval": True,
        },
        "inference": {
            "phase_p_performance_thesis": "closed",
            "streaming_phase": "blocked",
            "tree_index_work": "blocked",
            "novel_exactness_mechanism": "not disproved by timing, but insufficient for the proposed performance thesis",
        },
        "evidence_sha256": {
            str(path.relative_to(PROJECT)): digest for path, digest in EXPECTED.items()
        },
        "summarizer_sha256": sha256_file(Path(__file__)),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(payload)
    output_hash = hashlib.sha256(payload.encode()).hexdigest()
    lines = [
        "decision=REJECT_H2B_PHASE_P",
        "stop_rule_applied=true",
        "retained_exact_calls=800",
    ]
    for cell in cells:
        lines.append(
            f"{cell['dataset']}.t{cell['target_results_per_query']}.median_speedup={cell['median_speedup']}"
        )
    lines.extend(
        [
            "fresh_process_slots_1_7=not_executed",
            "sustained=not_executed",
            "streaming=blocked",
            "tree_index=blocked",
            f"result_sha256={output_hash}",
        ]
    )
    TEXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
