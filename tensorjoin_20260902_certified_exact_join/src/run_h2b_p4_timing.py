#!/usr/bin/env python3
"""Run frozen H2B-P4 fresh-process and sustained outer-wall timing."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton

import run_h1_multivector_gpu_screen as h1
import run_h2b_p1_pedantic_correctness as p1


EXPERIMENT_ID = "tensorjoin_20260903_h2b_p4_outer_wall"
PROJECT = Path(__file__).resolve().parents[1]
PHYSICAL_GPU = 4
FRESH_PROCESS_SLOTS = 8
WARMUPS_PER_VARIANT = 10
OBSERVATIONS_PER_VARIANT = 100
SUSTAINED_WARMUPS_PER_VARIANT = 20
SUSTAINED_BLOCK_OBSERVATIONS = 500
MIN_PROMOTION_SPEEDUP = 1.25

DEPENDENCIES = {
    "h1_correctness": (PROJECT / "results/h1_r1_multivector_correctness.json", "06461064d1b237aaa29f4face0eb3c223f5487abf478ddcf5fccb353ed28dca4"),
    "h1_runner": (PROJECT / "src/run_h1_multivector_gpu_screen.py", "11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e"),
    "h1_kernel": (PROJECT / "src/h1_multivector_kernels.py", "c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db"),
    "p1_correctness": (PROJECT / "results/h2b_p1_pedantic_correctness.json", "68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99"),
    "p1_runner": (PROJECT / "src/run_h2b_p1_pedantic_correctness.py", "d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1"),
    "p1_kernel": (PROJECT / "src/h2b_pedantic_kernels.py", "33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43"),
    "p2_audit": (PROJECT / "results/h2b_p2_cublas_sass_audit.json", "c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77"),
    "p3_safety": (PROJECT / "results/h2b_p3_safety_summary.json", "3a02d1a55016c70989d59ace2b1003d1ccbde9ba2bb7720bee45354ae9efd245"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("process", "sustained"), required=True)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    if args.phase == "process" and args.slot not in range(FRESH_PROCESS_SLOTS):
        parser.error(f"--phase process requires --slot in [0,{FRESH_PROCESS_SLOTS - 1}]")
    if args.phase == "sustained" and args.slot is not None:
        parser.error("--slot is valid only for --phase process")
    return args


def require_dependencies() -> tuple[dict, dict, dict]:
    for name, (path, expected) in DEPENDENCIES.items():
        if p1.sha256_file(path) != expected:
            raise RuntimeError(f"P4 dependency mismatch: {name} / {path}")
    h1_correctness = json.loads(DEPENDENCIES["h1_correctness"][0].read_text())
    p1_correctness = json.loads(DEPENDENCIES["p1_correctness"][0].read_text())
    if not h1_correctness.get("correctness_pass") or not p1_correctness.get("correctness_pass"):
        raise RuntimeError("P4 requires accepted H1 and H2B-P1 correctness")
    p2 = json.loads(DEPENDENCIES["p2_audit"][0].read_text())
    p3 = json.loads(DEPENDENCIES["p3_safety"][0].read_text())
    if p2.get("decision") != "PASS" or p3.get("decision") != "PASS":
        raise RuntimeError("P4 requires accepted P2 and P3")
    return h1_correctness, p1_correctness, p3


def expected_contracts(
    h1_result: dict, p1_result: dict, p3_result: dict, dataset: str, target: int
) -> dict:
    h1_dataset = next(row for row in h1_result["datasets"] if row["dataset"] == dataset)
    p1_dataset = next(row for row in p1_result["datasets"] if row["dataset"] == dataset)
    h1_cell = next(
        row for row in h1_dataset["cells"] if int(row["target_results_per_query"]) == target
    )
    p1_cell = next(
        row for row in p1_dataset["cells"] if int(row["target_results_per_query"]) == target
    )
    if h1_cell["oracle_ids_sha256"] != p1_cell["oracle_ids_sha256"]:
        raise RuntimeError(f"oracle hash disagreement: {dataset}/t{target}")
    if int(h1_cell["oracle_count"]) != int(p1_cell["oracle_count"]):
        raise RuntimeError(f"oracle count disagreement: {dataset}/t{target}")
    p3_cell = next(
        row
        for row in p3_result["stress"]["cells"]
        if row["dataset"] == dataset and int(row["target_results_per_query"]) == target
    )
    common = {
        "count": int(p1_cell["oracle_count"]),
        "ids_sha256": str(p1_cell["oracle_ids_sha256"]),
    }
    return {
        "candidate": {
            **common,
            "ambiguous_count": int(h1_cell["ambiguous_object_pairs"]),
            "direct_accept_count": int(h1_cell["direct_accept_object_pairs"]),
        },
        "baseline": {
            **common,
            "ambiguous_count": int(p1_cell["baseline_ambiguous_objects"]),
            "direct_accept_count": int(p3_cell["signature"]["direct_accept_count"]),
        },
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10_us": float(np.percentile(array, 10)),
        "median_us": float(np.median(array)),
        "p90_us": float(np.percentile(array, 90)),
        "mean_us": float(np.mean(array)),
        "std_us": float(np.std(array)),
    }


def checked_call(function, expected: dict) -> tuple[dict, float]:
    started_ns = time.perf_counter_ns()
    output = function()
    wall_us = (time.perf_counter_ns() - started_ns) / 1000.0
    ids = np.asarray(output["ids"], dtype=np.uint64)
    record = {
        "wall_us": float(wall_us),
        "final_count": int(output["final_count"]),
        "direct_accept_count": int(output["direct_accept_count"]),
        "ambiguous_count": int(output["ambiguous_count"]),
        "overflow": int(output["overflow"]),
        "duplicate_ids": h1.duplicate_count(ids),
        "ids_sha256": p1.sha256_u64(ids),
    }
    record["exact"] = bool(
        record["final_count"] == expected["count"]
        and record["direct_accept_count"] == expected["direct_accept_count"]
        and record["ambiguous_count"] == expected["ambiguous_count"]
        and record["ids_sha256"] == expected["ids_sha256"]
        and record["overflow"] == 0
        and record["duplicate_ids"] == 0
    )
    if not record["exact"]:
        raise RuntimeError(f"P4 timed output mismatch: {record} != {expected}")
    return record, wall_us


def make_functions(data: h1.PreparedDataset, cublas: p1.CuBLAS):
    state = p1.make_state(data)
    functions = {
        "baseline": lambda threshold: p1.run_baseline(state, cublas, threshold),
        "candidate": lambda threshold: h1.run_candidate(state.h1_state, threshold),
    }
    return state, functions


def warm(functions: dict, threshold: float, contracts: dict, count: int) -> None:
    for _ in range(count):
        for variant in ("baseline", "candidate"):
            checked_call(lambda v=variant: functions[v](threshold), contracts[variant])


def run_process(
    slot: int, h1_result: dict, p1_result: dict, p3_result: dict, cublas: p1.CuBLAS
) -> int:
    output_path = PROJECT / f"results/h2b_p4_process_{slot}.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.time()
    environment_pre = p1.environment_record(cublas)
    dataset_names = list(h1.DATASETS)
    if slot & 1:
        dataset_names.reverse()
    datasets = []
    cell_sequence = 0
    for dataset in dataset_names:
        data = h1.prepare_dataset(dataset)
        state, functions = make_functions(data, cublas)
        targets = sorted(data.thresholds)
        if (slot // 2) & 1:
            targets.reverse()
        cells = []
        for target in targets:
            threshold = data.thresholds[target]
            contracts = expected_contracts(h1_result, p1_result, p3_result, dataset, target)
            warm(functions, threshold, contracts, WARMUPS_PER_VARIANT)
            samples = []
            values = {"baseline": [], "candidate": []}
            for observation in range(OBSERVATIONS_PER_VARIANT):
                baseline_first = ((slot + cell_sequence + observation) & 1) == 0
                order = ("baseline", "candidate") if baseline_first else ("candidate", "baseline")
                for position, variant in enumerate(order):
                    checked, wall_us = checked_call(
                        lambda v=variant: functions[v](threshold), contracts[variant]
                    )
                    values[variant].append(wall_us)
                    samples.append(
                        {
                            "sequence": len(samples),
                            "observation": observation,
                            "order": "BC" if baseline_first else "CB",
                            "position": position,
                            "variant": variant,
                            **checked,
                        }
                    )
            baseline_summary = summarize(values["baseline"])
            candidate_summary = summarize(values["candidate"])
            speedup = baseline_summary["median_us"] / candidate_summary["median_us"]
            cells.append(
                {
                    "dataset": dataset,
                    "dimension": data.dimension,
                    "target_results_per_query": target,
                    "threshold": threshold,
                    "warmups_per_variant": WARMUPS_PER_VARIANT,
                    "observations_per_variant": OBSERVATIONS_PER_VARIANT,
                    "contracts": contracts,
                    "wall": {
                        "baseline": baseline_summary,
                        "candidate": candidate_summary,
                    },
                    "median_speedup": float(speedup),
                    "candidate_win": bool(speedup > 1.0),
                    "samples": samples,
                }
            )
            print(
                "P4_PROCESS_CELL "
                + json.dumps(
                    {
                        "slot": slot,
                        "dataset": dataset,
                        "target": target,
                        "baseline_median_us": baseline_summary["median_us"],
                        "candidate_median_us": candidate_summary["median_us"],
                        "median_speedup": speedup,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            cell_sequence += 1
        datasets.append({"dataset": dataset, "cells": cells})
        del functions, state
        torch.cuda.empty_cache()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "fresh_process_outer_wall",
        "slot": slot,
        "process_id": os.getpid(),
        "environment_pre": environment_pre,
        "environment_post": p1.environment_record(cublas),
        "datasets": datasets,
        "all_exact": True,
        "all_process_medians_won": all(
            cell["candidate_win"] for dataset in datasets for cell in dataset["cells"]
        ),
        "elapsed_s": time.time() - started,
        "sources": {
            "runner_sha256": p1.sha256_file(Path(__file__)),
            **{name + "_sha256": digest for name, (_, digest) in DEPENDENCIES.items()},
        },
    }
    p1.atomic_json(output_path, result)
    print("P4_PROCESS_DONE " + json.dumps({"slot": slot, "output": str(output_path)}, sort_keys=True))
    return 0 if result["all_process_medians_won"] else 3


def run_sustained(
    h1_result: dict, p1_result: dict, p3_result: dict, cublas: p1.CuBLAS
) -> int:
    output_path = PROJECT / "results/h2b_p4_sustained.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.time()
    environment_pre = p1.environment_record(cublas)
    datasets = []
    cell_index = 0
    for dataset in h1.DATASETS:
        data = h1.prepare_dataset(dataset)
        state, functions = make_functions(data, cublas)
        cells = []
        for target, threshold in sorted(data.thresholds.items()):
            contracts = expected_contracts(h1_result, p1_result, p3_result, dataset, target)
            warm(functions, threshold, contracts, SUSTAINED_WARMUPS_PER_VARIANT)
            blocks = []
            all_values = {"baseline": [], "candidate": []}
            for block in range(2):
                baseline_first = ((cell_index + block) & 1) == 0
                order = ("baseline", "candidate") if baseline_first else ("candidate", "baseline")
                observations = []
                block_values = {"baseline": [], "candidate": []}
                for variant in order:
                    for iteration in range(SUSTAINED_BLOCK_OBSERVATIONS):
                        checked, wall_us = checked_call(
                            lambda v=variant: functions[v](threshold), contracts[variant]
                        )
                        block_values[variant].append(wall_us)
                        all_values[variant].append(wall_us)
                        observations.append(
                            {
                                "sequence": len(observations),
                                "variant": variant,
                                "variant_iteration": iteration,
                                **checked,
                            }
                        )
                block_summary = {variant: summarize(values) for variant, values in block_values.items()}
                block_speedup = (
                    block_summary["baseline"]["median_us"]
                    / block_summary["candidate"]["median_us"]
                )
                blocks.append(
                    {
                        "block": block,
                        "order": "B500_C500" if baseline_first else "C500_B500",
                        "wall": block_summary,
                        "median_speedup": float(block_speedup),
                        "promotion_pass": bool(block_speedup > MIN_PROMOTION_SPEEDUP),
                        "observations": observations,
                    }
                )
            overall = {variant: summarize(values) for variant, values in all_values.items()}
            speedup = overall["baseline"]["median_us"] / overall["candidate"]["median_us"]
            cell_pass = bool(
                speedup > MIN_PROMOTION_SPEEDUP
                and all(block["promotion_pass"] for block in blocks)
            )
            cells.append(
                {
                    "dataset": dataset,
                    "dimension": data.dimension,
                    "target_results_per_query": target,
                    "threshold": threshold,
                    "warmups_per_variant": SUSTAINED_WARMUPS_PER_VARIANT,
                    "retained_observations_per_variant": 2 * SUSTAINED_BLOCK_OBSERVATIONS,
                    "contracts": contracts,
                    "wall": overall,
                    "median_speedup": float(speedup),
                    "blocks": blocks,
                    "promotion_pass": cell_pass,
                }
            )
            print(
                "P4_SUSTAINED_CELL "
                + json.dumps(
                    {
                        "dataset": dataset,
                        "target": target,
                        "median_speedup": speedup,
                        "block_speedups": [block["median_speedup"] for block in blocks],
                        "promotion_pass": cell_pass,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            cell_index += 1
        datasets.append({"dataset": dataset, "cells": cells})
        del functions, state
        torch.cuda.empty_cache()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "sustained_outer_wall",
        "environment_pre": environment_pre,
        "environment_post": p1.environment_record(cublas),
        "datasets": datasets,
        "all_exact": True,
        "all_sustained_blocks_pass": all(
            cell["promotion_pass"] for dataset in datasets for cell in dataset["cells"]
        ),
        "elapsed_s": time.time() - started,
        "sources": {
            "runner_sha256": p1.sha256_file(Path(__file__)),
            **{name + "_sha256": digest for name, (_, digest) in DEPENDENCIES.items()},
        },
    }
    p1.atomic_json(output_path, result)
    print("P4_SUSTAINED_DONE " + json.dumps({"output": str(output_path)}, sort_keys=True))
    return 0 if result["all_sustained_blocks_pass"] else 3


def main() -> int:
    args = parse_args()
    h1_result, p1_result, p3_result = require_dependencies()
    p1.require_dependencies()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("H2B-P4 requires NVIDIA_TF32_OVERRIDE=0 before CUDA init")
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    p1.PHYSICAL_GPU = PHYSICAL_GPU
    h1.PHYSICAL_GPU = PHYSICAL_GPU
    p1.require_environment()
    cublas = p1.CuBLAS()
    if args.phase == "process":
        return run_process(args.slot, h1_result, p1_result, p3_result, cublas)
    return run_sustained(h1_result, p1_result, p3_result, cublas)


if __name__ == "__main__":
    raise SystemExit(main())
