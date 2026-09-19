#!/usr/bin/env python3
"""Run the frozen H2B-P3 full-operator safety and stability gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import run_h1_multivector_gpu_screen as h1
import run_h2b_p1_pedantic_correctness as p1


EXPERIMENT_ID = "tensorjoin_20260903_h2b_p3_safety_stress"
PROJECT = Path(__file__).resolve().parents[1]
STRESS_OUTPUT = PROJECT / "results/h2b_p3_stress.json"
P1_CORRECTNESS_SHA256 = "68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99"
P1_REPEAT_SHA256 = {
    0: "6970b41c794ec787e880778f9a501de61331a7eb4cda5303d85130db4215fc27",
    1: "75d22742a4a952aa19bfb8b2a53bd36508b45bc9211a71fe1e51cb84b4edda30",
}
P2_AUDIT_SHA256 = "c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77"
P1_RUNNER_SHA256 = "d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1"
P1_KERNEL_SHA256 = "33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43"
ITERATIONS_PER_CELL = 1000
PHYSICAL_GPU = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("safety", "stress"), required=True)
    parser.add_argument("--dataset", choices=tuple(h1.DATASETS))
    parser.add_argument("--target", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    if args.phase == "safety" and args.dataset is None:
        parser.error("--phase safety requires --dataset")
    if args.phase == "stress" and args.dataset is not None:
        parser.error("--dataset is valid only for --phase safety")
    return args


def require_dependencies() -> dict:
    expected = {
        PROJECT / "src/run_h2b_p1_pedantic_correctness.py": P1_RUNNER_SHA256,
        PROJECT / "src/h2b_pedantic_kernels.py": P1_KERNEL_SHA256,
        PROJECT / "results/h2b_p1_pedantic_correctness.json": P1_CORRECTNESS_SHA256,
        PROJECT / "results/h2b_p1_pedantic_repeat_0.json": P1_REPEAT_SHA256[0],
        PROJECT / "results/h2b_p1_pedantic_repeat_1.json": P1_REPEAT_SHA256[1],
        PROJECT / "results/h2b_p2_cublas_sass_audit.json": P2_AUDIT_SHA256,
    }
    for path, expected_hash in expected.items():
        if p1.sha256_file(path) != expected_hash:
            raise RuntimeError(f"H2B-P3 dependency mismatch: {path}")
    correctness = json.loads(
        (PROJECT / "results/h2b_p1_pedantic_correctness.json").read_text()
    )
    if not correctness.get("correctness_pass"):
        raise RuntimeError("H2B-P3 requires accepted P1 correctness")
    p2 = json.loads((PROJECT / "results/h2b_p2_cublas_sass_audit.json").read_text())
    if p2.get("decision") != "PASS" or not all(p2.get("gate", {}).values()):
        raise RuntimeError("H2B-P3 requires accepted P2 SASS audit")
    return correctness


def expected_cell(correctness: dict, dataset: str, target: int) -> dict:
    dataset_row = next(row for row in correctness["datasets"] if row["dataset"] == dataset)
    return next(
        row
        for row in dataset_row["cells"]
        if int(row["target_results_per_query"]) == target
    )


def checked_output(output: dict, expected: dict) -> dict:
    ids = np.asarray(output["ids"], dtype=np.uint64)
    output_hash = p1.sha256_u64(ids)
    record = {
        "final_count": int(output["final_count"]),
        "direct_accept_count": int(output["direct_accept_count"]),
        "ambiguous_count": int(output["ambiguous_count"]),
        "overflow": int(output["overflow"]),
        "duplicate_ids": h1.duplicate_count(ids),
        "ids_sha256": output_hash,
        "expected_count": int(expected["oracle_count"]),
        "expected_ambiguous_count": int(expected["baseline_ambiguous_objects"]),
        "expected_ids_sha256": str(expected["oracle_ids_sha256"]),
    }
    record["pass"] = bool(
        record["final_count"] == record["expected_count"]
        and record["ambiguous_count"] == record["expected_ambiguous_count"]
        and record["ids_sha256"] == record["expected_ids_sha256"]
        and record["overflow"] == 0
        and record["duplicate_ids"] == 0
    )
    return record


def run_safety(dataset: str, target: int, correctness: dict, cublas: p1.CuBLAS) -> int:
    data = h1.prepare_dataset(dataset)
    state = p1.make_state(data)
    threshold = data.thresholds[target]
    output = p1.run_baseline(state, cublas, threshold)
    checked = checked_output(output, expected_cell(correctness, dataset, target))
    cublas_record = cublas.record()
    record = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "full_actual_data_safety",
        "physical_gpu": PHYSICAL_GPU,
        "dataset": dataset,
        "target_results_per_query": target,
        "dimension": data.dimension,
        "query_objects": len(data.query.slices),
        "base_objects": len(data.base.slices),
        "query_tokens": len(data.query.vectors),
        "base_tokens": len(data.base.vectors),
        "operator": "pedantic SGEMM -> fused object certificate/compaction -> dynamic count -> exact FP64 repair -> final sorted IDs",
        "output": checked,
        "cublas": cublas_record,
        "sources": {
            "runner_sha256": p1.sha256_file(Path(__file__)),
            "p1_runner_sha256": P1_RUNNER_SHA256,
            "p1_kernel_sha256": P1_KERNEL_SHA256,
            "p1_correctness_sha256": P1_CORRECTNESS_SHA256,
            "p2_audit_sha256": P2_AUDIT_SHA256,
        },
    }
    # Compute Sanitizer leak mode must observe released experiment allocations,
    # not PyTorch caching-allocator or cuBLAS-workspace retention at interpreter
    # shutdown.  The safety phase performs no later CUDA work.
    del output, state
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("P3_SAFETY " + json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["output"]["pass"] else 3


def run_stress(correctness: dict, cublas: p1.CuBLAS) -> int:
    if STRESS_OUTPUT.exists():
        raise FileExistsError(STRESS_OUTPUT)
    started = time.time()
    datasets = []
    for dataset in h1.DATASETS:
        data = h1.prepare_dataset(dataset)
        states = [p1.make_state(data), p1.make_state(data)]
        cells = []
        for target, threshold in sorted(data.thresholds.items()):
            expected = expected_cell(correctness, dataset, target)
            # Compile/warm each independent state before the retained sequence.
            warmup = [
                checked_output(p1.run_baseline(state, cublas, threshold), expected)
                for state in states
            ]
            if not all(row["pass"] for row in warmup):
                raise RuntimeError(f"H2B-P3 warmup mismatch: {dataset}/t{target}")
            observations = []
            signatures = set()
            for iteration in range(ITERATIONS_PER_CELL):
                slot = iteration & 1
                checked = checked_output(
                    p1.run_baseline(states[slot], cublas, threshold), expected
                )
                signature = (
                    checked["final_count"],
                    checked["direct_accept_count"],
                    checked["ambiguous_count"],
                    checked["overflow"],
                    checked["duplicate_ids"],
                    checked["ids_sha256"],
                )
                signatures.add(signature)
                observations.append(
                    {
                        "iteration": iteration,
                        "state_slot": slot,
                        **checked,
                    }
                )
                if not checked["pass"]:
                    raise RuntimeError(
                        f"H2B-P3 stress mismatch: {dataset}/t{target}/i{iteration}"
                    )
            cell_pass = len(signatures) == 1 and all(row["pass"] for row in observations)
            cells.append(
                {
                    "target_results_per_query": target,
                    "iterations": ITERATIONS_PER_CELL,
                    "independent_state_buffers": len(states),
                    "warmup": warmup,
                    "unique_signatures": len(signatures),
                    "observations": observations,
                    "pass": cell_pass,
                }
            )
            print(
                "P3_STRESS_CELL "
                + json.dumps(
                    {
                        "dataset": dataset,
                        "target": target,
                        "iterations": ITERATIONS_PER_CELL,
                        "unique_signatures": len(signatures),
                        "pass": cell_pass,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        datasets.append(
            {
                "dataset": dataset,
                "dimension": data.dimension,
                "cells": cells,
                "pass": all(row["pass"] for row in cells),
            }
        )
        del states
        torch.cuda.empty_cache()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "two_buffer_long_loop_stress",
        "environment": p1.environment_record(cublas),
        "iterations_per_cell": ITERATIONS_PER_CELL,
        "total_retained_invocations": ITERATIONS_PER_CELL * 2 * len(h1.DATASETS),
        "datasets": datasets,
        "stress_pass": all(row["pass"] for row in datasets),
        "elapsed_s": time.time() - started,
        "sources": {
            "runner_sha256": p1.sha256_file(Path(__file__)),
            "p1_runner_sha256": P1_RUNNER_SHA256,
            "p1_kernel_sha256": P1_KERNEL_SHA256,
            "p1_correctness_sha256": P1_CORRECTNESS_SHA256,
            "p2_audit_sha256": P2_AUDIT_SHA256,
        },
    }
    p1.atomic_json(STRESS_OUTPUT, result)
    print(
        "P3_STRESS_SUMMARY "
        + json.dumps(
            {
                "output": str(STRESS_OUTPUT),
                "total_retained_invocations": result["total_retained_invocations"],
                "stress_pass": result["stress_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["stress_pass"] else 3


def main() -> int:
    args = parse_args()
    correctness = require_dependencies()
    p1.require_dependencies()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("H2B-P3 requires NVIDIA_TF32_OVERRIDE=0 before CUDA init")
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    # P3-R2 is a pre-run-only device revision to an otherwise identical SM120
    # GPU after the P3-R1 preflight found physical GPU1 externally occupied.
    p1.PHYSICAL_GPU = PHYSICAL_GPU
    p1.require_environment()
    cublas = p1.CuBLAS()
    if args.phase == "safety":
        return run_safety(args.dataset, args.target, correctness, cublas)
    return run_stress(correctness, cublas)


if __name__ == "__main__":
    raise SystemExit(main())
