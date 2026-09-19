#!/usr/bin/env python3
"""Run the frozen H2A certified-FP32 strong-baseline kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import triton

import run_h1_multivector_gpu_screen as h1
from h1_multivector_kernels import (
    aggregate_exact_panels_i64,
    exact_ambiguous_panels_fp64,
)
from h2a_fp32_multivector_kernels import (
    aggregate_d2_interval_objects_i64,
    token_d2_interval_fp32_rect,
)


EXPERIMENT_ID = "tensorjoin_20260903_h2a_certified_fp32_strong_baseline"
PROJECT = Path(__file__).resolve().parents[1]
CORRECTNESS_OUTPUT = PROJECT / "results/h2a_fp32_strong_correctness.json"
SAFETY_OUTPUT = PROJECT / "results/h2a_fp32_strong_memcheck_summary.json"
TIMING_OUTPUT = PROJECT / "results/h2a_fp32_strong_timing_screen.json"
H1_CORRECTNESS_SHA256 = "06461064d1b237aaa29f4face0eb3c223f5487abf478ddcf5fccb353ed28dca4"
H1_SAFETY_SHA256 = "f022cc90db19b295dd481a5b7946f3690b0cd935ba398befb31dab8218305d09"
H1_TIMING_SHA256 = "c98d0e33c95ff1707caee5a1b37cd345fbbb1dabe239644a1f82582b8c1f3173"
H1_RUNNER_SHA256 = "11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e"
H1_KERNEL_SHA256 = "c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db"
FP32_BLOCK_M = 8
FP32_BLOCK_N = 8
FP32_BLOCK_K = 32
FP32_DISTANCE_RELATIVE_RADIUS = 2.0**-14
FP32_INPUT_MAGNITUDE_RADIUS = 2.0**-22
FP32_FINAL_RELATIVE_RADIUS = 2.0**-22
FP32_ABSOLUTE_RADIUS = 4096.0 * float(np.finfo(np.float32).tiny)
OUTWARD_ABSOLUTE_GUARD = 1e-10
CONTAINMENT_GUARD = 2e-10
WARMUPS = 10
OBSERVATIONS = 50
MIN_SPEEDUP = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("correctness", "memcheck", "timing"), required=True
    )
    parser.add_argument("--dataset", choices=tuple(h1.DATASETS))
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_u64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u8").tobytes(order="C")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_dependencies() -> None:
    h1.require_environment()
    dependencies = {
        PROJECT / "results/h1_r1_multivector_correctness.json": H1_CORRECTNESS_SHA256,
        PROJECT / "results/h1_r1_multivector_memcheck_summary.json": H1_SAFETY_SHA256,
        PROJECT / "results/h1_r1_multivector_timing_screen.json": H1_TIMING_SHA256,
        PROJECT / "src/run_h1_multivector_gpu_screen.py": H1_RUNNER_SHA256,
        PROJECT / "src/h1_multivector_kernels.py": H1_KERNEL_SHA256,
    }
    for path, expected in dependencies.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"H1 dependency hash mismatch: {path}")


def make_state(data: h1.PreparedDataset) -> h1.DeviceState:
    state = h1.make_device_state(data)
    state.tensors["fp32_lower_d2"] = torch.empty(
        state.query_tokens * state.base_tokens,
        device="cuda",
        dtype=torch.float32,
    )
    state.tensors["fp32_upper_d2"] = torch.empty_like(
        state.tensors["fp32_lower_d2"]
    )
    torch.cuda.synchronize()
    return state


def launch_fp32_intervals(state: h1.DeviceState) -> None:
    t = state.tensors
    token_d2_interval_fp32_rect[
        (
            triton.cdiv(state.query_tokens, FP32_BLOCK_M),
            triton.cdiv(state.base_tokens, FP32_BLOCK_N),
        )
    ](
        t["query"],
        t["base"],
        t["fp32_lower_d2"],
        t["fp32_upper_d2"],
        FP32_DISTANCE_RELATIVE_RADIUS,
        FP32_INPUT_MAGNITUDE_RADIUS,
        FP32_FINAL_RELATIVE_RADIUS,
        FP32_ABSOLUTE_RADIUS,
        M=state.query_tokens,
        N_=state.base_tokens,
        K=state.data.dimension,
        BLOCK_M=FP32_BLOCK_M,
        BLOCK_N=FP32_BLOCK_N,
        BLOCK_K=FP32_BLOCK_K,
        num_warps=4,
    )


def run_fp32_baseline(state: h1.DeviceState, threshold: float) -> dict[str, object]:
    t = state.tensors
    base_objects = len(state.data.base.slices)
    t["threshold"].fill_(threshold)
    state.start_event.record()
    t["counters"].zero_()
    launch_fp32_intervals(state)
    aggregate_d2_interval_objects_i64[(state.capacity,)](
        t["fp32_lower_d2"],
        t["fp32_upper_d2"],
        t["query_starts"],
        t["query_counts"],
        t["base_starts"],
        t["base_counts"],
        t["threshold"],
        t["object_lower"],
        t["object_upper"],
        t["result_ids"],
        t["ambiguous_ids"],
        t["counters"],
        OUTWARD_ABSOLUTE_GUARD,
        BASE_OBJECTS=base_objects,
        BASE_TOKENS=state.base_tokens,
        CAPACITY=state.capacity,
        MAX_TOKENS=h1.MAX_TOKENS,
        num_warps=1,
    )
    first_counts = t["counters"].cpu().numpy().astype(np.int64)
    direct_count = int(first_counts[0])
    ambiguous_count = int(first_counts[1])
    overflow = int(first_counts[2])
    if max(direct_count, ambiguous_count) > state.capacity or overflow:
        raise RuntimeError(f"FP32 compaction overflow: {first_counts.tolist()}")
    if ambiguous_count:
        exact_ambiguous_panels_fp64[(ambiguous_count * h1.MAX_TOKENS**2,)](
            t["query"],
            t["base"],
            t["ambiguous_ids"],
            t["query_starts"],
            t["query_counts"],
            t["base_starts"],
            t["base_counts"],
            t["exact_panels"],
            AMBIGUOUS_COUNT=ambiguous_count,
            BASE_OBJECTS=base_objects,
            K=state.data.dimension,
            MAX_TOKENS=h1.MAX_TOKENS,
            BLOCK_K=h1.PANEL_BLOCK_K,
            num_warps=4,
        )
        aggregate_exact_panels_i64[(ambiguous_count,)](
            t["exact_panels"],
            t["ambiguous_ids"],
            t["query_counts"],
            t["base_counts"],
            t["threshold"],
            t["refined_scores"],
            t["result_ids"],
            t["counters"],
            BASE_OBJECTS=base_objects,
            CAPACITY=state.capacity,
            MAX_TOKENS=h1.MAX_TOKENS,
            num_warps=1,
        )
    state.end_event.record()
    state.end_event.synchronize()
    final_counts = t["counters"].cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    if final_count > state.capacity or int(final_counts[2]):
        raise RuntimeError(f"FP32 final overflow: {final_counts.tolist()}")
    ids = np.sort(
        t["result_ids"][:final_count].cpu().numpy().astype(np.uint64, copy=False)
    )
    ambiguous_ids = np.sort(
        t["ambiguous_ids"][:ambiguous_count]
        .cpu()
        .numpy()
        .astype(np.uint64, copy=False)
    )
    return {
        "ids": ids,
        "ambiguous_ids": ambiguous_ids,
        "direct_accept_count": direct_count,
        "ambiguous_count": ambiguous_count,
        "final_count": final_count,
        "overflow": int(final_counts[2]),
        "cuda_ms": float(state.start_event.elapsed_time(state.end_event)),
    }


def duplicate_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] == values[:-1]))


def correctness_dataset(data: h1.PreparedDataset) -> dict[str, object]:
    started = time.time()
    print(f"CPU_ORACLE_START dataset={data.name}", flush=True)
    cpu_tokens, cpu_objects = h1.cpu_exact(data)
    print(f"CPU_ORACLE_DONE dataset={data.name} elapsed_s={time.time()-started:.6f}", flush=True)
    state = make_state(data)
    rows = []
    token_lower_violations = token_upper_violations = None
    for target, threshold in sorted(data.thresholds.items()):
        baseline = run_fp32_baseline(state, threshold)
        lower_tokens = (
            state.tensors["fp32_lower_d2"].cpu().numpy().reshape(cpu_tokens.shape)
        )
        upper_tokens = (
            state.tensors["fp32_upper_d2"].cpu().numpy().reshape(cpu_tokens.shape)
        )
        object_lower = (
            state.tensors["object_lower"].cpu().numpy().reshape(cpu_objects.shape)
        )
        object_upper = (
            state.tensors["object_upper"].cpu().numpy().reshape(cpu_objects.shape)
        )
        if token_lower_violations is None:
            token_lower_violations = int(
                np.count_nonzero(
                    cpu_tokens + CONTAINMENT_GUARD
                    < lower_tokens.astype(np.float64) - OUTWARD_ABSOLUTE_GUARD
                )
            )
            token_upper_violations = int(
                np.count_nonzero(
                    cpu_tokens - CONTAINMENT_GUARD
                    > upper_tokens.astype(np.float64) + OUTWARD_ABSOLUTE_GUARD
                )
            )
        candidate = h1.run_candidate(state, threshold)
        oracle_ids = np.flatnonzero(cpu_objects.reshape(-1) <= threshold).astype(
            np.uint64
        )
        baseline_ids = np.asarray(baseline["ids"], dtype=np.uint64)
        candidate_ids = np.asarray(candidate["ids"], dtype=np.uint64)
        exact_inside = cpu_objects <= threshold
        direct_accept = object_upper <= threshold
        direct_reject = object_lower > threshold
        ambiguous_ids = np.asarray(baseline["ambiguous_ids"], dtype=np.uint64)
        valid_refinement_cells = int(
            sum(
                int(data.query.token_counts[int(pair) // len(data.base.slices)])
                * int(data.base.token_counts[int(pair) % len(data.base.slices)])
                for pair in ambiguous_ids
            )
        )
        row = {
            "target_results_per_query": target,
            "threshold": threshold,
            "oracle_count": int(len(oracle_ids)),
            "oracle_ids_sha256": sha256_u64(oracle_ids),
            "baseline_count": int(baseline["final_count"]),
            "baseline_ids_sha256": sha256_u64(baseline_ids),
            "candidate_count": int(candidate["final_count"]),
            "candidate_ids_sha256": sha256_u64(candidate_ids),
            "baseline_direct_accepts": int(baseline["direct_accept_count"]),
            "baseline_ambiguous_objects": int(baseline["ambiguous_count"]),
            "baseline_ambiguous_fraction": float(
                int(baseline["ambiguous_count"]) / state.capacity
            ),
            "baseline_valid_refinement_cells": valid_refinement_cells,
            "baseline_global_refinement_fraction": float(
                valid_refinement_cells / cpu_tokens.size
            ),
            "token_lower_containment_violations": int(token_lower_violations),
            "token_upper_containment_violations": int(token_upper_violations),
            "object_lower_containment_violations": int(
                np.count_nonzero(cpu_objects + CONTAINMENT_GUARD < object_lower)
            ),
            "object_upper_containment_violations": int(
                np.count_nonzero(cpu_objects - CONTAINMENT_GUARD > object_upper)
            ),
            "unsafe_direct_accepts": int(
                np.count_nonzero(direct_accept & ~exact_inside)
            ),
            "unsafe_direct_rejects": int(
                np.count_nonzero(direct_reject & exact_inside)
            ),
            "baseline_mismatches": int(
                len(np.setxor1d(baseline_ids, oracle_ids, assume_unique=True))
            ),
            "candidate_mismatches": int(
                len(np.setxor1d(candidate_ids, oracle_ids, assume_unique=True))
            ),
            "candidate_baseline_mismatches": int(
                len(np.setxor1d(candidate_ids, baseline_ids, assume_unique=True))
            ),
            "baseline_duplicate_ids": duplicate_count(baseline_ids),
            "candidate_duplicate_ids": duplicate_count(candidate_ids),
            "baseline_overflow": int(baseline["overflow"]),
            "candidate_overflow": int(candidate["overflow"]),
            "nonfinite_object_bounds": int(
                np.count_nonzero(~np.isfinite(object_lower))
                + np.count_nonzero(~np.isfinite(object_upper))
            ),
        }
        zero_keys = (
            "token_lower_containment_violations",
            "token_upper_containment_violations",
            "object_lower_containment_violations",
            "object_upper_containment_violations",
            "unsafe_direct_accepts",
            "unsafe_direct_rejects",
            "baseline_mismatches",
            "candidate_mismatches",
            "candidate_baseline_mismatches",
            "baseline_duplicate_ids",
            "candidate_duplicate_ids",
            "baseline_overflow",
            "candidate_overflow",
            "nonfinite_object_bounds",
        )
        row["correctness_pass"] = all(int(row[key]) == 0 for key in zero_keys)
        rows.append(row)
        print("CORRECTNESS_CELL " + json.dumps({"dataset": data.name, **row}, sort_keys=True), flush=True)
    return {
        "dataset": data.name,
        "dimension": data.dimension,
        "query_tokens": len(data.query.vectors),
        "base_tokens": len(data.base.vectors),
        "cells": rows,
        "correctness_pass": all(bool(row["correctness_pass"]) for row in rows),
        "elapsed_s": time.time() - started,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
    }


def timed(function: Callable[[], dict[str, object]]) -> tuple[dict[str, object], float]:
    started = time.perf_counter_ns()
    result = function()
    return result, float((time.perf_counter_ns() - started) / 1_000.0)


def timing_cell(
    state: h1.DeviceState, target: int, threshold: float, expected_hash: str
) -> dict[str, object]:
    functions = {
        "baseline": lambda: run_fp32_baseline(state, threshold),
        "candidate": lambda: h1.run_candidate(state, threshold),
    }
    for _ in range(WARMUPS):
        functions["baseline"]()
        functions["candidate"]()
    wall = {"baseline": [], "candidate": []}
    cuda = {"baseline": [], "candidate": []}
    samples = []
    for observation in range(OBSERVATIONS):
        order = ("baseline", "candidate") if observation % 2 == 0 else ("candidate", "baseline")
        observed = {}
        for position, variant in enumerate(order):
            output, wall_us = timed(functions[variant])
            ids_hash = sha256_u64(np.asarray(output["ids"], dtype=np.uint64))
            if ids_hash != expected_hash:
                raise RuntimeError(f"Timed output mismatch: {state.data.name}/{target}/{variant}")
            wall[variant].append(wall_us)
            cuda[variant].append(float(output["cuda_ms"]) * 1_000.0)
            observed[variant] = wall_us
            samples.append(
                {
                    "sequence": len(samples),
                    "observation": observation,
                    "order": "BC" if observation % 2 == 0 else "CB",
                    "position": position,
                    "variant": variant,
                    "wall_us": wall_us,
                    "cuda_event_us": float(output["cuda_ms"]) * 1_000.0,
                    "final_count": int(output["final_count"]),
                    "ambiguous_count": int(output["ambiguous_count"]),
                    "ids_sha256": ids_hash,
                }
            )
        samples[-1]["paired_candidate_win"] = observed["candidate"] < observed["baseline"]
    baseline_summary = summarize(wall["baseline"])
    candidate_summary = summarize(wall["candidate"])
    speedup = baseline_summary["median"] / candidate_summary["median"]
    paired_wins = sum(
        candidate < baseline
        for candidate, baseline in zip(wall["candidate"], wall["baseline"])
    )
    return {
        "target_results_per_query": target,
        "threshold": threshold,
        "expected_ids_sha256": expected_hash,
        "warmups_per_variant": WARMUPS,
        "observations_per_variant": OBSERVATIONS,
        "wall_us": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
        },
        "cuda_event_us": {
            "baseline": summarize(cuda["baseline"]),
            "candidate": summarize(cuda["candidate"]),
        },
        "median_speedup": float(speedup),
        "paired_candidate_wins": int(paired_wins),
        "samples": samples,
        "performance_pass": bool(
            speedup >= MIN_SPEEDUP and paired_wins == OBSERVATIONS
        ),
    }


def source_record() -> dict[str, str]:
    return {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "h2a_kernel_sha256": sha256_file(PROJECT / "src/h2a_fp32_multivector_kernels.py"),
        "h1_runner_sha256": sha256_file(PROJECT / "src/run_h1_multivector_gpu_screen.py"),
        "h1_kernel_sha256": sha256_file(PROJECT / "src/h1_multivector_kernels.py"),
    }


def run_correctness() -> int:
    if CORRECTNESS_OUTPUT.exists():
        raise FileExistsError(CORRECTNESS_OUTPUT)
    started = time.time()
    datasets = [correctness_dataset(h1.prepare_dataset(name)) for name in h1.DATASETS]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "correctness",
        "host": platform.node(),
        "physical_gpu": h1.PHYSICAL_GPU,
        "sources": source_record(),
        "constants": {
            "distance_relative_radius": FP32_DISTANCE_RELATIVE_RADIUS,
            "input_magnitude_radius": FP32_INPUT_MAGNITUDE_RADIUS,
            "final_relative_radius": FP32_FINAL_RELATIVE_RADIUS,
            "absolute_radius": FP32_ABSOLUTE_RADIUS,
            "outward_absolute_guard": OUTWARD_ABSOLUTE_GUARD,
        },
        "datasets": datasets,
        "correctness_pass": all(bool(row["correctness_pass"]) for row in datasets),
        "elapsed_s": time.time() - started,
    }
    atomic_json(CORRECTNESS_OUTPUT, result)
    print("CORRECTNESS_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["correctness_pass"] else 2


def run_memcheck(dataset: str) -> int:
    data = h1.prepare_memcheck_subset(h1.prepare_dataset(dataset))
    state = make_state(data)
    output = run_fp32_baseline(state, data.thresholds[8])
    print(
        "MEMCHECK_SMOKE "
        + json.dumps(
            {
                "dataset": dataset,
                "dimension": data.dimension,
                "query_counts": data.query.token_counts.tolist(),
                "base_counts": data.base.token_counts.tolist(),
                "final_count": int(output["final_count"]),
                "ids_sha256": sha256_u64(np.asarray(output["ids"], dtype=np.uint64)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def run_timing() -> int:
    if TIMING_OUTPUT.exists():
        raise FileExistsError(TIMING_OUTPUT)
    if not CORRECTNESS_OUTPUT.is_file() or not SAFETY_OUTPUT.is_file():
        raise RuntimeError("H2A correctness and safety results are required")
    correctness = json.loads(CORRECTNESS_OUTPUT.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY_OUTPUT.read_text(encoding="utf-8"))
    current_sources = source_record()
    if not correctness.get("correctness_pass") or not safety.get("safety_pass"):
        raise RuntimeError("H2A prerequisite gate failed")
    if correctness.get("sources") != current_sources:
        raise RuntimeError("H2A correctness/source drift")
    if safety.get("sources") != current_sources:
        raise RuntimeError("H2A safety/source drift")
    started = time.time()
    datasets = []
    for name in h1.DATASETS:
        data = h1.prepare_dataset(name)
        state = make_state(data)
        correctness_row = next(
            row for row in correctness["datasets"] if row["dataset"] == name
        )
        cells = []
        for target, threshold in sorted(data.thresholds.items()):
            c_row = next(
                row for row in correctness_row["cells"]
                if int(row["target_results_per_query"]) == target
            )
            cell = timing_cell(
                state, target, threshold, str(c_row["oracle_ids_sha256"])
            )
            cells.append(cell)
            print("TIMING_CELL " + json.dumps({"dataset": name, "target": target, "median_speedup": cell["median_speedup"], "paired_candidate_wins": cell["paired_candidate_wins"], "performance_pass": cell["performance_pass"]}, sort_keys=True), flush=True)
        datasets.append(
            {
                "dataset": name,
                "dimension": data.dimension,
                "cells": cells,
                "performance_pass": all(bool(cell["performance_pass"]) for cell in cells),
            }
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "timing_screen",
        "host": platform.node(),
        "physical_gpu": h1.PHYSICAL_GPU,
        "sources": current_sources,
        "correctness_sha256": sha256_file(CORRECTNESS_OUTPUT),
        "safety_sha256": sha256_file(SAFETY_OUTPUT),
        "timing_scope": "resident outer wall including dynamic count and canonical final IDs",
        "warmups_per_variant": WARMUPS,
        "observations_per_variant": OBSERVATIONS,
        "minimum_speedup": MIN_SPEEDUP,
        "datasets": datasets,
        "screen_pass": all(bool(row["performance_pass"]) for row in datasets),
        "elapsed_s": time.time() - started,
    }
    atomic_json(TIMING_OUTPUT, result)
    print("TIMING_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["screen_pass"] else 2


def main() -> int:
    args = parse_args()
    require_dependencies()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": args.phase,
                "dataset": args.dataset,
                "host": platform.node(),
                "sources": source_record(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.phase == "correctness":
        if args.dataset:
            raise RuntimeError("Correctness runs all frozen cells")
        return run_correctness()
    if args.phase == "timing":
        if args.dataset:
            raise RuntimeError("Timing runs all frozen cells")
        return run_timing()
    if not args.dataset:
        raise RuntimeError("Memcheck requires --dataset")
    return run_memcheck(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())

