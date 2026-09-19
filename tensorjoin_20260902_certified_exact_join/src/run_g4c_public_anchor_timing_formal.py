#!/usr/bin/env python3
"""Formally time one G4C public N4096/k64 dynamic-router anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import triton

from g4b_r1_ragged_safe_kernel import analytic_certificate_ragged_safe_i64
from run_g2a_tensorjoin import refine_ambiguous_fp64_i64
from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    outward_float32,
)
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    certified_fp32_filter_i64,
)
from run_g4b_r1_public_opportunity import (
    DATASETS,
    quantize_with_analytic_residual,
)


PROJECT = Path(__file__).resolve().parents[1]
N = 4_096
TARGET_DEGREE = 64
FP64_BLOCK_K = 256
EXPECTED_OPPORTUNITY_RESULTS = {
    "sift128": "d987a0271d7945883328e0c95a3028d10276d1ed7bf8686016db4788cb743204",
    "cifar_gist512": "1bb98444527f64aba34a59df3b9ae96184b9477560c8b19471b1e2c25e00a09b",
    "fashion784": "4bf374d4b8e506f8c0fc0579c7f342acaf4852c99eb15f016fd8b1add2a90cbb",
}
EXPECTED_OPPORTUNITY_SUMMARY = (
    "4cc069e20b52412d267e5bb945f11362df7dbcf443fbbd5c7a96d4ddc097d1de"
)
EXPECTED_SCREEN_SUMMARY = (
    "3bc1c7aafb1488650eb68088c8b25b72085aeb308022e1e9fc63a49cca716037"
)
EXPECTED_SCREEN_RUNTIME_AUDIT = (
    "a19142fe8259cd63db343233835daf189a62a95be4e0ee73debd2e107bfe3542"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--order", choices=("KC", "CK"), required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--sustained-launches", type=int, required=True)
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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sorted_ids(tensor: torch.Tensor, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    return np.sort(tensor[:count].cpu().numpy().astype(np.uint64, copy=False))


def gpu_state() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pstate,temperature.gpu,power.draw,"
        "clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
        "--id=1",
    ]
    return subprocess.check_output(command, text=True).strip()


def time_callable(
    function: Callable[[], dict[str, int]],
    warmups: int,
    observations: int,
    variant: str,
    sequence_start: int,
) -> tuple[list[float], list[dict[str, object]]]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    ordered: list[dict[str, object]] = []
    for observation in range(observations):
        started_ns = time.perf_counter_ns()
        counts = function()
        elapsed_us = float((time.perf_counter_ns() - started_ns) / 1_000.0)
        samples.append(elapsed_us)
        ordered.append(
            {
                "sequence": sequence_start + observation,
                "observation": observation,
                "variant": variant,
                "elapsed_us": elapsed_us,
                "dynamic_stage_counts": counts,
            }
        )
    return samples, ordered


def time_sustained(
    function: Callable[[], dict[str, int]], launches: int
) -> dict[str, object]:
    observed: set[tuple[int, int, int]] = set()
    started_ns = time.perf_counter_ns()
    for _ in range(launches):
        counts = function()
        observed.add(
            (
                counts["g3b_ambiguous"],
                counts["fp64_refined"],
                counts["final_upper"],
            )
        )
    total_us = float((time.perf_counter_ns() - started_ns) / 1_000.0)
    return {
        "launches": launches,
        "total_us": total_us,
        "mean_us_per_pipeline": total_us / launches,
        "unique_dynamic_stage_counts": [
            {"g3b_ambiguous": a, "fp64_refined": f, "final_upper": o}
            for a, f, o in sorted(observed)
        ],
    }


def main() -> int:
    args = parse_args()
    if args.phase != "formal":
        raise RuntimeError("This immutable runner is formal-only")
    screen_summary_path = PROJECT / "results/g4c_screen_summary.json"
    screen_audit_path = PROJECT / "results/g4c_screen_runtime_audit.json"
    if sha256_file(screen_summary_path) != EXPECTED_SCREEN_SUMMARY:
        raise RuntimeError("G4C screen summary hash mismatch")
    if sha256_file(screen_audit_path) != EXPECTED_SCREEN_RUNTIME_AUDIT:
        raise RuntimeError("G4C screen runtime-audit hash mismatch")
    screen_summary = json.loads(screen_summary_path.read_text(encoding="utf-8"))
    screen_audit = json.loads(screen_audit_path.read_text(encoding="utf-8"))
    if not screen_summary.get("screen_gate_pass") or not screen_audit.get(
        "runtime_binary_gate_pass"
    ):
        raise RuntimeError("G4C screen has not admitted formal timing")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("G4C requires physical GPU1 as the sole visible device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    expected_settings = (10, 50, 200) if args.phase == "screen" else (20, 200, 1_000)
    supplied_settings = (
        args.warmups,
        args.observations,
        args.sustained_launches,
    )
    if supplied_settings != expected_settings:
        raise RuntimeError(
            f"Frozen {args.phase} settings are {expected_settings}, got {supplied_settings}"
        )
    process_range = range(2) if args.phase == "screen" else range(8)
    if args.process_id not in process_range:
        raise RuntimeError("Process ID is outside the frozen phase schedule")
    result_path = PROJECT / (
        f"results/g4c_{args.phase}_{args.dataset}_n4096_k64_"
        f"process_{args.process_id}_{args.order.lower()}_a{args.attempt}.json"
    )
    if result_path.exists():
        raise FileExistsError(result_path)

    summary_path = PROJECT / "results/g4b_r1_opportunity_summary.json"
    if sha256_file(summary_path) != EXPECTED_OPPORTUNITY_SUMMARY:
        raise RuntimeError("G4B-R1 opportunity summary hash mismatch")
    opportunity_path = PROJECT / f"results/g4b_r1_opportunity_{args.dataset}.json"
    if sha256_file(opportunity_path) != EXPECTED_OPPORTUNITY_RESULTS[args.dataset]:
        raise RuntimeError("G4B-R1 dataset opportunity hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    opportunity = json.loads(opportunity_path.read_text(encoding="utf-8"))
    if not summary.get("opportunity_gate_pass") or not opportunity.get(
        "process_gate_pass"
    ):
        raise RuntimeError("G4B-R1 opportunity gate has not admitted timing")
    cell = next(
        cell
        for cell in opportunity["cells"]
        if int(cell["n"]) == N
        and int(cell["target_average_directed_nonself_degree"]) == TARGET_DEGREE
    )
    if not cell.get("cell_gate_pass"):
        raise RuntimeError("Selected G4B-R1 opportunity cell did not pass")

    contract = DATASETS[args.dataset]
    dimension = int(contract["dimension"])
    dataset_root = PROJECT / "data/g4b_public" / args.dataset
    vector_path = dataset_root / "vectors_f32.npy"
    oracle_path = dataset_root / "n4096/k64/oracle_upper_ids_u64.npy"
    if sha256_file(vector_path) != contract["vector_file_sha256"]:
        raise RuntimeError("Prepared vector hash mismatch")
    if sha256_file(oracle_path) != cell["oracle_file_sha256"]:
        raise RuntimeError("Oracle file hash mismatch")
    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False)[:N], dtype=np.float32
    )
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if sha256_u64(oracle) != cell["oracle_upper_raw_u64_sha256"]:
        raise RuntimeError("Oracle raw hash mismatch")
    threshold_d2 = float(cell["threshold_d2"])
    epsilon = float(cell["epsilon"])
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    codes, scales, norms, errors = quantize_with_analytic_residual(vectors)
    tile_extent = (N + BLOCK_M - 1) // BLOCK_M
    tile_rows, tile_columns = np.triu_indices(tile_extent)
    tile_rows = np.asarray(tile_rows, dtype=np.int32)
    tile_columns = np.asarray(tile_columns, dtype=np.int32)
    scheduled_tiles = int(tile_rows.size)
    capacity = scheduled_tiles * BLOCK_M * BLOCK_N

    device = torch.device("cuda:0")
    vectors_gpu = torch.from_numpy(vectors).to(device)
    codes_gpu = torch.from_numpy(codes).to(device)
    codes_t_gpu = torch.from_numpy(codes.T.copy()).to(device)
    scales_gpu = torch.from_numpy(scales).to(device)
    norms_gpu = torch.from_numpy(norms).to(device)
    errors_gpu = torch.from_numpy(errors).to(device)
    tile_rows_gpu = torch.from_numpy(tile_rows).to(device)
    tile_columns_gpu = torch.from_numpy(tile_columns).to(device)
    keeper_result = torch.empty(capacity, dtype=torch.int64, device=device)
    keeper_ambiguous = torch.empty(capacity, dtype=torch.int64, device=device)
    keeper_counters = torch.zeros(6, dtype=torch.int32, device=device)
    candidate_result = torch.empty(capacity, dtype=torch.int64, device=device)
    candidate_ambiguous = torch.empty(capacity, dtype=torch.int64, device=device)
    candidate_fp64 = torch.empty(capacity, dtype=torch.int64, device=device)
    candidate_counters = torch.zeros(6, dtype=torch.int32, device=device)

    def launch_g3b(result_ids, ambiguous_ids, counters) -> None:
        analytic_certificate_ragged_safe_i64[(scheduled_tiles,)](
            codes_gpu,
            codes_t_gpu,
            scales_gpu,
            norms_gpu,
            errors_gpu,
            tile_rows_gpu,
            tile_columns_gpu,
            result_ids,
            ambiguous_ids,
            counters,
            float(epsilon_lower),
            float(epsilon_upper),
            EXPRESSION_RELATIVE_RADIUS,
            EXPRESSION_ABSOLUTE_RADIUS,
            SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
            SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
            M=N,
            N_=N,
            K=dimension,
            CAPACITY=capacity,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

    def keeper() -> dict[str, int]:
        keeper_counters.zero_()
        launch_g3b(keeper_result, keeper_ambiguous, keeper_counters)
        ambiguous_count = int(keeper_counters[1].item())
        if ambiguous_count:
            refine_ambiguous_fp64_i64[(ambiguous_count,)](
                vectors_gpu,
                vectors_gpu,
                keeper_ambiguous,
                keeper_result,
                keeper_counters,
                threshold_d2,
                N_=N,
                K=dimension,
                CAPACITY=capacity,
                BLOCK_K=FP64_BLOCK_K,
                num_warps=4,
            )
        final_count = int(keeper_counters[0].item())
        return {
            "g3b_ambiguous": ambiguous_count,
            "fp64_refined": ambiguous_count,
            "final_upper": final_count,
        }

    def candidate() -> dict[str, int]:
        candidate_counters.zero_()
        launch_g3b(candidate_result, candidate_ambiguous, candidate_counters)
        ambiguous_count = int(candidate_counters[1].item())
        if ambiguous_count:
            certified_fp32_filter_i64[(ambiguous_count,)](
                vectors_gpu,
                candidate_ambiguous,
                candidate_result,
                candidate_fp64,
                candidate_counters,
                float(threshold_lower),
                float(threshold_upper),
                FP32_DISTANCE_RELATIVE_RADIUS,
                FP32_INPUT_MAGNITUDE_RADIUS,
                FP32_FINAL_RELATIVE_RADIUS,
                FP32_ABSOLUTE_RADIUS,
                N_=N,
                K=dimension,
                CAPACITY=capacity,
                BLOCK_K=FP32_BLOCK_K,
                num_warps=4,
            )
        fp64_count = int(candidate_counters[3].item())
        if fp64_count:
            refine_ambiguous_fp64_i64[(fp64_count,)](
                vectors_gpu,
                vectors_gpu,
                candidate_fp64,
                candidate_result,
                candidate_counters,
                threshold_d2,
                N_=N,
                K=dimension,
                CAPACITY=capacity,
                BLOCK_K=FP64_BLOCK_K,
                num_warps=4,
            )
        final_count = int(candidate_counters[0].item())
        return {
            "g3b_ambiguous": ambiguous_count,
            "fp64_refined": fp64_count,
            "final_upper": final_count,
        }

    expected_final = int(cell["final_accepted_upper_pairs"])
    expected_g3b_ambiguous = int(cell["g3b_ambiguous_upper_pairs"])
    expected_fp64 = int(cell["fp64_refined_upper_pairs"])
    expected_fp32_reject = int(cell["fp32_direct_reject_upper_pairs"])
    expected_bitwise_equal = int(cell["fp32_bitwise_equal_upper_pairs"])

    def validate(function, result_ids, counters, variant: str) -> dict[str, object]:
        dynamic_counts = function()
        torch.cuda.synchronize()
        count_values = counters.cpu().numpy().astype(np.int64)
        final_count = int(count_values[0])
        values = sorted_ids(result_ids, min(final_count, capacity))
        expected_counts = (
            [expected_final, expected_g3b_ambiguous, 0, 0, 0, 0]
            if variant == "keeper"
            else [
                expected_final,
                expected_g3b_ambiguous,
                0,
                expected_fp64,
                expected_fp32_reject,
                expected_bitwise_equal,
            ]
        )
        return {
            "counts": count_values.tolist(),
            "expected_counts": expected_counts,
            "dynamic_stage_counts": dynamic_counts,
            "final_upper_count": final_count,
            "final_upper_sha256": sha256_u64(values),
            "pass": bool(
                count_values.tolist() == expected_counts
                and final_count == oracle.size
                and sha256_u64(values) == sha256_u64(oracle)
                and np.array_equal(values, oracle)
            ),
        }

    gpu_before = gpu_state()
    validation_before = {
        "keeper": validate(keeper, keeper_result, keeper_counters, "keeper"),
        "candidate": validate(
            candidate, candidate_result, candidate_counters, "candidate"
        ),
    }
    if not all(record["pass"] for record in validation_before.values()):
        raise RuntimeError(f"Pre-timing correctness failed: {validation_before}")

    functions = {"K": keeper, "C": candidate}
    names = {"K": "keeper_two_stage", "C": "candidate_three_stage"}
    metrics: dict[str, list[float]] = {}
    ordered_samples: list[dict[str, object]] = []
    sequence = 0
    for key in args.order:
        name = names[key]
        samples, ordered = time_callable(
            functions[key], args.warmups, args.observations, name, sequence
        )
        metrics[name] = samples
        ordered_samples.extend(ordered)
        sequence += args.observations
    sustained: dict[str, dict[str, object]] = {}
    for key in args.order:
        name = names[key]
        sustained[name] = time_sustained(
            functions[key], args.sustained_launches
        )

    validation_after = {
        "keeper": validate(keeper, keeper_result, keeper_counters, "keeper"),
        "candidate": validate(
            candidate, candidate_result, candidate_counters, "candidate"
        ),
    }
    correctness_pass = all(
        record["pass"]
        for phase in (validation_before, validation_after)
        for record in phase.values()
    )
    medians = {name: float(np.median(values)) for name, values in metrics.items()}
    percentiles = {
        name: {
            "p10_us": float(np.percentile(values, 10)),
            "median_us": float(np.median(values)),
            "p90_us": float(np.percentile(values, 90)),
        }
        for name, values in metrics.items()
    }
    median_speedup = medians["keeper_two_stage"] / medians["candidate_three_stage"]
    sustained_speedup = (
        float(sustained["keeper_two_stage"]["mean_us_per_pipeline"])
        / float(sustained["candidate_three_stage"]["mean_us_per_pipeline"])
    )
    gpu_after = gpu_state()
    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_timing",
        "phase": args.phase,
        "measurement_status": "screen_not_paper_claim" if args.phase == "screen" else "formal_candidate",
        "dataset_id": args.dataset,
        "n": N,
        "dimension": dimension,
        "target_average_directed_nonself_degree": TARGET_DEGREE,
        "process_id": args.process_id,
        "attempt": args.attempt,
        "order": args.order,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": properties.name,
        "gpu_uuid": gpu_before.split(",")[1].strip(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_state_before": gpu_before,
        "gpu_state_after": gpu_after,
        "clock_policy": "natural_dynamic_clocks_not_modified",
        "shape": [N, dimension],
        "dtype": "f32 source; int8/int32 G3B; f32 G3C; f64 exact refinement",
        "layout": "contiguous row-major vectors; transposed contiguous int8 codes",
        "quality_contract": "exact sorted upper-triangle u64 pair IDs",
        "timing_scope": (
            "host perf_counter from before device counter zero through the D2H "
            "final count; includes actual D2H dynamic-count reads, their stream "
            "synchronizations, dynamic GPU launches, and final completion"
        ),
        "scope_exclusions": [
            "host input ingest and quantization",
            "host-to-device resident-data setup",
            "device-to-host output copy and sort",
        ],
        "variants": {
            "keeper_two_stage": "ragged-safe G3B plus FP64 over all G3B ambiguity",
            "candidate_three_stage": (
                "identical ragged-safe G3B plus certified FP32 filter plus FP64 "
                "over measured residual ambiguity"
            ),
        },
        "threshold_d2": threshold_d2,
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
        "expected_dynamic_counts": {
            "g3b_ambiguous": expected_g3b_ambiguous,
            "fp64_candidate": expected_fp64,
            "final_upper": expected_final,
        },
        "warmups_per_variant": args.warmups,
        "retained_observations_per_variant": args.observations,
        "sustained_launches_per_variant": args.sustained_launches,
        "metrics_us": metrics,
        "ordered_samples": ordered_samples,
        "percentiles_us": percentiles,
        "medians_us": medians,
        "paired_process_median_speedup_keeper_over_candidate": median_speedup,
        "sustained": sustained,
        "sustained_speedup_keeper_over_candidate": sustained_speedup,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "correctness_pass": correctness_pass,
        "process_measurement_pass": bool(correctness_pass),
        "opportunity_result_sha256": sha256_file(opportunity_path),
        "opportunity_summary_sha256": sha256_file(summary_path),
        "screen_summary_sha256": sha256_file(screen_summary_path),
        "screen_runtime_audit_sha256": sha256_file(screen_audit_path),
        "ragged_safe_kernel_sha256": sha256_file(
            PROJECT / "src/g4b_r1_ragged_safe_kernel.py"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(
            PROJECT
            / (
                "PROTOCOL_G4C_SCREEN.md"
                if args.phase == "screen"
                else "PROTOCOL_G4C_FORMAL.md"
            )
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(result_path, record)
    print("RUN_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
    return 0 if correctness_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
