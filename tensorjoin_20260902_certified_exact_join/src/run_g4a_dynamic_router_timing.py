#!/usr/bin/env python3
"""Time frozen G3B-R1 and G3C-B-R1 host-dispatched dynamic-count pipelines."""

from __future__ import annotations

import argparse
import gc
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

from run_g2a_tensorjoin import refine_ambiguous_fp64_i64
from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    D,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    N,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    analytic_certificate_compact_i64,
    outward_float32,
    quantize_with_analytic_residual,
)
from run_g3c_b_r1_gpu_cascade import (
    FP32_ABSOLUTE_RADIUS,
    FP32_BLOCK_K,
    FP32_DISTANCE_RELATIVE_RADIUS,
    FP32_FINAL_RELATIVE_RADIUS,
    FP32_INPUT_MAGNITUDE_RADIUS,
    certified_fp32_filter_i64,
)


EXPERIMENT_ID = "tensorjoin_20260903_g4a_dynamic_count_router"
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/g2a_cifar4096"
FP64_BLOCK_K = 256

EXPECTED_VECTOR_SHA256 = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"
EXPECTED_ORACLE_FILE_SHA256 = "ac93d96ac39ea5a2bc2d1e81baa179a533a6344aae97f058b3b33a8155e6c0c9"
EXPECTED_ORACLE_RAW_SHA256 = "da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d"
EXPECTED_UPPER_RAW_SHA256 = "036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959"
EXPECTED_G3B_DIRECT = 90_946
EXPECTED_G3B_AMBIGUOUS = 102_079
EXPECTED_FP32_ACCEPT = 42_123
EXPECTED_FP32_REJECT = 59_859
EXPECTED_FP64_CASCADE = 97
EXPECTED_FINAL_UPPER = 133_120

EXPECTED_G2A_RUNNER_SHA256 = "84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8"
EXPECTED_G3B_R1_RUNNER_SHA256 = "057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec"
EXPECTED_G3C_B_R1_RUNNER_SHA256 = "637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d"
EXPECTED_G3C_B_R1_AUDIT_SHA256 = "8f3690a6f8b37c3d6e1bcb6ff1c61b3ac8a0504f7d0efda2dfd996598c2cb51a"
EXPECTED_G3C_B_R1_FINAL_SHA256 = "82aa22b30ca0380fcb807b3ffd4cef5e33456d17cb39cd784ba234c5c3547f8d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return hashlib.sha256(np.asarray(values, dtype="<u8").tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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
        "--query-gpu=index,uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
        "--id=0",
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
        stage_counts = function()
        elapsed_us = float((time.perf_counter_ns() - started_ns) / 1000.0)
        samples.append(elapsed_us)
        ordered.append(
            {
                "sequence": sequence_start + observation,
                "variant": variant,
                "observation": observation,
                "elapsed_us": elapsed_us,
                "dynamic_stage_counts": stage_counts,
            }
        )
    return samples, ordered


def time_sustained(
    function: Callable[[], dict[str, int]], launches: int
) -> dict[str, float | int | list[dict[str, int]]]:
    observed: set[tuple[int, int, int]] = set()
    started_ns = time.perf_counter_ns()
    for _ in range(launches):
        counts = function()
        observed.add(
            (counts["g3b_ambiguous"], counts["fp64_refined"], counts["final_upper"])
        )
    total_us = float((time.perf_counter_ns() - started_ns) / 1000.0)
    return {
        "launches": launches,
        "total_us": total_us,
        "mean_us_per_pipeline": total_us / launches,
        "unique_dynamic_stage_counts": [
            {
                "g3b_ambiguous": ambiguous,
                "fp64_refined": fp64,
                "final_upper": final,
            }
            for ambiguous, fp64, final in sorted(observed)
        ],
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("G4A requires physical GPU 0 as the sole visible device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    if args.phase == "screen":
        expected = (10, 50, 200)
    else:
        expected = (20, 200, 1_000)
    supplied = (args.warmups, args.observations, args.sustained_launches)
    if supplied != expected:
        raise RuntimeError(f"Frozen {args.phase} timing settings are {expected}, got {supplied}")
    if args.phase == "screen" and args.process_id not in range(2):
        raise RuntimeError("Screen process-id must be 0 or 1")
    if args.phase == "formal" and args.process_id not in range(8):
        raise RuntimeError("Formal process-id must be in [0, 7]")

    result_path = PROJECT / (
        f"results/g4a_{args.phase}_process_{args.process_id}_"
        f"{args.order.lower()}_a{args.attempt}.json"
    )
    if result_path.exists():
        raise FileExistsError(result_path)

    source_contract = {
        "g2a_runner": sha256_file(PROJECT / "src/run_g2a_tensorjoin.py"),
        "g3b_r1_runner": sha256_file(
            PROJECT / "src/run_g3b_r1_gpu_analytic_certificate.py"
        ),
        "g3c_b_r1_runner": sha256_file(PROJECT / "src/run_g3c_b_r1_gpu_cascade.py"),
        "g3c_b_r1_generated_code_audit": sha256_file(
            PROJECT / "results/g3c_b_r1_generated_code_audit.json"
        ),
        "g3c_b_r1_final_summary": sha256_file(
            PROJECT / "results/g3c_b_r1_final_summary.json"
        ),
    }
    expected_source_contract = {
        "g2a_runner": EXPECTED_G2A_RUNNER_SHA256,
        "g3b_r1_runner": EXPECTED_G3B_R1_RUNNER_SHA256,
        "g3c_b_r1_runner": EXPECTED_G3C_B_R1_RUNNER_SHA256,
        "g3c_b_r1_generated_code_audit": EXPECTED_G3C_B_R1_AUDIT_SHA256,
        "g3c_b_r1_final_summary": EXPECTED_G3C_B_R1_FINAL_SHA256,
    }
    if source_contract != expected_source_contract:
        raise RuntimeError("Accepted G3B-R1/G3C-B-R1 source or evidence changed")

    vector_path = DATA / "vectors_f32.npy"
    oracle_path = DATA / "oracle_pairs_u64.npy"
    metadata_path = DATA / "metadata.json"
    if sha256_file(vector_path) != EXPECTED_VECTOR_SHA256:
        raise RuntimeError("Vector hash mismatch")
    if sha256_file(oracle_path) != EXPECTED_ORACLE_FILE_SHA256:
        raise RuntimeError("Oracle file hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = np.ascontiguousarray(
        np.load(vector_path, allow_pickle=False), dtype=np.float32
    )
    oracle = np.asarray(np.load(oracle_path, allow_pickle=False), dtype=np.uint64)
    if vectors.shape != (N, D) or sha256_u64(oracle) != EXPECTED_ORACLE_RAW_SHA256:
        raise RuntimeError("Frozen input/oracle contract mismatch")
    threshold_d2 = float(metadata["radius"]["effective_epsilon_d2"])
    epsilon = float(metadata["radius"]["epsilon"])
    epsilon_lower, epsilon_upper = outward_float32(epsilon)
    threshold_lower, threshold_upper = outward_float32(threshold_d2)
    codes, scales, reconstructed_norm2, residual_error = (
        quantize_with_analytic_residual(vectors)
    )

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
    norms_gpu = torch.from_numpy(reconstructed_norm2).to(device)
    errors_gpu = torch.from_numpy(residual_error).to(device)
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
        analytic_certificate_compact_i64[(scheduled_tiles,)](
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
            K=D,
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
                K=D,
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
                K=D,
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
                K=D,
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

    def validate(
        function: Callable[[], dict[str, int]], result_ids, counters, variant: str
    ):
        dynamic_counts = function()
        torch.cuda.synchronize()
        counts = counters.cpu().numpy().astype(np.int64)
        final_count = int(counts[0])
        values = sorted_ids(result_ids, min(final_count, capacity))
        upper_hash = sha256_u64(values)
        expected_counts = (
            [EXPECTED_FINAL_UPPER, EXPECTED_G3B_AMBIGUOUS, 0, 0, 0, 0]
            if variant == "keeper"
            else [
                EXPECTED_FINAL_UPPER,
                EXPECTED_G3B_AMBIGUOUS,
                0,
                EXPECTED_FP64_CASCADE,
                EXPECTED_FP32_REJECT,
                0,
            ]
        )
        return {
            "counts": counts.tolist(),
            "expected_counts": expected_counts,
            "dynamic_stage_counts": dynamic_counts,
            "final_upper_count": final_count,
            "final_upper_sha256": upper_hash,
            "pass": bool(
                counts.tolist() == expected_counts
                and final_count == EXPECTED_FINAL_UPPER
                and upper_hash == EXPECTED_UPPER_RAW_SHA256
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
        print(
            f"TIMING variant={name} median_us={np.median(samples):.6f} "
            f"p10_us={np.percentile(samples, 10):.6f} "
            f"p90_us={np.percentile(samples, 90):.6f}",
            flush=True,
        )

    sustained: dict[str, dict[str, float | int]] = {}
    for key in args.order:
        name = names[key]
        sustained[name] = time_sustained(functions[key], args.sustained_launches)
        print(
            f"SUSTAINED variant={name} "
            f"mean_us={sustained[name]['mean_us_per_pipeline']:.6f}",
            flush=True,
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
        "experiment_id": EXPERIMENT_ID,
        "phase": args.phase,
        "measurement_status": "host_dispatched_dynamic_count_resident_gpu_operator",
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
        "shape": [N, D],
        "dtype": "f32 source; int8/int32 G3B certificate; f32 G3C filter; f64 exact refinement",
        "layout": "contiguous row-major vectors; transposed contiguous int8 codes",
        "quality_contract": "exact sorted upper-triangle u64 pair IDs",
        "timing_scope": (
            "host perf_counter from before device counter zero through the D2H final "
            "count; includes actual D2H ambiguity/FP64 scalar reads, their stream "
            "synchronizations, dynamic GPU launches, and final completion"
        ),
        "scope_exclusions": [
            "host input ingest and preprocessing",
            "device-to-host output copy and sort",
        ],
        "dynamic_launch_contract": {
            "keeper": "read actual G3B ambiguity count, launch FP64, read final count",
            "candidate": (
                "read actual G3B ambiguity count, launch FP32, read actual FP64 "
                "count, launch FP64, read final count"
            ),
            "scalar_transport": "torch Tensor.item D2H on the default stream",
        },
        "variants": {
            "keeper_two_stage": (
                "G3B-R1 analytic certificate plus FP64 refinement of all 102079 "
                "validated ambiguous pairs"
            ),
            "candidate_three_stage": (
                "identical G3B-R1 certificate plus certified G3C-B-R1 FP32 filter "
                "plus FP64 refinement of 97 validated residual pairs"
            ),
        },
        "threshold_d2": threshold_d2,
        "scheduled_tiles": scheduled_tiles,
        "buffer_capacity": capacity,
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
        "process_measurement_pass": bool(
            correctness_pass and median_speedup > 1.0 and sustained_speedup > 1.0
        ),
        "source_contract": source_contract,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4A.md"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(result_path, record)
    print("RUN_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)

    del (
        vectors_gpu,
        codes_gpu,
        codes_t_gpu,
        scales_gpu,
        norms_gpu,
        errors_gpu,
        tile_rows_gpu,
        tile_columns_gpu,
        keeper_result,
        keeper_ambiguous,
        keeper_counters,
        candidate_result,
        candidate_ambiguous,
        candidate_fp64,
        candidate_counters,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return 0 if correctness_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
