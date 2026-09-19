#!/usr/bin/env python3
"""Run the frozen H1 exact GPU multi-vector join correctness/timing screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import triton

from h1_multivector_kernels import (
    aggregate_exact_matrix_i64,
    aggregate_exact_panels_i64,
    aggregate_interval_objects_i64,
    exact_ambiguous_panels_fp64,
    exact_token_distances_fp64,
    token_l2_interval_rect_i64,
)
from run_g3b_r1_gpu_analytic_certificate import (
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
)
from run_g4b_r1_public_opportunity import quantize_with_analytic_residual
from run_h0_multivector_certificate import (
    BASE_OBJECTS,
    QUERY_OBJECTS,
    SEED,
    ObjectBatch,
    deterministic_split,
    direct_squared_distances,
    normalize_once,
    symmetric_chamfer,
)


EXPERIMENT_ID = "tensorjoin_20260903_h1_r1_exact_gpu_multivector_screen"
PROJECT = Path(__file__).resolve().parents[1]
PHYSICAL_GPU = 3
H0_RESULT = PROJECT / "results/h0_multivector_certificate.json"
H0_RESULT_SHA256 = "7d8b8e68c3ecaa6bb38de133e83afefcab4ddefa4b375572ba09fe3e9a68af31"
CORRECTNESS_OUTPUT = PROJECT / "results/h1_r1_multivector_correctness.json"
TIMING_OUTPUT = PROJECT / "results/h1_r1_multivector_timing_screen.json"
SAFETY_OUTPUT = PROJECT / "results/h1_r1_multivector_memcheck_summary.json"
MAX_TOKENS = 8
TOKEN_BLOCK_M = 64
TOKEN_BLOCK_N = 64
TOKEN_BLOCK_K = 64
EXACT_BLOCK_M = 4
EXACT_BLOCK_N = 4
EXACT_BLOCK_K = 64
PANEL_BLOCK_K = 256
SQUARE_ABSOLUTE_GUARD = 1e-10
CONTAINMENT_GUARD = 2e-10
WARMUPS = 10
OBSERVATIONS = 50
MIN_MEDIAN_SPEEDUP = 1.25

DATASETS = {
    "esc50_panns": {
        "cache": PROJECT / "data/esc50_panns_cnn14_1s_2048d.npz",
        "id_key": "clip",
        "dimension": 2048,
    },
    "ucf101_r3d18": {
        "cache": PROJECT / "data/ucf101_r3d18_512d.npz",
        "id_key": "group",
        "dimension": 512,
    },
}


@dataclass
class PreparedDataset:
    name: str
    dimension: int
    query: ObjectBatch
    base: ObjectBatch
    thresholds: dict[int, float]
    cache: Path
    cache_sha256: str


@dataclass
class DeviceState:
    data: PreparedDataset
    tensors: dict[str, torch.Tensor]
    capacity: int
    query_tokens: int
    base_tokens: int
    max_exact_panels: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("correctness", "timing", "memcheck"),
        required=True,
    )
    parser.add_argument("--variant", choices=("candidate", "keeper"))
    parser.add_argument("--dataset", choices=tuple(DATASETS))
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


def environment_record() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    gpu_state = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pstate,temperature.gpu,power.draw,"
            "clocks.sm,clocks.mem,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--id={PHYSICAL_GPU}",
        ],
        text=True,
    ).strip()
    return {
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "physical_gpu": PHYSICAL_GPU,
        "nvidia_smi_target_gpu": gpu_state,
    }


def require_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(
            f"H1-R1 requires physical GPU{PHYSICAL_GPU} as the sole visible device"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("H1 expected exactly one visible CUDA device")
    if torch.cuda.get_device_properties(0).major != 12:
        raise RuntimeError("H1 target is the frozen compute-capability 12.x GPU")
    if sha256_file(H0_RESULT) != H0_RESULT_SHA256:
        raise RuntimeError("Immutable H0 result hash mismatch")
    physical_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            f"--id={PHYSICAL_GPU}",
        ],
        text=True,
    ).strip()
    process_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    foreign = []
    for row in process_rows:
        fields = [value.strip() for value in row.split(",", 3)]
        if len(fields) != 4 or fields[0] != physical_uuid:
            continue
        try:
            process_id = int(fields[1])
        except ValueError:
            foreign.append(row)
            continue
        if process_id != os.getpid():
            foreign.append(row)
    if foreign:
        raise RuntimeError(
            f"Refusing to share physical GPU{PHYSICAL_GPU} with foreign compute processes: "
            + " | ".join(foreign)
        )


def object_starts(batch: ObjectBatch) -> np.ndarray:
    return np.asarray([value.start for value in batch.slices], dtype=np.int32)


def prepare_dataset(name: str) -> PreparedDataset:
    contract = DATASETS[name]
    cache = Path(contract["cache"])
    with np.load(cache, allow_pickle=False) as data:
        vectors = np.ascontiguousarray(data["features"], dtype=np.float32)
        ids = np.asarray(data[str(contract["id_key"])])
    dimension = int(contract["dimension"])
    if vectors.ndim != 2 or vectors.shape[1] != dimension or len(ids) != len(vectors):
        raise RuntimeError((name, vectors.shape, ids.shape))
    vectors = normalize_once(vectors)
    query, base = deterministic_split(vectors, ids, np.random.default_rng(SEED))
    if max(int(query.token_counts.max()), int(base.token_counts.max())) > MAX_TOKENS:
        raise RuntimeError("H1 refuses objects with more than eight tokens")
    if min(int(query.token_counts.min()), int(base.token_counts.min())) <= 0:
        raise RuntimeError("H1 refuses empty objects")
    h0 = json.loads(H0_RESULT.read_text(encoding="utf-8"))
    h0_dataset = next(row for row in h0["datasets"] if row["dataset"] == name)
    if len(query.vectors) != int(h0_dataset["query_token_count"]):
        raise RuntimeError("H0/H1 query-token split mismatch")
    if len(base.vectors) != int(h0_dataset["base_token_count"]):
        raise RuntimeError("H0/H1 base-token split mismatch")
    thresholds = {
        int(row["target_results_per_query"]): float(row["threshold"])
        for row in h0_dataset["thresholds"]
    }
    return PreparedDataset(
        name=name,
        dimension=dimension,
        query=query,
        base=base,
        thresholds=thresholds,
        cache=cache,
        cache_sha256=sha256_file(cache),
    )


def truncate_batch(batch: ObjectBatch, object_count: int) -> ObjectBatch:
    chunks = [batch.vectors[value] for value in batch.slices[:object_count]]
    counts = batch.token_counts[:object_count].copy()
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    slices = tuple(
        slice(int(start), int(start + count)) for start, count in zip(starts, counts)
    )
    return ObjectBatch(
        vectors=np.ascontiguousarray(np.concatenate(chunks, axis=0)),
        object_ids=batch.object_ids[:object_count].copy(),
        slices=slices,
        token_counts=np.ascontiguousarray(counts, dtype=np.int32),
    )


def prepare_memcheck_subset(data: PreparedDataset) -> PreparedDataset:
    query = truncate_batch(data.query, 4)
    base = truncate_batch(data.base, 8)
    return PreparedDataset(
        name=data.name,
        dimension=data.dimension,
        query=query,
        base=base,
        thresholds=data.thresholds,
        cache=data.cache,
        cache_sha256=data.cache_sha256,
    )


def cpu_exact(data: PreparedDataset) -> tuple[np.ndarray, np.ndarray]:
    token_scores = direct_squared_distances(data.query.vectors, data.base.vectors)
    object_scores = np.empty(
        (len(data.query.slices), len(data.base.slices)), dtype=np.float64
    )
    for qi, query_slice in enumerate(data.query.slices):
        for bi, base_slice in enumerate(data.base.slices):
            object_scores[qi, bi] = symmetric_chamfer(
                token_scores[query_slice, base_slice]
            )
    return token_scores, object_scores


def to_cuda(values: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cuda()


def make_device_state(data: PreparedDataset) -> DeviceState:
    query_codes, query_scales, query_norms, query_errors = (
        quantize_with_analytic_residual(data.query.vectors)
    )
    base_codes, base_scales, base_norms, base_errors = (
        quantize_with_analytic_residual(data.base.vectors)
    )
    query_tokens = len(data.query.vectors)
    base_tokens = len(data.base.vectors)
    query_objects = len(data.query.slices)
    base_objects = len(data.base.slices)
    capacity = query_objects * base_objects
    tensors = {
        "query": to_cuda(data.query.vectors),
        "base": to_cuda(data.base.vectors),
        "query_codes": to_cuda(query_codes),
        "base_codes_t": to_cuda(np.ascontiguousarray(base_codes.T)),
        "query_scales": to_cuda(query_scales),
        "base_scales": to_cuda(base_scales),
        "query_norms": to_cuda(query_norms),
        "base_norms": to_cuda(base_norms),
        "query_errors": to_cuda(query_errors),
        "base_errors": to_cuda(base_errors),
        "query_starts": to_cuda(object_starts(data.query)),
        "base_starts": to_cuda(object_starts(data.base)),
        "query_counts": to_cuda(data.query.token_counts),
        "base_counts": to_cuda(data.base.token_counts),
        "lower_l2": torch.empty(
            query_tokens * base_tokens, device="cuda", dtype=torch.float32
        ),
        "upper_l2": torch.empty(
            query_tokens * base_tokens, device="cuda", dtype=torch.float32
        ),
        "object_lower": torch.empty(capacity, device="cuda", dtype=torch.float64),
        "object_upper": torch.empty(capacity, device="cuda", dtype=torch.float64),
        "object_scores": torch.empty(capacity, device="cuda", dtype=torch.float64),
        "refined_scores": torch.empty(capacity, device="cuda", dtype=torch.float64),
        "exact_tokens": torch.empty(
            query_tokens * base_tokens, device="cuda", dtype=torch.float64
        ),
        "exact_panels": torch.empty(
            capacity * MAX_TOKENS * MAX_TOKENS,
            device="cuda",
            dtype=torch.float64,
        ),
        "result_ids": torch.empty(capacity, device="cuda", dtype=torch.int64),
        "ambiguous_ids": torch.empty(capacity, device="cuda", dtype=torch.int64),
        "counters": torch.zeros(3, device="cuda", dtype=torch.int32),
        "threshold": torch.empty(1, device="cuda", dtype=torch.float64),
    }
    torch.cuda.synchronize()
    return DeviceState(
        data=data,
        tensors=tensors,
        capacity=capacity,
        query_tokens=query_tokens,
        base_tokens=base_tokens,
        max_exact_panels=capacity,
        start_event=torch.cuda.Event(enable_timing=True),
        end_event=torch.cuda.Event(enable_timing=True),
    )


def launch_token_intervals(state: DeviceState) -> None:
    t = state.tensors
    token_l2_interval_rect_i64[
        (
            triton.cdiv(state.query_tokens, TOKEN_BLOCK_M),
            triton.cdiv(state.base_tokens, TOKEN_BLOCK_N),
        )
    ](
        t["query_codes"],
        t["base_codes_t"],
        t["query_scales"],
        t["base_scales"],
        t["query_norms"],
        t["base_norms"],
        t["query_errors"],
        t["base_errors"],
        t["lower_l2"],
        t["upper_l2"],
        EXPRESSION_RELATIVE_RADIUS,
        EXPRESSION_ABSOLUTE_RADIUS,
        SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
        SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
        M=state.query_tokens,
        N_=state.base_tokens,
        K=state.data.dimension,
        BLOCK_M=TOKEN_BLOCK_M,
        BLOCK_N=TOKEN_BLOCK_N,
        BLOCK_K=TOKEN_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )


def run_candidate(state: DeviceState, threshold: float) -> dict[str, object]:
    t = state.tensors
    query_objects = len(state.data.query.slices)
    base_objects = len(state.data.base.slices)
    t["threshold"].fill_(threshold)
    state.start_event.record()
    t["counters"].zero_()
    launch_token_intervals(state)
    aggregate_interval_objects_i64[(state.capacity,)](
        t["lower_l2"],
        t["upper_l2"],
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
        SQUARE_ABSOLUTE_GUARD,
        QUERY_OBJECTS=query_objects,
        BASE_OBJECTS=base_objects,
        BASE_TOKENS=state.base_tokens,
        CAPACITY=state.capacity,
        MAX_TOKENS=MAX_TOKENS,
        num_warps=1,
    )
    first_counts = t["counters"].cpu().numpy().astype(np.int64)
    direct_count = int(first_counts[0])
    ambiguous_count = int(first_counts[1])
    overflow = int(first_counts[2])
    if max(direct_count, ambiguous_count) > state.capacity or overflow:
        raise RuntimeError(f"Candidate compaction overflow: {first_counts.tolist()}")
    if ambiguous_count:
        exact_ambiguous_panels_fp64[(ambiguous_count * MAX_TOKENS * MAX_TOKENS,)](
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
            MAX_TOKENS=MAX_TOKENS,
            BLOCK_K=PANEL_BLOCK_K,
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
            MAX_TOKENS=MAX_TOKENS,
            num_warps=1,
        )
    state.end_event.record()
    state.end_event.synchronize()
    final_counts = t["counters"].cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    if final_count > state.capacity or int(final_counts[2]):
        raise RuntimeError(f"Candidate final overflow: {final_counts.tolist()}")
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


def run_keeper(state: DeviceState, threshold: float) -> dict[str, object]:
    t = state.tensors
    base_objects = len(state.data.base.slices)
    t["threshold"].fill_(threshold)
    state.start_event.record()
    t["counters"].zero_()
    exact_token_distances_fp64[
        (
            triton.cdiv(state.query_tokens, EXACT_BLOCK_M),
            triton.cdiv(state.base_tokens, EXACT_BLOCK_N),
        )
    ](
        t["query"],
        t["base"],
        t["exact_tokens"],
        M=state.query_tokens,
        N_=state.base_tokens,
        K=state.data.dimension,
        BLOCK_M=EXACT_BLOCK_M,
        BLOCK_N=EXACT_BLOCK_N,
        BLOCK_K=EXACT_BLOCK_K,
        num_warps=4,
    )
    aggregate_exact_matrix_i64[(state.capacity,)](
        t["exact_tokens"],
        t["query_starts"],
        t["query_counts"],
        t["base_starts"],
        t["base_counts"],
        t["threshold"],
        t["object_scores"],
        t["result_ids"],
        t["counters"],
        BASE_OBJECTS=base_objects,
        BASE_TOKENS=state.base_tokens,
        CAPACITY=state.capacity,
        MAX_TOKENS=MAX_TOKENS,
        num_warps=1,
    )
    state.end_event.record()
    state.end_event.synchronize()
    counts = t["counters"].cpu().numpy().astype(np.int64)
    final_count = int(counts[0])
    if final_count > state.capacity or int(counts[2]):
        raise RuntimeError(f"Keeper overflow: {counts.tolist()}")
    ids = np.sort(
        t["result_ids"][:final_count].cpu().numpy().astype(np.uint64, copy=False)
    )
    return {
        "ids": ids,
        "final_count": final_count,
        "overflow": int(counts[2]),
        "cuda_ms": float(state.start_event.elapsed_time(state.end_event)),
    }


def duplicate_count(ids: np.ndarray) -> int:
    return int(np.count_nonzero(ids[1:] == ids[:-1]))


def correctness_dataset(data: PreparedDataset) -> dict[str, object]:
    started = time.time()
    print(f"CPU_ORACLE_START dataset={data.name}", flush=True)
    cpu_tokens, cpu_objects = cpu_exact(data)
    print(
        f"CPU_ORACLE_DONE dataset={data.name} elapsed_s={time.time()-started:.6f}",
        flush=True,
    )
    state = make_device_state(data)
    rows: list[dict[str, object]] = []
    token_audit_done = False
    token_lower_violations = -1
    token_upper_violations = -1
    for target, threshold in sorted(data.thresholds.items()):
        keeper = run_keeper(state, threshold)
        candidate = run_candidate(state, threshold)
        keeper_ids = np.asarray(keeper["ids"], dtype=np.uint64)
        candidate_ids = np.asarray(candidate["ids"], dtype=np.uint64)
        oracle_ids = np.flatnonzero(cpu_objects.reshape(-1) <= threshold).astype(
            np.uint64
        )
        if not token_audit_done:
            lower_l2 = (
                state.tensors["lower_l2"]
                .cpu()
                .numpy()
                .reshape(cpu_tokens.shape)
                .astype(np.float64)
            )
            upper_l2 = (
                state.tensors["upper_l2"]
                .cpu()
                .numpy()
                .reshape(cpu_tokens.shape)
                .astype(np.float64)
            )
            lower_token_d2 = np.maximum(
                lower_l2 * lower_l2 - SQUARE_ABSOLUTE_GUARD, 0.0
            )
            upper_token_d2 = upper_l2 * upper_l2 + SQUARE_ABSOLUTE_GUARD
            token_lower_violations = int(
                np.count_nonzero(cpu_tokens + CONTAINMENT_GUARD < lower_token_d2)
            )
            token_upper_violations = int(
                np.count_nonzero(cpu_tokens - CONTAINMENT_GUARD > upper_token_d2)
            )
            token_audit_done = True
        object_lower = (
            state.tensors["object_lower"].cpu().numpy().reshape(cpu_objects.shape)
        )
        object_upper = (
            state.tensors["object_upper"].cpu().numpy().reshape(cpu_objects.shape)
        )
        keeper_scores = (
            state.tensors["object_scores"].cpu().numpy().reshape(cpu_objects.shape)
        )
        direct_accept = object_upper <= threshold
        direct_reject = object_lower > threshold
        exact_inside = cpu_objects <= threshold
        ambiguous_ids = np.asarray(candidate["ambiguous_ids"], dtype=np.uint64)
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
            "keeper_count": int(keeper["final_count"]),
            "keeper_ids_sha256": sha256_u64(keeper_ids),
            "candidate_count": int(candidate["final_count"]),
            "candidate_ids_sha256": sha256_u64(candidate_ids),
            "direct_accept_object_pairs": int(candidate["direct_accept_count"]),
            "ambiguous_object_pairs": int(candidate["ambiguous_count"]),
            "ambiguous_object_fraction": float(
                int(candidate["ambiguous_count"]) / state.capacity
            ),
            "padded_refinement_token_cells": int(
                int(candidate["ambiguous_count"]) * MAX_TOKENS * MAX_TOKENS
            ),
            "valid_refinement_token_cells": valid_refinement_cells,
            "valid_global_refinement_fraction": float(
                valid_refinement_cells / cpu_tokens.size
            ),
            "token_lower_containment_violations": token_lower_violations,
            "token_upper_containment_violations": token_upper_violations,
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
            "keeper_mismatches": int(
                len(np.setxor1d(keeper_ids, oracle_ids, assume_unique=True))
            ),
            "candidate_mismatches": int(
                len(np.setxor1d(candidate_ids, oracle_ids, assume_unique=True))
            ),
            "candidate_keeper_mismatches": int(
                len(np.setxor1d(candidate_ids, keeper_ids, assume_unique=True))
            ),
            "keeper_duplicate_ids": duplicate_count(keeper_ids),
            "candidate_duplicate_ids": duplicate_count(candidate_ids),
            "keeper_overflow": int(keeper["overflow"]),
            "candidate_overflow": int(candidate["overflow"]),
            "max_keeper_cpu_object_score_abs_error": float(
                np.max(np.abs(keeper_scores - cpu_objects))
            ),
            "nonfinite_valid_scores": int(
                np.count_nonzero(~np.isfinite(keeper_scores))
                + np.count_nonzero(~np.isfinite(object_lower))
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
            "keeper_mismatches",
            "candidate_mismatches",
            "candidate_keeper_mismatches",
            "keeper_duplicate_ids",
            "candidate_duplicate_ids",
            "keeper_overflow",
            "candidate_overflow",
            "nonfinite_valid_scores",
        )
        row["correctness_pass"] = all(int(row[key]) == 0 for key in zero_keys)
        rows.append(row)
        print("CORRECTNESS_CELL " + json.dumps({"dataset": data.name, **row}, sort_keys=True), flush=True)
    return {
        "dataset": data.name,
        "dimension": data.dimension,
        "cache": str(data.cache),
        "cache_sha256": data.cache_sha256,
        "query_objects": len(data.query.slices),
        "base_objects": len(data.base.slices),
        "query_tokens": len(data.query.vectors),
        "base_tokens": len(data.base.vectors),
        "query_token_count_range": [
            int(data.query.token_counts.min()),
            int(data.query.token_counts.max()),
        ],
        "base_token_count_range": [
            int(data.base.token_counts.min()),
            int(data.base.token_counts.max()),
        ],
        "cells": rows,
        "correctness_pass": all(bool(row["correctness_pass"]) for row in rows),
        "elapsed_s": time.time() - started,
    }


def summarize_samples(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
    }


def timed_call(function: Callable[[], dict[str, object]]) -> tuple[dict[str, object], float]:
    started_ns = time.perf_counter_ns()
    output = function()
    wall_us = (time.perf_counter_ns() - started_ns) / 1_000.0
    return output, float(wall_us)


def timing_cell(
    state: DeviceState,
    target: int,
    threshold: float,
    expected_sha256: str,
) -> dict[str, object]:
    keeper_fn = lambda: run_keeper(state, threshold)
    candidate_fn = lambda: run_candidate(state, threshold)
    for _ in range(WARMUPS):
        keeper_fn()
        candidate_fn()
    samples: list[dict[str, object]] = []
    wall: dict[str, list[float]] = {"keeper": [], "candidate": []}
    cuda: dict[str, list[float]] = {"keeper": [], "candidate": []}
    output_hashes: dict[str, set[str]] = {"keeper": set(), "candidate": set()}
    for observation in range(OBSERVATIONS):
        order = ("keeper", "candidate") if observation % 2 == 0 else ("candidate", "keeper")
        for position, variant in enumerate(order):
            function = keeper_fn if variant == "keeper" else candidate_fn
            output, wall_us = timed_call(function)
            ids_hash = sha256_u64(np.asarray(output["ids"], dtype=np.uint64))
            if ids_hash != expected_sha256:
                raise RuntimeError(
                    f"Timed {variant} output mismatch for {state.data.name}/t{target}"
                )
            wall[variant].append(wall_us)
            cuda[variant].append(float(output["cuda_ms"]) * 1_000.0)
            output_hashes[variant].add(ids_hash)
            samples.append(
                {
                    "sequence": len(samples),
                    "observation": observation,
                    "order": "KC" if observation % 2 == 0 else "CK",
                    "position": position,
                    "variant": variant,
                    "wall_us": wall_us,
                    "cuda_event_us": float(output["cuda_ms"]) * 1_000.0,
                    "final_count": int(output["final_count"]),
                    "ambiguous_count": (
                        int(output["ambiguous_count"])
                        if variant == "candidate"
                        else None
                    ),
                    "ids_sha256": ids_hash,
                }
            )
    keeper_summary = summarize_samples(wall["keeper"])
    candidate_summary = summarize_samples(wall["candidate"])
    speedup = keeper_summary["median"] / candidate_summary["median"]
    return {
        "target_results_per_query": target,
        "threshold": threshold,
        "expected_ids_sha256": expected_sha256,
        "warmups_per_variant": WARMUPS,
        "observations_per_variant": OBSERVATIONS,
        "wall_us": {
            "keeper": keeper_summary,
            "candidate": candidate_summary,
        },
        "cuda_event_us": {
            "keeper": summarize_samples(cuda["keeper"]),
            "candidate": summarize_samples(cuda["candidate"]),
        },
        "median_speedup": float(speedup),
        "candidate_process_wins": int(
            sum(c < k for c, k in zip(wall["candidate"], wall["keeper"]))
        ),
        "unique_output_hashes": {
            key: sorted(value) for key, value in output_hashes.items()
        },
        "samples": samples,
        "performance_pass": bool(speedup >= MIN_MEDIAN_SPEEDUP),
    }


def run_correctness() -> int:
    if CORRECTNESS_OUTPUT.exists():
        raise FileExistsError(CORRECTNESS_OUTPUT)
    started = time.time()
    datasets = [correctness_dataset(prepare_dataset(name)) for name in DATASETS]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "correctness",
        "environment": environment_record(),
        "h0_result": str(H0_RESULT),
        "h0_result_sha256": sha256_file(H0_RESULT),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "kernel_source": str((PROJECT / "src/h1_multivector_kernels.py").resolve()),
        "kernel_source_sha256": sha256_file(PROJECT / "src/h1_multivector_kernels.py"),
        "square_absolute_guard": SQUARE_ABSOLUTE_GUARD,
        "containment_guard": CONTAINMENT_GUARD,
        "datasets": datasets,
        "correctness_pass": all(bool(row["correctness_pass"]) for row in datasets),
        "elapsed_s": time.time() - started,
    }
    atomic_json(CORRECTNESS_OUTPUT, result)
    print("CORRECTNESS_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["correctness_pass"] else 2


def run_memcheck(dataset: str, variant: str) -> int:
    data = prepare_memcheck_subset(prepare_dataset(dataset))
    state = make_device_state(data)
    threshold = data.thresholds[8]
    output = run_candidate(state, threshold) if variant == "candidate" else run_keeper(state, threshold)
    print(
        "MEMCHECK_SMOKE "
        + json.dumps(
            {
                "dataset": dataset,
                "variant": variant,
                "dimension": data.dimension,
                "query_objects": len(data.query.slices),
                "base_objects": len(data.base.slices),
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
    if not CORRECTNESS_OUTPUT.is_file():
        raise RuntimeError("Correctness result is required before timing")
    correctness = json.loads(CORRECTNESS_OUTPUT.read_text(encoding="utf-8"))
    if not correctness.get("correctness_pass"):
        raise RuntimeError("Correctness gate did not pass")
    current_script_sha256 = sha256_file(Path(__file__).resolve())
    current_kernel_sha256 = sha256_file(PROJECT / "src/h1_multivector_kernels.py")
    if correctness.get("script_sha256") != current_script_sha256:
        raise RuntimeError("Correctness runner/source drift before timing")
    if correctness.get("kernel_source_sha256") != current_kernel_sha256:
        raise RuntimeError("Correctness kernel/source drift before timing")
    if not SAFETY_OUTPUT.is_file():
        raise RuntimeError("Memcheck summary is required before timing")
    safety = json.loads(SAFETY_OUTPUT.read_text(encoding="utf-8"))
    if not safety.get("safety_pass"):
        raise RuntimeError("Memcheck safety gate did not pass")
    if safety.get("runner_sha256") != current_script_sha256:
        raise RuntimeError("Memchecked runner/source drift before timing")
    if safety.get("kernel_sha256") != current_kernel_sha256:
        raise RuntimeError("Memchecked kernel/source drift before timing")
    started = time.time()
    dataset_results: list[dict[str, object]] = []
    for name in DATASETS:
        data = prepare_dataset(name)
        state = make_device_state(data)
        correctness_dataset_row = next(
            row for row in correctness["datasets"] if row["dataset"] == name
        )
        cells: list[dict[str, object]] = []
        for target, threshold in sorted(data.thresholds.items()):
            correctness_cell = next(
                row
                for row in correctness_dataset_row["cells"]
                if int(row["target_results_per_query"]) == target
            )
            cell = timing_cell(
                state,
                target,
                threshold,
                str(correctness_cell["oracle_ids_sha256"]),
            )
            cells.append(cell)
            print(
                "TIMING_CELL "
                + json.dumps(
                    {
                        "dataset": name,
                        "target": target,
                        "median_speedup": cell["median_speedup"],
                        "performance_pass": cell["performance_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dataset_results.append(
            {
                "dataset": name,
                "dimension": data.dimension,
                "query_tokens": len(data.query.vectors),
                "base_tokens": len(data.base.vectors),
                "cells": cells,
                "performance_pass": all(bool(row["performance_pass"]) for row in cells),
            }
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "timing_screen",
        "environment": environment_record(),
        "correctness_result": str(CORRECTNESS_OUTPUT),
        "correctness_result_sha256": sha256_file(CORRECTNESS_OUTPUT),
        "safety_result": str(SAFETY_OUTPUT),
        "safety_result_sha256": sha256_file(SAFETY_OUTPUT),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "kernel_source_sha256": sha256_file(PROJECT / "src/h1_multivector_kernels.py"),
        "timing_scope": "resident_gpu_outer_operator_wall_including_dynamic_count_and_final_ids",
        "warmups_per_variant": WARMUPS,
        "observations_per_variant": OBSERVATIONS,
        "minimum_median_speedup": MIN_MEDIAN_SPEEDUP,
        "datasets": dataset_results,
        "screen_pass": all(bool(row["performance_pass"]) for row in dataset_results),
        "elapsed_s": time.time() - started,
    }
    atomic_json(TIMING_OUTPUT, result)
    print("TIMING_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["screen_pass"] else 2


def main() -> int:
    args = parse_args()
    require_environment()
    print(
        "RUN_START "
        + json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": args.phase,
                "variant": args.variant,
                "dataset": args.dataset,
                "host": platform.node(),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "kernel_sha256": sha256_file(PROJECT / "src/h1_multivector_kernels.py"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.phase == "correctness":
        if args.variant or args.dataset:
            raise RuntimeError("Correctness phase runs all frozen cells")
        return run_correctness()
    if args.phase == "timing":
        if args.variant or args.dataset:
            raise RuntimeError("Timing phase runs all frozen cells")
        return run_timing()
    if not args.variant or not args.dataset:
        raise RuntimeError("Memcheck phase requires --variant and --dataset")
    return run_memcheck(args.dataset, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
