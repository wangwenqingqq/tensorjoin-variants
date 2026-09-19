#!/usr/bin/env python3
"""Run H2B-P1 direct-cuBLAS exactness and repeatability gates."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton

import run_h1_multivector_gpu_screen as h1
from h1_multivector_kernels import (
    aggregate_exact_panels_i64,
    exact_ambiguous_panels_fp64,
)
from h2b_pedantic_kernels import (
    aggregate_pedantic_dot_interval_objects_i64,
    dot_to_d2_interval_fp64_rect,
)


EXPERIMENT_ID = "tensorjoin_20260903_h2b_p1_pedantic_correctness"
PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results/h2b_p1_pedantic_correctness.json"
H1_RUNNER_SHA256 = "11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e"
H1_KERNEL_SHA256 = "c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db"
H2A_CORRECTNESS_SHA256 = "d773c3cf08578af1b4845d5dc3bdddced4138f167659e24c7bf25eb3a3b5c9be"
H2A_SAFETY_SHA256 = "d10f7dab881175cb742325104d1f93479a0c76aacf5f107a5768505b42e39da4"
H2A_TIMING_SHA256 = "a97efa258375b3bc87949fc6a0efaf15fb863958dc8d099aa6f1344952cfdcb4"
H2B_P0_SHA256 = "7b514ab72359320dd51384d9a791e3467be6f02417932ce0125fe0e0128c21e4"
INTERVAL_BLOCK = 256
FP64_RECONSTRUCTION_FACTOR = 32.0
ABSOLUTE_RECONSTRUCTION_GUARD = 1e-12
CONTAINMENT_GUARD = 2e-10
PHYSICAL_GPU = 1

CUBLAS_STATUS_SUCCESS = 0
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_POINTER_MODE_HOST = 0
CUBLAS_PEDANTIC_MATH = 2
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F_PEDANTIC = 69
CUBLAS_GEMM_DEFAULT = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("correctness", "repeat"), required=True)
    parser.add_argument("--slot", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.phase == "repeat" and args.slot is None:
        parser.error("--phase repeat requires --slot")
    if args.phase == "correctness" and args.slot is not None:
        parser.error("--slot is valid only for repeat")
    return args


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
    expected = {
        PROJECT / "src/run_h1_multivector_gpu_screen.py": H1_RUNNER_SHA256,
        PROJECT / "src/h1_multivector_kernels.py": H1_KERNEL_SHA256,
        PROJECT / "results/h2a_fp32_strong_correctness.json": H2A_CORRECTNESS_SHA256,
        PROJECT / "results/h2a_fp32_strong_memcheck_summary.json": H2A_SAFETY_SHA256,
        PROJECT / "results/h2a_fp32_strong_timing_screen.json": H2A_TIMING_SHA256,
        PROJECT / "results/h2b_p0_pedantic_opportunity.json": H2B_P0_SHA256,
        h1.H0_RESULT: h1.H0_RESULT_SHA256,
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"immutable dependency mismatch: {path}")


def require_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(
            f"H2B-P1-R1 requires physical GPU{PHYSICAL_GPU} as the sole visible device"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("H2B-P1 expected exactly one visible CUDA device")
    if torch.cuda.get_device_properties(0).major != 12:
        raise RuntimeError("H2B-P1 target is the frozen compute-capability 12.x GPU")
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


class CuBLAS:
    """Minimal typed binding for the frozen pedantic FP32 GEMM call."""

    def __init__(self) -> None:
        candidates = [
            str(
                Path(torch.__file__).resolve().parents[1]
                / "nvidia/cu13/lib/libcublas.so.13"
            ),
            "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13",
        ]
        found = ctypes.util.find_library("cublas")
        if found:
            candidates.append(found)
        error = None
        for candidate in candidates:
            try:
                self.library = ctypes.CDLL(candidate)
                candidate_path = Path(candidate)
                if not candidate_path.is_file():
                    raise OSError(f"loaded cuBLAS path is not hashable: {candidate}")
                self.library_path = candidate_path.resolve()
                break
            except OSError as caught:
                error = caught
        else:
            raise RuntimeError(f"unable to load libcublas: {error}")

        handle_type = ctypes.c_void_p
        self.library.cublasGetVersion_v2.argtypes = [handle_type, ctypes.POINTER(ctypes.c_int)]
        self.library.cublasGetVersion_v2.restype = ctypes.c_int
        self.library.cublasSetMathMode.argtypes = [handle_type, ctypes.c_int]
        self.library.cublasSetMathMode.restype = ctypes.c_int
        self.library.cublasGetMathMode.argtypes = [handle_type, ctypes.POINTER(ctypes.c_int)]
        self.library.cublasGetMathMode.restype = ctypes.c_int
        self.library.cublasSetPointerMode_v2.argtypes = [handle_type, ctypes.c_int]
        self.library.cublasSetPointerMode_v2.restype = ctypes.c_int
        self.library.cublasGetPointerMode_v2.argtypes = [
            handle_type,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.cublasGetPointerMode_v2.restype = ctypes.c_int
        self.library.cublasGemmEx.argtypes = [
            handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.cublasGemmEx.restype = ctypes.c_int

        self.handle_value = int(torch.cuda.current_blas_handle())
        self.handle = ctypes.c_void_p(self.handle_value)
        self._check(
            self.library.cublasSetMathMode(self.handle, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode",
        )
        self._check(
            self.library.cublasSetPointerMode_v2(
                self.handle, CUBLAS_POINTER_MODE_HOST
            ),
            "cublasSetPointerMode_v2",
        )
        version = ctypes.c_int()
        math_mode = ctypes.c_int()
        pointer_mode = ctypes.c_int()
        self._check(
            self.library.cublasGetVersion_v2(self.handle, ctypes.byref(version)),
            "cublasGetVersion_v2",
        )
        self._check(
            self.library.cublasGetMathMode(self.handle, ctypes.byref(math_mode)),
            "cublasGetMathMode",
        )
        self._check(
            self.library.cublasGetPointerMode_v2(
                self.handle, ctypes.byref(pointer_mode)
            ),
            "cublasGetPointerMode_v2",
        )
        if math_mode.value != CUBLAS_PEDANTIC_MATH:
            raise RuntimeError(f"cuBLAS math mode is not pedantic: {math_mode.value}")
        if pointer_mode.value != CUBLAS_POINTER_MODE_HOST:
            raise RuntimeError(f"cuBLAS pointer mode is not host: {pointer_mode.value}")
        self.version = version.value
        self.math_mode = math_mode.value
        self.pointer_mode = pointer_mode.value

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{operation} failed with cuBLAS status {status}")

    def gemm(self, query: torch.Tensor, base: torch.Tensor, output: torch.Tensor) -> None:
        if query.dtype != torch.float32 or base.dtype != torch.float32:
            raise TypeError("pedantic GEMM requires FP32 input")
        if not query.is_contiguous() or not base.is_contiguous() or not output.is_contiguous():
            raise RuntimeError("pedantic GEMM requires contiguous tensors")
        m, k = map(int, query.shape)
        n, base_k = map(int, base.shape)
        if base_k != k or output.numel() != m * n:
            raise RuntimeError((query.shape, base.shape, output.shape))
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # Row-major output [M,N] shares memory with column-major C^T [N,M].
        # B(row N,K) is column-major B^T(K,N), hence op(B^T)=B via transa=T.
        self._check(
            self.library.cublasGemmEx(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                ctypes.byref(alpha),
                ctypes.c_void_p(base.data_ptr()),
                CUDA_R_32F,
                k,
                ctypes.c_void_p(query.data_ptr()),
                CUDA_R_32F,
                k,
                ctypes.byref(beta),
                ctypes.c_void_p(output.data_ptr()),
                CUDA_R_32F,
                n,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        )

    def record(self) -> dict[str, object]:
        return {
            "library": str(self.library_path),
            "library_sha256": sha256_file(self.library_path),
            "version": self.version,
            "handle": self.handle_value,
            "math_mode": self.math_mode,
            "required_math_mode": CUBLAS_PEDANTIC_MATH,
            "pointer_mode": self.pointer_mode,
            "compute_type": CUBLAS_COMPUTE_32F_PEDANTIC,
            "algorithm": CUBLAS_GEMM_DEFAULT,
            "input_output_type": CUDA_R_32F,
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        }


@dataclass
class PedanticState:
    h1_state: h1.DeviceState
    gamma_dot: float
    dot_absolute_radius: float
    fp64_relative_guard: float


def norm_metadata(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = vectors.astype(np.float64)
    norm2_center = np.einsum("ij,ij->i", values, values)
    operations = values.shape[1]
    eps64 = np.finfo(np.float64).eps
    gamma64 = (operations * eps64) / (1.0 - operations * eps64)
    norm2_radius = (
        gamma64 * norm2_center
        + 2.0 * operations * float(np.finfo(np.float64).tiny)
    )
    l2_upper = np.sqrt(norm2_center + norm2_radius)
    l2_upper = np.nextafter(l2_upper * (1.0 + 8.0 * eps64), np.inf)
    return tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (norm2_center, norm2_radius, l2_upper)
    )


def make_state(data: h1.PreparedDataset) -> PedanticState:
    state = h1.make_device_state(data)
    qn, qnr, ql2 = norm_metadata(data.query.vectors)
    bn, bnr, bl2 = norm_metadata(data.base.vectors)
    pairs = state.query_tokens * state.base_tokens
    state.tensors.update(
        {
            "pedantic_dots": torch.empty(pairs, device="cuda", dtype=torch.float32),
            "pedantic_lower_d2": torch.empty(pairs, device="cuda", dtype=torch.float64),
            "pedantic_upper_d2": torch.empty(pairs, device="cuda", dtype=torch.float64),
            "query_norm2_center_fp64": h1.to_cuda(qn),
            "query_norm2_radius_fp64": h1.to_cuda(qnr),
            "query_l2_upper_fp64": h1.to_cuda(ql2),
            "base_norm2_center_fp64": h1.to_cuda(bn),
            "base_norm2_radius_fp64": h1.to_cuda(bnr),
            "base_l2_upper_fp64": h1.to_cuda(bl2),
        }
    )
    torch.cuda.synchronize()
    unit_roundoff = 2.0**-24
    path_operations = 2 * data.dimension + 2
    gamma_dot = (path_operations * unit_roundoff) / (
        1.0 - path_operations * unit_roundoff
    )
    return PedanticState(
        h1_state=state,
        gamma_dot=gamma_dot,
        dot_absolute_radius=4.0
        * data.dimension
        * float(np.finfo(np.float32).tiny),
        fp64_relative_guard=FP64_RECONSTRUCTION_FACTOR * np.finfo(np.float64).eps,
    )


def launch_validation_intervals(state: PedanticState) -> None:
    s = state.h1_state
    t = s.tensors
    pairs = s.query_tokens * s.base_tokens
    dot_to_d2_interval_fp64_rect[(triton.cdiv(pairs, INTERVAL_BLOCK),)](
        t["pedantic_dots"],
        t["query_norm2_center_fp64"],
        t["query_norm2_radius_fp64"],
        t["query_l2_upper_fp64"],
        t["base_norm2_center_fp64"],
        t["base_norm2_radius_fp64"],
        t["base_l2_upper_fp64"],
        t["pedantic_lower_d2"],
        t["pedantic_upper_d2"],
        state.gamma_dot,
        state.dot_absolute_radius,
        state.fp64_relative_guard,
        ABSOLUTE_RECONSTRUCTION_GUARD,
        PAIRS=pairs,
        BASE_TOKENS=s.base_tokens,
        BLOCK=INTERVAL_BLOCK,
        num_warps=4,
    )


def run_baseline(
    state: PedanticState, cublas: CuBLAS, threshold: float
) -> dict[str, object]:
    s = state.h1_state
    t = s.tensors
    base_objects = len(s.data.base.slices)
    t["threshold"].fill_(threshold)
    t["counters"].zero_()
    cublas.gemm(t["query"], t["base"], t["pedantic_dots"])
    aggregate_pedantic_dot_interval_objects_i64[(s.capacity,)](
        t["pedantic_dots"],
        t["query_norm2_center_fp64"],
        t["query_norm2_radius_fp64"],
        t["query_l2_upper_fp64"],
        t["base_norm2_center_fp64"],
        t["base_norm2_radius_fp64"],
        t["base_l2_upper_fp64"],
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
        state.gamma_dot,
        state.dot_absolute_radius,
        state.fp64_relative_guard,
        ABSOLUTE_RECONSTRUCTION_GUARD,
        BASE_OBJECTS=base_objects,
        BASE_TOKENS=s.base_tokens,
        CAPACITY=s.capacity,
        MAX_TOKENS=h1.MAX_TOKENS,
        num_warps=1,
    )
    first_counts = t["counters"].cpu().numpy().astype(np.int64)
    direct_count = int(first_counts[0])
    ambiguous_count = int(first_counts[1])
    if max(direct_count, ambiguous_count) > s.capacity or int(first_counts[2]):
        raise RuntimeError(f"pedantic compaction overflow: {first_counts.tolist()}")
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
            K=s.data.dimension,
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
            CAPACITY=s.capacity,
            MAX_TOKENS=h1.MAX_TOKENS,
            num_warps=1,
        )
    torch.cuda.synchronize()
    final_counts = t["counters"].cpu().numpy().astype(np.int64)
    final_count = int(final_counts[0])
    if final_count > s.capacity or int(final_counts[2]):
        raise RuntimeError(f"pedantic final overflow: {final_counts.tolist()}")
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
    }


def direct_fp64_dots(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    return query.astype(np.float64) @ base.astype(np.float64).T


def validate_dataset(
    data: h1.PreparedDataset,
    cublas: CuBLAS,
    *,
    include_candidate: bool,
) -> dict[str, object]:
    started = time.time()
    exact_tokens, exact_objects = h1.cpu_exact(data)
    exact_dots = direct_fp64_dots(data.query.vectors, data.base.vectors)
    state = make_state(data)
    s = state.h1_state
    t = s.tensors
    cublas.gemm(t["query"], t["base"], t["pedantic_dots"])
    launch_validation_intervals(state)
    torch.cuda.synchronize()
    observed_dots = t["pedantic_dots"].cpu().numpy().reshape(exact_dots.shape)
    lower_tokens = t["pedantic_lower_d2"].cpu().numpy().reshape(exact_tokens.shape)
    upper_tokens = t["pedantic_upper_d2"].cpu().numpy().reshape(exact_tokens.shape)
    q_l2 = t["query_l2_upper_fp64"].cpu().numpy()
    b_l2 = t["base_l2_upper_fp64"].cpu().numpy()
    dot_radius = state.gamma_dot * q_l2[:, None] * b_l2[None, :] + state.dot_absolute_radius
    dot_error = np.abs(observed_dots.astype(np.float64) - exact_dots)
    dot_radius_violations = int(
        np.count_nonzero(dot_error > dot_radius + ABSOLUTE_RECONSTRUCTION_GUARD)
    )
    token_lower_violations = int(
        np.count_nonzero(exact_tokens + CONTAINMENT_GUARD < lower_tokens)
    )
    token_upper_violations = int(
        np.count_nonzero(exact_tokens - CONTAINMENT_GUARD > upper_tokens)
    )
    cells = []
    for target, threshold in sorted(data.thresholds.items()):
        baseline = run_baseline(state, cublas, threshold)
        object_lower = t["object_lower"].cpu().numpy().reshape(exact_objects.shape)
        object_upper = t["object_upper"].cpu().numpy().reshape(exact_objects.shape)
        oracle_ids = np.flatnonzero(exact_objects.reshape(-1) <= threshold).astype(np.uint64)
        baseline_ids = np.asarray(baseline["ids"], dtype=np.uint64)
        exact_inside = exact_objects <= threshold
        direct_accept = object_upper <= threshold
        direct_reject = object_lower > threshold
        candidate_ids = None
        if include_candidate:
            candidate_ids = np.asarray(
                h1.run_candidate(s, threshold)["ids"], dtype=np.uint64
            )
        row = {
            "target_results_per_query": int(target),
            "threshold": float(threshold),
            "oracle_count": int(oracle_ids.size),
            "baseline_count": int(baseline_ids.size),
            "baseline_ambiguous_objects": int(baseline["ambiguous_count"]),
            "baseline_ambiguous_fraction": float(
                baseline["ambiguous_count"] / exact_objects.size
            ),
            "dot_radius_violations": dot_radius_violations,
            "token_lower_containment_violations": token_lower_violations,
            "token_upper_containment_violations": token_upper_violations,
            "object_lower_containment_violations": int(
                np.count_nonzero(exact_objects + CONTAINMENT_GUARD < object_lower)
            ),
            "object_upper_containment_violations": int(
                np.count_nonzero(exact_objects - CONTAINMENT_GUARD > object_upper)
            ),
            "unsafe_direct_accepts": int(np.count_nonzero(direct_accept & ~exact_inside)),
            "unsafe_direct_rejects": int(np.count_nonzero(direct_reject & exact_inside)),
            "baseline_mismatches": int(np.setxor1d(baseline_ids, oracle_ids).size),
            "baseline_duplicate_ids": h1.duplicate_count(baseline_ids),
            "baseline_overflow": int(baseline["overflow"]),
            "oracle_ids_sha256": sha256_u64(oracle_ids),
            "baseline_ids_sha256": sha256_u64(baseline_ids),
        }
        if candidate_ids is not None:
            row.update(
                {
                    "candidate_count": int(candidate_ids.size),
                    "candidate_mismatches": int(
                        np.setxor1d(candidate_ids, oracle_ids).size
                    ),
                    "candidate_baseline_mismatches": int(
                        np.setxor1d(candidate_ids, baseline_ids).size
                    ),
                    "candidate_ids_sha256": sha256_u64(candidate_ids),
                }
            )
        row["correctness_pass"] = bool(
            all(
                row[key] == 0
                for key in (
                    "dot_radius_violations",
                    "token_lower_containment_violations",
                    "token_upper_containment_violations",
                    "object_lower_containment_violations",
                    "object_upper_containment_violations",
                    "unsafe_direct_accepts",
                    "unsafe_direct_rejects",
                    "baseline_mismatches",
                    "baseline_duplicate_ids",
                    "baseline_overflow",
                )
            )
            and (candidate_ids is None or row["candidate_mismatches"] == 0)
            and (candidate_ids is None or row["candidate_baseline_mismatches"] == 0)
        )
        cells.append(row)
    return {
        "dataset": data.name,
        "dimension": data.dimension,
        "query_objects": len(data.query.slices),
        "base_objects": len(data.base.slices),
        "query_tokens": s.query_tokens,
        "base_tokens": s.base_tokens,
        "gamma_dot": state.gamma_dot,
        "dot_absolute_radius": state.dot_absolute_radius,
        "maximum_dot_error": float(dot_error.max(initial=0.0)),
        "maximum_dot_radius": float(dot_radius.max(initial=0.0)),
        "maximum_interval_width": float((upper_tokens - lower_tokens).max(initial=0.0)),
        "cells": cells,
        "correctness_pass": all(row["correctness_pass"] for row in cells),
        "elapsed_s": time.time() - started,
    }


def make_object_batch(vectors: np.ndarray, counts: np.ndarray, prefix: str) -> h1.ObjectBatch:
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    slices = tuple(
        slice(int(start), int(start + count)) for start, count in zip(starts, counts)
    )
    ids = np.asarray([f"{prefix}_{index}" for index in range(len(counts))])
    return h1.ObjectBatch(
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        object_ids=ids,
        slices=slices,
        token_counts=np.ascontiguousarray(counts, dtype=np.int32),
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    return np.ascontiguousarray(values / np.maximum(norms, 1e-30), dtype=np.float32)


def adversarial_dataset(case: str, dimension: int) -> h1.PreparedDataset:
    q_counts = np.asarray([4, 5, 6, 7], dtype=np.int32)
    b_counts = np.asarray([7, 6, 5, 4, 4, 5, 6, 7], dtype=np.int32)
    q_tokens = int(q_counts.sum())
    b_tokens = int(b_counts.sum())
    rng = np.random.default_rng(0x483242 + dimension + sum(map(ord, case)))
    query = normalize_rows(rng.normal(size=(q_tokens, dimension)).astype(np.float32))
    base = normalize_rows(rng.normal(size=(b_tokens, dimension)).astype(np.float32))
    if case == "all_zero":
        query.fill(0.0)
        base.fill(0.0)
    elif case == "identical_prefix":
        base[:q_tokens] = query
    elif case == "sign_flipped_prefix":
        base[:q_tokens] = -query
    elif case == "cancellation_alternating":
        signs = np.where(np.arange(dimension) % 2 == 0, 1.0, -1.0).astype(np.float32)
        base[:q_tokens] = normalize_rows(query * signs[None, :])
    elif case == "one_hot":
        query.fill(0.0)
        base.fill(0.0)
        query[np.arange(q_tokens), np.arange(q_tokens) % dimension] = 1.0
        base[np.arange(b_tokens), (3 * np.arange(b_tokens) + 1) % dimension] = 1.0
    elif case == "small_normal_scale":
        query *= np.float32(2.0**-12)
        base *= np.float32(2.0**-12)
    elif case == "ragged_high_entropy":
        pass
    else:
        raise ValueError(case)
    query_batch = make_object_batch(query, q_counts, f"{case}_q")
    base_batch = make_object_batch(base, b_counts, f"{case}_b")
    provisional = h1.PreparedDataset(
        name=f"adversarial_{case}_d{dimension}",
        dimension=dimension,
        query=query_batch,
        base=base_batch,
        thresholds={},
        cache=Path(f"synthetic://{case}/d{dimension}"),
        cache_sha256=hashlib.sha256(query.tobytes() + base.tobytes()).hexdigest(),
    )
    _, object_scores = h1.cpu_exact(provisional)
    if case == "all_zero":
        threshold = 0.0
    else:
        ordered = np.sort(object_scores.reshape(-1))
        position = max(1, len(ordered) // 4)
        threshold = float(0.5 * (ordered[position - 1] + ordered[position]))
    provisional.thresholds = {8: threshold}
    return provisional


def rectangular_smoke(cublas: CuBLAS, dimension: int) -> dict[str, object]:
    rng = np.random.default_rng(0xC0B1A5 + dimension)
    query = normalize_rows(rng.normal(size=(5, dimension)).astype(np.float32))
    base = normalize_rows(rng.normal(size=(7, dimension)).astype(np.float32))
    q_gpu = h1.to_cuda(query)
    b_gpu = h1.to_cuda(base)
    out = torch.empty(35, device="cuda", dtype=torch.float32)
    cublas.gemm(q_gpu, b_gpu, out)
    torch.cuda.synchronize()
    observed = out.cpu().numpy().reshape(5, 7).astype(np.float64)
    exact = direct_fp64_dots(query, base)
    qn = np.linalg.norm(query.astype(np.float64), axis=1)
    bn = np.linalg.norm(base.astype(np.float64), axis=1)
    u = 2.0**-24
    operations = 2 * dimension + 2
    gamma = operations * u / (1.0 - operations * u)
    radius = gamma * qn[:, None] * bn[None, :] + 4.0 * dimension * np.finfo(np.float32).tiny
    errors = np.abs(observed - exact)
    violations = int(
        np.count_nonzero(errors > radius + ABSOLUTE_RECONSTRUCTION_GUARD)
    )
    return {
        "dimension": dimension,
        "shape": [5, 7, dimension],
        "maximum_dot_error": float(errors.max()),
        "maximum_dot_radius": float(radius.max()),
        "dot_radius_violations": violations,
        "pass": violations == 0,
    }


def environment_record(cublas: CuBLAS) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "physical_gpu": PHYSICAL_GPU,
        "compute_capability": [properties.major, properties.minor],
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader",
                f"--id={PHYSICAL_GPU}",
            ],
            text=True,
        ).strip(),
        "cublas": cublas.record(),
    }


def run_correctness(cublas: CuBLAS) -> int:
    smokes = [rectangular_smoke(cublas, dimension) for dimension in (512, 2048)]
    datasets = [
        validate_dataset(h1.prepare_dataset(name), cublas, include_candidate=True)
        for name in h1.DATASETS
    ]
    cases = [
        "all_zero",
        "identical_prefix",
        "sign_flipped_prefix",
        "cancellation_alternating",
        "one_hot",
        "small_normal_scale",
        "ragged_high_entropy",
    ]
    adversarial = [
        validate_dataset(adversarial_dataset(case, dimension), cublas, include_candidate=False)
        for dimension in (512, 2048)
        for case in cases
    ]
    passed = bool(
        all(row["pass"] for row in smokes)
        and all(row["correctness_pass"] for row in datasets)
        and all(row["correctness_pass"] for row in adversarial)
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "correctness_single_process",
        "environment": environment_record(cublas),
        "constants": {
            "unit_roundoff": 2.0**-24,
            "dot_path_operations": "2D+2",
            "underflow_floor": "4D*tiny32",
            "fp64_reconstruction_factor": FP64_RECONSTRUCTION_FACTOR,
            "absolute_reconstruction_guard": ABSOLUTE_RECONSTRUCTION_GUARD,
            "containment_guard": CONTAINMENT_GUARD,
        },
        "rectangular_smokes": smokes,
        "datasets": datasets,
        "adversarial": adversarial,
        "correctness_pass": passed,
        "sources": {
            "runner_sha256": sha256_file(Path(__file__)),
            "kernel_sha256": sha256_file(PROJECT / "src/h2b_pedantic_kernels.py"),
            "h1_runner_sha256": H1_RUNNER_SHA256,
            "h1_kernel_sha256": H1_KERNEL_SHA256,
            "h2b_p0_sha256": H2B_P0_SHA256,
        },
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


def run_repeat(cublas: CuBLAS, slot: int) -> int:
    if sha256_file(OUTPUT) == "":
        raise RuntimeError("unreachable")
    accepted = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if not accepted.get("correctness_pass"):
        raise RuntimeError("repeat requires passing single-process correctness")
    datasets = []
    for name in h1.DATASETS:
        data = h1.prepare_dataset(name)
        state = make_state(data)
        accepted_dataset = next(row for row in accepted["datasets"] if row["dataset"] == name)
        cells = []
        for target, threshold in sorted(data.thresholds.items()):
            baseline = run_baseline(state, cublas, threshold)
            expected = next(
                row for row in accepted_dataset["cells"]
                if row["target_results_per_query"] == target
            )
            ids = np.asarray(baseline["ids"], dtype=np.uint64)
            output_hash = sha256_u64(ids)
            cells.append(
                {
                    "target_results_per_query": target,
                    "output_count": int(ids.size),
                    "output_sha256": output_hash,
                    "expected_count": int(expected["oracle_count"]),
                    "expected_sha256": expected["oracle_ids_sha256"],
                    "ambiguous_objects": int(baseline["ambiguous_count"]),
                    "pass": bool(
                        ids.size == int(expected["oracle_count"])
                        and output_hash == expected["oracle_ids_sha256"]
                        and int(baseline["overflow"]) == 0
                    ),
                }
            )
        datasets.append({"dataset": name, "cells": cells, "pass": all(x["pass"] for x in cells)})
    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "repeatability",
        "slot": slot,
        "environment": environment_record(cublas),
        "accepted_correctness_sha256": sha256_file(OUTPUT),
        "datasets": datasets,
        "repeat_pass": all(row["pass"] for row in datasets),
        "sources": {
            "runner_sha256": sha256_file(Path(__file__)),
            "kernel_sha256": sha256_file(PROJECT / "src/h2b_pedantic_kernels.py"),
        },
    }
    output = PROJECT / f"results/h2b_p1_pedantic_repeat_{slot}.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["repeat_pass"] else 3


def main() -> int:
    args = parse_args()
    require_dependencies()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("H2B-P1 requires NVIDIA_TF32_OVERRIDE=0 before CUDA init")
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    require_environment()
    cublas = CuBLAS()
    if args.phase == "correctness":
        return run_correctness(cublas)
    return run_repeat(cublas, args.slot)


if __name__ == "__main__":
    raise SystemExit(main())
