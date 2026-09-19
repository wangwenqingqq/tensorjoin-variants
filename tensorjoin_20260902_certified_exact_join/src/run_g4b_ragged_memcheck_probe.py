#!/usr/bin/env python3
"""Launch one minimal D=784 suspect G3B tile for Compute Sanitizer."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from run_g3b_r1_gpu_analytic_certificate import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    EXPRESSION_ABSOLUTE_RADIUS,
    EXPRESSION_RELATIVE_RADIUS,
    SQRT_RESIDUAL_FINAL_ABSOLUTE_RADIUS,
    SQRT_RESIDUAL_FINAL_RELATIVE_RADIUS,
    analytic_certificate_compact_i64,
    outward_float32,
)
from run_g4b_public_opportunity import quantize_with_analytic_residual


PROJECT = Path(__file__).resolve().parents[1]
N = 64
D = 784


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Probe requires physical GPU1 as sole visible device")
    vectors = np.ascontiguousarray(
        np.load(
            PROJECT / "data/g4b_public/fashion784/vectors_f32.npy",
            allow_pickle=False,
        )[:N],
        dtype=np.float32,
    )
    codes, scales, norms, errors = quantize_with_analytic_residual(vectors)
    device = torch.device("cuda:0")
    tile_rows = np.asarray([0], dtype=np.int32)
    tile_columns = np.asarray([0], dtype=np.int32)
    capacity = BLOCK_M * BLOCK_N
    buffers = {
        "codes": torch.from_numpy(codes).to(device),
        "codes_t": torch.from_numpy(codes.T.copy()).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "norms": torch.from_numpy(norms).to(device),
        "errors": torch.from_numpy(errors).to(device),
        "tile_rows": torch.from_numpy(tile_rows).to(device),
        "tile_columns": torch.from_numpy(tile_columns).to(device),
        "result": torch.empty(capacity, dtype=torch.int64, device=device),
        "ambiguous": torch.empty(capacity, dtype=torch.int64, device=device),
        "counters": torch.zeros(3, dtype=torch.int32, device=device),
    }
    epsilon_lower, epsilon_upper = outward_float32(1_000.0)
    analytic_certificate_compact_i64[(1,)](
        buffers["codes"],
        buffers["codes_t"],
        buffers["scales"],
        buffers["norms"],
        buffers["errors"],
        buffers["tile_rows"],
        buffers["tile_columns"],
        buffers["result"],
        buffers["ambiguous"],
        buffers["counters"],
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
    torch.cuda.synchronize()
    print("PROBE_COMPLETED", buffers["counters"].cpu().tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
