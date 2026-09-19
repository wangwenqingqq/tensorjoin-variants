#!/usr/bin/env python3
"""Issue one frozen H2B pedantic cuBLAS GEMM for profiler attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import run_h1_multivector_gpu_screen as h1
import run_h2b_p1_pedantic_correctness as p1


PROJECT = Path(__file__).resolve().parents[1]
P1_RUNNER_SHA256 = "d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1"
P1_KERNEL_SHA256 = "33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43"
P1_CORRECTNESS_SHA256 = "68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99"
P1_REPEAT_SHA256 = {
    0: "6970b41c794ec787e880778f9a501de61331a7eb4cda5303d85130db4215fc27",
    1: "75d22742a4a952aa19bfb8b2a53bd36508b45bc9211a71fe1e51cb84b4edda30",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(h1.DATASETS), required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def require_dependencies() -> None:
    expected = {
        PROJECT / "src/run_h2b_p1_pedantic_correctness.py": P1_RUNNER_SHA256,
        PROJECT / "src/h2b_pedantic_kernels.py": P1_KERNEL_SHA256,
        PROJECT / "results/h2b_p1_pedantic_correctness.json": P1_CORRECTNESS_SHA256,
        PROJECT / "results/h2b_p1_pedantic_repeat_0.json": P1_REPEAT_SHA256[0],
        PROJECT / "results/h2b_p1_pedantic_repeat_1.json": P1_REPEAT_SHA256[1],
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"P2 dependency mismatch: {path}")


def main() -> int:
    args = parse_args()
    require_dependencies()
    p1.require_dependencies()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("P2 requires NVIDIA_TF32_OVERRIDE=0 before CUDA init")
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    p1.require_environment()
    cublas = p1.CuBLAS()
    data = h1.prepare_dataset(args.dataset)
    query = h1.to_cuda(data.query.vectors)
    base = h1.to_cuda(data.base.vectors)
    output = torch.empty(
        len(data.query.vectors) * len(data.base.vectors),
        device="cuda",
        dtype=torch.float32,
    )
    torch.cuda.synchronize()
    range_name = f"H2B_P2_PEDANTIC_SGEMM_{args.dataset}"
    torch.cuda.nvtx.range_push(range_name)
    cublas.gemm(query, base, output)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    values = output.cpu().numpy().astype("<f4", copy=False)
    record = {
        "dataset": args.dataset,
        "range": range_name,
        "m_query_tokens": len(data.query.vectors),
        "n_base_tokens": len(data.base.vectors),
        "k_dimension": data.dimension,
        "output_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "output_min": float(values.min()),
        "output_max": float(values.max()),
        "output_nonfinite": int(np.count_nonzero(~np.isfinite(values))),
        "cublas": cublas.record(),
        "sources": {
            "trace_runner_sha256": sha256_file(Path(__file__)),
            "p1_runner_sha256": P1_RUNNER_SHA256,
            "p1_kernel_sha256": P1_KERNEL_SHA256,
            "p1_correctness_sha256": P1_CORRECTNESS_SHA256,
        },
    }
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record["output_nonfinite"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
