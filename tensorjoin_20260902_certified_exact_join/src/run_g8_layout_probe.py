#!/usr/bin/env python3
"""G8 frozen-layout correctness, occupancy, and diagnostic timing probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, D, Q = 4096, 512, 102079
THRESHOLD = 0.5686872086178483
METHODS = ("P1", "P16", "T4x4")
ID_SHA = "6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42"
X_SHA = "e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rawsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def build_cases():
    xp = ROOT / "data/g2a_cifar4096/vectors_f32.npy"
    ip = ROOT / "results/g3c_a_g3b_r1_ambiguous_u64_le.bin"
    assert sha(xp) == X_SHA and sha(ip) == ID_SHA
    x = np.load(xp, allow_pickle=False)
    ids = np.fromfile(ip, dtype="<u8").astype(np.int64)
    assert x.shape == (N, D) and ids.size == Q
    r, c = ids // N, ids % N
    identity = np.arange(N, dtype=np.int64)
    started = time.perf_counter()
    random = np.random.default_rng(20260904).permutation(N)
    random_s = time.perf_counter() - started
    started = time.perf_counter()
    degree = np.bincount(np.r_[r, c], minlength=N)
    degree_order = np.argsort(-degree, kind="stable")
    degree_s = time.perf_counter() - started
    started = time.perf_counter()
    centered = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    values, basis = np.linalg.eigh(centered.T @ centered)
    pc = basis[:, -1]
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc
    pca_order = np.argsort(centered @ pc, kind="stable")
    pca_s = time.perf_counter() - started
    sr, sc = np.triu_indices(453)
    synthetic = (sr * N + sc)[:Q].astype(np.int64)
    definitions = [
        ("real_identity", ids, identity, 0.),
        ("real_random", ids, random, random_s),
        ("real_degree_oracle", ids, degree_order, degree_s),
        ("real_pca1", ids, pca_order, pca_s),
        ("synthetic_clustered", synthetic, identity, 0.),
        ("synthetic_scattered", synthetic, random, random_s),
    ]
    cases = []
    for name, logical, order, layout_s in definitions:
        started = time.perf_counter()
        inv = np.empty(N, dtype=np.int64)
        inv[order] = np.arange(N)
        left, right = inv[logical // N], inv[logical % N]
        row, col = np.minimum(left, right), np.maximum(left, right)
        tile = (row // 4) * (N // 4) + col // 4
        cell = (row % 4) * 4 + col % 4
        permutation = np.argsort(tile * 16 + cell, kind="stable")
        row, col, tile, cell = [v[permutation] for v in (row, col, tile, cell)]
        unique, inverse, counts = np.unique(tile, return_inverse=True, return_counts=True)
        slots = np.full((len(unique), 16), -1, dtype=np.int32)
        slots[inverse, cell] = np.arange(Q, dtype=np.int32)
        physical_ids = (row * N + col).astype(np.int64)
        mapped = np.minimum(order[row], order[col]) * N + np.maximum(order[row], order[col])
        assert np.array_equal(mapped, logical[permutation])
        assert np.array_equal(np.sort(slots[slots >= 0]), np.arange(Q))
        arranged_x = np.ascontiguousarray(x[order])
        planning_s = time.perf_counter() - started
        info = {
            "name": name, "synthetic": name.startswith("synthetic"),
            "count": Q, "logical_ids_sha256": rawsha(np.sort(logical).astype("<u8")),
            "row_permutation_sha256": rawsha(order.astype("<i8")),
            "active_tiles": len(unique), "padded_pairs": len(unique) * 16,
            "padding_amplification": len(unique) * 16 / Q,
            "mean_tile_occupancy_fraction": Q / (len(unique) * 16),
            "occupancy_quantiles": dict(zip(("min", "p50", "p90", "max"),
                np.quantile(counts, [0, .5, .9, 1]).tolist())),
            "occupancy_histogram": np.bincount(counts, minlength=17).tolist(),
            "layout_host_seconds": layout_s, "queue_and_row_copy_host_seconds": planning_s,
        }
        cases.append(dict(info=info, x=arranged_x, ids=physical_ids,
                          tiles=unique.astype(np.int32), slots=slots,
                          logical=logical[permutation]))
    return x, cases


def cpu_truth(x, logical):
    truth = np.empty(len(logical), dtype=np.uint8)
    for start in range(0, len(logical), 1024):
        p = logical[start:start + 1024]
        delta = x[p // N].astype(np.float64) - x[p % N].astype(np.float64)
        truth[start:start + len(p)] = (np.sum(delta * delta, axis=1) <= THRESHOLD).astype(np.uint8)
    return truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("structure", "check", "timing"), required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    outpath = ROOT / f"results/{args.label}.json"
    if outpath.exists():
        raise FileExistsError(outpath)
    result = dict(experiment_id="tensorjoin_20260904_g8_refinement_layout_g2a4096",
                  mode=args.mode, label=args.label, reverse=args.reverse,
                  host=platform.node(), python=sys.version, numpy=np.__version__,
                  pid=os.getpid(), runner_sha256=sha(__file__),
                  kernel_source_sha256=sha(ROOT / "src/g8_layout_kernels.py"),
                  protocol_sha256=sha(ROOT / "PROTOCOL_G8_REFINEMENT_LAYOUT_A0.md"),
                  scope="resident GPU three-way FP32 refinement only; no FP64 repair or planning",
                  cases=[])
    x, cases = build_cases()
    if args.mode == "structure":
        result["cases"] = [c["info"] for c in cases]
        write_json(outpath, result)
        print(json.dumps(result), flush=True)
        return
    assert platform.node() == "gpu-host-8"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3"
    import torch
    import triton
    from g8_layout_kernels import pair_filter, tile_filter
    result.update(torch=torch.__version__, triton=triton.__version__,
                  cuda_runtime=torch.version.cuda, device=torch.cuda.get_device_name(0))
    artifacts = ROOT / f"artifacts/{args.label}"
    artifacts.mkdir(exist_ok=False)
    rounded = np.float32(THRESHOLD)
    lo = float(np.nextafter(rounded, np.float32(-np.inf))) if float(rounded) > THRESHOLD else float(rounded)
    hi = float(np.nextafter(rounded, np.float32(np.inf))) if float(rounded) < THRESHOLD else float(rounded)
    references = {}
    expected_truth_hash = {}
    compile_records = {}
    if args.reverse:
        cases.reverse()
    for case in cases:
        info = case["info"]
        name = info["name"]
        print("CASE", name, "tiles", info["active_tiles"], flush=True)
        gx, gi, gt, gs = [torch.from_numpy(case[k]).cuda() for k in ("x", "ids", "tiles", "slots")]
        outputs = {m: torch.full((Q + 64,), 73, device="cuda", dtype=torch.uint8) for m in METHODS}
        def launch(method):
            target = outputs[method][32:32 + Q]
            if method == "T4x4":
                return tile_filter[(len(case["tiles"]),)](gx, gt, gs, target,
                    N=N, D=D, LO=lo, HI=hi, K=256, num_warps=4,
                    num_stages=1, enable_fp_fusion=False)
            b = 1 if method == "P1" else 16
            return pair_filter[(triton.cdiv(Q, b),)](gx, gi, target, Q=Q,
                N=N, D=D, LO=lo, HI=hi, B=b, K=256, num_warps=4,
                num_stages=1, enable_fp_fusion=False)
        truth = cpu_truth(x, case["logical"])
        order = np.argsort(case["logical"])
        def validate():
            observed = {}
            states = {}
            for m in METHODS:
                a = outputs[m].cpu().numpy()
                assert np.all(a[:32] == 73) and np.all(a[-32:] == 73), (name, m, "guard")
                s = a[32:32 + Q]
                assert np.all(s <= 2), (name, m, "unwritten")
                assert np.all(s[s != 2] == truth[s != 2]), (name, m, "false decision")
                observed[m] = dict(counts=np.bincount(s, minlength=3).tolist(),
                                   logical_state_sha256=rawsha(s[order]))
                states[m] = s.copy()
            assert np.array_equal(states["P1"], states["P16"]), (name, "P16 state mismatch")
            assert np.array_equal(states["P1"], states["T4x4"]), (name, "tile state mismatch")
            family = "synthetic" if info["synthetic"] else "real"
            state_sha = observed["P1"]["logical_state_sha256"]
            truth_sha = rawsha(truth[order])
            if family in references:
                assert references[family] == state_sha and expected_truth_hash[family] == truth_sha
            references[family], expected_truth_hash[family] = state_sha, truth_sha
            return observed
        for m in METHODS:
            compiled = launch(m)
            if m not in compile_records:
                prefix = artifacts / m
                for kind in ("ptx", "cubin"):
                    value = compiled.asm[kind]
                    path = prefix.with_suffix("." + kind)
                    path.write_bytes(value if isinstance(value, bytes) else value.encode())
                compile_records[m] = dict(metadata=compiled.metadata._asdict(),
                    n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                    cubin_sha256=sha(prefix.with_suffix(".cubin")),
                    ptx_sha256=sha(prefix.with_suffix(".ptx")))
        torch.cuda.synchronize()
        info["correctness_before"] = validate()
        info["oracle_true_count"] = int(truth.sum())
        if args.mode == "timing":
            for m in METHODS:
                for _ in range(20):
                    launch(m)
            torch.cuda.synchronize()
            raw = []
            methods = list(METHODS)[::-1] if args.reverse else list(METHODS)
            def event_time(m, repetitions):
                torch.cuda.synchronize()
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(repetitions):
                    launch(m)
                end.record()
                end.synchronize()
                return float(begin.elapsed_time(end)) / repetitions
            for round_id in range(40):
                schedule = methods[round_id % 3:] + methods[:round_id % 3]
                for position, m in enumerate(schedule):
                    raw.append(dict(round=round_id, position=position, method=m,
                                    ms=event_time(m, 20)))
            info["samples"] = raw
            info["sustained_ms_per_call"] = {m: event_time(m, 500) for m in methods}
        # Reallocation tests independently owned output addresses, including guard bytes.
        for m in METHODS:
            outputs[m] = torch.full((Q + 64,), 73, device="cuda", dtype=torch.uint8)
            for _ in range(3):
                launch(m)
        torch.cuda.synchronize()
        info["correctness_after"] = validate()
        info["correctness_pass"] = True
        result["cases"].append(info)
        result["compiled"] = compile_records
        write_json(outpath, result)
        print("PASS", name, info["correctness_before"]["P1"]["counts"], flush=True)
    result["correctness_pass"] = all(c["correctness_pass"] for c in result["cases"])
    write_json(outpath, result)
    print("COMPLETE", args.label, flush=True)


if __name__ == "__main__":
    main()
