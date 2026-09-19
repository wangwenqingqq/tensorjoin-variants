#!/usr/bin/env python3
"""Summarize frozen H2B-P3-R3 access-safety and stress evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "h2b_p3_safety_summary.json"
TEXT = RAW / "h2b_p3_safety_summary.txt"
RUNNER_SHA256 = "63b19aab011f8a4e51f34c739011897ea5fc2a3afed99040622dd3c444d945ba"
P1_CORRECTNESS_SHA256 = "68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99"
P2_AUDIT_SHA256 = "c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77"
STRESS_SHA256 = "bd3c4b1c3c960da10962a1f31452a0018b4299046d75bfe02ba82a4ae18a1225"
GPU_UUID = "GPU-863c06a5-9f33-0265-b098-013fa840d5db"

EXPECTED_LOG_HASHES = {
    ("memcheck", "esc50_panns"): "4dfd64712600f41a825e97ff5b6ae14488f2836b65c31e4299fe4bc1ee0d050c",
    ("memcheck", "ucf101_r3d18"): "5379e72dadfce0231a4c7f38601867345939e997475e626a2f908a3d09567631",
    ("synccheck", "esc50_panns"): "58927da00f5ea065dbd5c89025a3b18216615cf28025c4c9bdd0786b4410875d",
    ("synccheck", "ucf101_r3d18"): "0a0a3a4c6342f83988cdf22e24d1eeb8c9e2f471282275891967b06b2d73b219",
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_sanitizer(tool: str, dataset: str, expected_hash: str) -> dict:
    path = RAW / f"h2b_p3_{tool}_{dataset}_r3.log"
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"sanitizer log hash mismatch: {path}")
    text = path.read_text()
    records = [
        json.loads(line.removeprefix("P3_SAFETY "))
        for line in text.splitlines()
        if line.startswith("P3_SAFETY ")
    ]
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", text)
    if len(records) != 1 or summaries != ["0"]:
        raise RuntimeError(f"bad sanitizer result in {path}")
    record = records[0]
    if (
        record["dataset"] != dataset
        or record["target_results_per_query"] != 8
        or record["physical_gpu"] != 4
        or not record["output"]["pass"]
        or record["sources"]["runner_sha256"] != RUNNER_SHA256
        or record["sources"]["p1_correctness_sha256"] != P1_CORRECTNESS_SHA256
        or record["sources"]["p2_audit_sha256"] != P2_AUDIT_SHA256
    ):
        raise RuntimeError(f"sanitizer semantic mismatch in {path}")
    return {
        "tool": tool,
        "dataset": dataset,
        "target_results_per_query": 8,
        "log": str(path.relative_to(PROJECT)),
        "log_sha256": actual_hash,
        "error_summary": 0,
        "output": record["output"],
        "pass": True,
    }


def parse_stress() -> tuple[dict, list[dict]]:
    path = RESULTS / "h2b_p3_stress.json"
    if sha256_file(path) != STRESS_SHA256:
        raise RuntimeError("stress artifact hash mismatch")
    result = json.loads(path.read_text())
    if (
        not result.get("stress_pass")
        or result.get("total_retained_invocations") != 4000
        or result["sources"]["runner_sha256"] != RUNNER_SHA256
        or result["sources"]["p1_correctness_sha256"] != P1_CORRECTNESS_SHA256
        or result["sources"]["p2_audit_sha256"] != P2_AUDIT_SHA256
    ):
        raise RuntimeError("stress header mismatch")
    cells = []
    observed_total = 0
    for dataset in result["datasets"]:
        for cell in dataset["cells"]:
            observations = cell["observations"]
            observed_total += len(observations)
            signatures = {
                (
                    row["final_count"],
                    row["direct_accept_count"],
                    row["ambiguous_count"],
                    row["overflow"],
                    row["duplicate_ids"],
                    row["ids_sha256"],
                )
                for row in observations
            }
            slots = {row["state_slot"] for row in observations}
            passed = bool(
                cell["pass"]
                and cell["iterations"] == 1000
                and len(observations) == 1000
                and len(signatures) == 1
                and slots == {0, 1}
                and all(row["pass"] for row in observations)
            )
            if not passed:
                raise RuntimeError(
                    f"stress cell mismatch: {dataset['dataset']}/t{cell['target_results_per_query']}"
                )
            signature = next(iter(signatures))
            cells.append(
                {
                    "dataset": dataset["dataset"],
                    "target_results_per_query": cell["target_results_per_query"],
                    "iterations": 1000,
                    "state_slots": sorted(slots),
                    "unique_signatures": 1,
                    "signature": {
                        "final_count": signature[0],
                        "direct_accept_count": signature[1],
                        "ambiguous_count": signature[2],
                        "overflow": signature[3],
                        "duplicate_ids": signature[4],
                        "ids_sha256": signature[5],
                    },
                    "pass": True,
                }
            )
    if observed_total != 4000 or len(cells) != 4:
        raise RuntimeError("stress matrix size mismatch")
    return result, cells


def main() -> int:
    if OUTPUT.exists() or TEXT.exists():
        raise FileExistsError("P3 summary already exists")
    if sha256_file(PROJECT / "src/run_h2b_p3_safety_stress.py") != RUNNER_SHA256:
        raise RuntimeError("P3 runner hash mismatch")
    sanitizer = [
        parse_sanitizer(tool, dataset, digest)
        for (tool, dataset), digest in EXPECTED_LOG_HASHES.items()
    ]
    _, stress_cells = parse_stress()
    preflight = RAW / "h2b_p3_r3_preflight.txt"
    postflight = RAW / "h2b_p3_r3_postflight.txt"
    postflight_text = postflight.read_text()
    postflight_lines = postflight_text.splitlines()
    apps_index = postflight_lines.index("COMPUTE_APPS")
    target_lines = [line for line in postflight_lines[apps_index + 1 :] if GPU_UUID in line]
    if target_lines:
        raise RuntimeError("target GPU still has a compute process at postflight")
    gpu_lines = [line for line in postflight_text.splitlines() if line.startswith("4,")]
    if len(gpu_lines) != 1 or ", 14 MiB, 0 %" not in gpu_lines[0]:
        raise RuntimeError("GPU4 postflight is not idle")
    summary = {
        "experiment": "tensorjoin_20260903_h2b_p3_r3_access_safety_stress",
        "decision": "PASS",
        "claim_scope": "device-memory access and synchronization checks plus two-buffer output/count stability",
        "not_claimed": "leak-free process shutdown; R2 retained three PyTorch-owned cuBLAS-handle allocations",
        "host": "gpu-host-8",
        "physical_gpu": 4,
        "gpu_uuid": GPU_UUID,
        "sanitizer": sanitizer,
        "stress": {
            "artifact": "results/h2b_p3_stress.json",
            "artifact_sha256": STRESS_SHA256,
            "total_retained_invocations": 4000,
            "cells": stress_cells,
            "pass": True,
        },
        "known_resource_lifetime_limitation": {
            "rejected_full_leak_log": "raw/h2b_p3_memcheck_esc50_panns_r2.log",
            "rejected_full_leak_log_sha256": "6ae8d7e7d28961e0b68f13c0952f5a60c408fdae8abe95ec566240fa04a1f11c",
            "outstanding_allocations": 3,
            "outstanding_bytes": 8520704,
            "classification": "process-owned cuBLAS handle allocations rooted at cublasCreate_v2",
        },
        "prior_rejected_attempt": {
            "log": "raw/h2b_p3_memcheck_esc50_panns.log",
            "log_sha256": "60eb6d140957e85bd0c070220e5b3c8b3b3f78d7ea63086f1d32e6965a9a8610",
            "outstanding_allocations": 12,
            "outstanding_bytes": 153224192,
        },
        "environment_evidence": {
            "preflight_sha256": sha256_file(preflight),
            "postflight_sha256": sha256_file(postflight),
            "postflight_gpu_line": gpu_lines[0],
        },
        "sources": {
            "runner_sha256": RUNNER_SHA256,
            "p1_correctness_sha256": P1_CORRECTNESS_SHA256,
            "p2_audit_sha256": P2_AUDIT_SHA256,
            "summarizer_sha256": sha256_file(Path(__file__)),
        },
    }
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(payload)
    result_hash = hashlib.sha256(payload.encode()).hexdigest()
    lines = [
        "decision=PASS",
        "sanitizer_clean=4/4",
        "stress_cells=4/4",
        "stress_retained_invocations=4000",
        "unique_signatures_per_cell=1/1/1/1",
        "leak_free_claim=false",
        "known_cublas_handle_allocations=3",
        "known_cublas_handle_bytes=8520704",
        f"result_sha256={result_hash}",
    ]
    TEXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
