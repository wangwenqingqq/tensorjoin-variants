#!/usr/bin/env python3
"""Summarize the two immutable H2A FP32-baseline memcheck logs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results/h2a_fp32_strong_memcheck_summary.json"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sources() -> dict[str, str]:
    return {
        "runner_sha256": sha256_file(PROJECT / "src/run_h2a_fp32_strong_baseline.py"),
        "h2a_kernel_sha256": sha256_file(PROJECT / "src/h2a_fp32_multivector_kernels.py"),
        "h1_runner_sha256": sha256_file(PROJECT / "src/run_h1_multivector_gpu_screen.py"),
        "h1_kernel_sha256": sha256_file(PROJECT / "src/h1_multivector_kernels.py"),
    }


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


def main() -> int:
    expected_sources = sources()
    rows = []
    for dataset in ("esc50_panns", "ucf101_r3d18"):
        path = PROJECT / f"raw/h2a_memcheck_{dataset}.log"
        text = path.read_text(encoding="utf-8", errors="replace")
        errors = [int(value) for value in re.findall(r"ERROR SUMMARY:\s*(\d+) errors", text)]
        run_lines = [line.split("RUN_START ", 1)[1] for line in text.splitlines() if "RUN_START " in line]
        run_record = json.loads(run_lines[-1]) if run_lines else {}
        row = {
            "dataset": dataset,
            "log": str(path),
            "log_sha256": sha256_file(path),
            "error_summaries": errors,
            "memcheck_smoke_present": "MEMCHECK_SMOKE " in text,
            "sources_match": run_record.get("sources") == expected_sources,
        }
        row["pass"] = bool(
            errors
            and all(value == 0 for value in errors)
            and row["memcheck_smoke_present"]
            and row["sources_match"]
        )
        rows.append(row)
    result = {
        "experiment_id": "tensorjoin_20260903_h2a_fp32_memcheck",
        "tool": "NVIDIA Compute Sanitizer memcheck",
        "scope": "new certified-FP32 baseline on actual 4x8 D2048-fixed and D512-ragged subsets",
        "sources": expected_sources,
        "h1_safety_sha256": sha256_file(
            PROJECT / "results/h1_r1_multivector_memcheck_summary.json"
        ),
        "cases": rows,
        "safety_pass": all(bool(row["pass"]) for row in rows),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

