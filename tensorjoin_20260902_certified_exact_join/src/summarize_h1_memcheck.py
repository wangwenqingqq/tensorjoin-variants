#!/usr/bin/env python3
"""Create the immutable H1 Compute Sanitizer gate from four raw logs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "src/run_h1_multivector_gpu_screen.py"
KERNEL = PROJECT / "src/h1_multivector_kernels.py"
OUTPUT = PROJECT / "results/h1_r1_multivector_memcheck_summary.json"
CASES = tuple(
    (dataset, variant)
    for dataset in ("esc50_panns", "ucf101_r3d18")
    for variant in ("keeper", "candidate")
)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> int:
    runner_hash = sha256_file(RUNNER)
    kernel_hash = sha256_file(KERNEL)
    rows: list[dict[str, object]] = []
    for dataset, variant in CASES:
        path = PROJECT / f"raw/h1_r1_memcheck_{dataset}_{variant}.log"
        text = path.read_text(encoding="utf-8", errors="replace")
        summaries = [int(value) for value in re.findall(r"ERROR SUMMARY:\s*(\d+) errors", text)]
        run_lines = [line.split("RUN_START ", 1)[1] for line in text.splitlines() if "RUN_START " in line]
        smoke_present = "MEMCHECK_SMOKE " in text
        run_record = json.loads(run_lines[-1]) if run_lines else {}
        row = {
            "dataset": dataset,
            "variant": variant,
            "log": str(path),
            "log_sha256": sha256_file(path),
            "error_summaries": summaries,
            "memcheck_smoke_present": smoke_present,
            "runner_hash_match": run_record.get("script_sha256") == runner_hash,
            "kernel_hash_match": run_record.get("kernel_sha256") == kernel_hash,
        }
        row["pass"] = bool(
            summaries
            and all(value == 0 for value in summaries)
            and smoke_present
            and row["runner_hash_match"]
            and row["kernel_hash_match"]
        )
        rows.append(row)
    result = {
        "experiment_id": "tensorjoin_20260903_h1_r1_memcheck_gate",
        "tool": "NVIDIA Compute Sanitizer memcheck",
        "scope": "bounded actual-data 4x8 object smoke for both D512 ragged and D2048 fixed-cardinality paths",
        "runner": str(RUNNER),
        "runner_sha256": runner_hash,
        "kernel": str(KERNEL),
        "kernel_sha256": kernel_hash,
        "cases": rows,
        "safety_pass": all(bool(row["pass"]) for row in rows),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
