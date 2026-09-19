#!/usr/bin/env python3
"""Verify admitted G4A-R1 GPU1 processes used accepted runtime cubins."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "analytic_certificate_compact_i64.cubin": "dd32e97936d8e0c6abd3924d8fdeeb17088e7a6e1e4d8ac2f1fc492bbc2114ac",
    "certified_fp32_filter_i64.cubin": "db750295875f8363679cfe9f9762cef884e1ecd371e3e419fd500d91c3a20670",
    "refine_ambiguous_fp64_i64.cubin": "ff0572a7eefcb6ccba91c7da0020d370e45d7f8f7c03d3a55e8f5ca53c477c50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = PROJECT / f"results/g4a_r1_{args.phase}_runtime_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifests = sorted(
        (PROJECT / "results").glob(f"g4a_r1_{args.phase}_process_*_manifest.json")
    )
    admitted = []
    for path in manifests:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("admitted"):
            admitted.append(record)
    expected_processes = 2 if args.phase == "screen" else 8
    if len(admitted) != expected_processes or sorted(
        int(record["process_id"]) for record in admitted
    ) != list(range(expected_processes)):
        raise RuntimeError("Expected exactly one admitted manifest per process slot")

    processes: list[dict[str, object]] = []
    for record in sorted(admitted, key=lambda value: int(value["process_id"])):
        cache = PROJECT / str(record["cache_path"])
        by_name: dict[str, list[dict[str, str]]] = {}
        for cubin in sorted(cache.rglob("*.cubin")):
            by_name.setdefault(cubin.name, []).append(
                {
                    "path": str(cubin.relative_to(PROJECT)),
                    "sha256": sha256_file(cubin),
                }
            )
        gate = bool(
            set(by_name) == set(EXPECTED)
            and all(len(by_name[name]) == 1 for name in EXPECTED)
            and all(by_name[name][0]["sha256"] == value for name, value in EXPECTED.items())
        )
        processes.append(
            {
                "process_id": record["process_id"],
                "attempt": record["attempt"],
                "cache_path": record["cache_path"],
                "cubins": by_name,
                "pass": gate,
            }
        )
    gate = all(record["pass"] for record in processes)
    result = {
        "experiment_id": "tensorjoin_20260903_g4a_r1_dynamic_count_router_gpu1",
        "phase": args.phase,
        "measurement_status": "runtime_binary_identity_not_performance",
        "expected_cubins": EXPECTED,
        "processes": processes,
        "runtime_binary_gate_pass": gate,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
