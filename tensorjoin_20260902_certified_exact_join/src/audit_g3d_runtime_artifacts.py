#!/usr/bin/env python3
"""Verify that G3D processes timed the accepted runtime cubins."""

from __future__ import annotations

import argparse
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest_path = PROJECT / f"results/g3d_{args.phase}_campaign_manifest.json"
    output_path = PROJECT / f"results/g3d_{args.phase}_generated_code_audit.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["admitted"]]
    expected_processes = 2 if args.phase == "screen" else 8
    if len(records) != expected_processes:
        raise RuntimeError(f"Expected exactly {expected_processes} admitted processes")
    process_records: list[dict[str, object]] = []
    for record in sorted(records, key=lambda value: value["process_id"]):
        cache = PROJECT / str(record["cache_path"])
        cubins = sorted(cache.rglob("*.cubin"))
        by_name: dict[str, list[dict[str, str]]] = {}
        for cubin in cubins:
            by_name.setdefault(cubin.name, []).append(
                {
                    "path": str(cubin.relative_to(PROJECT)),
                    "sha256": sha256_file(cubin),
                }
            )
        names_exact = set(by_name) == set(EXPECTED)
        one_each = all(len(by_name.get(name, [])) == 1 for name in EXPECTED)
        hashes_exact = all(
            len(by_name.get(name, [])) == 1
            and by_name[name][0]["sha256"] == expected_hash
            for name, expected_hash in EXPECTED.items()
        )
        process_records.append(
            {
                "process_id": record["process_id"],
                "attempt": record["attempt"],
                "cache_path": record["cache_path"],
                "cubins": by_name,
                "names_exact": names_exact,
                "one_each": one_each,
                "hashes_exact": hashes_exact,
                "pass": bool(names_exact and one_each and hashes_exact),
            }
        )
    gate = all(record["pass"] for record in process_records)
    result = {
        "experiment_id": "tensorjoin_20260903_g3d_complete_pipeline_timing",
        "phase": args.phase,
        "measurement_status": "runtime_binary_identity_not_performance",
        "expected_cubins": EXPECTED,
        "processes": process_records,
        "runtime_binary_gate_pass": gate,
        f"{args.phase}_runtime_binary_gate_pass": gate,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
