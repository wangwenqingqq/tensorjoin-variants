#!/usr/bin/env python3
"""Verify all G4C formal slots used the screen-audited per-dimension cubins."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results/g4c_formal_runtime_audit.json"
DATASETS = ("sift128", "cifar_gist512", "fashion784")
ORDERS = ("kc", "ck", "kc", "ck", "ck", "kc", "ck", "kc")
KERNELS = (
    "analytic_certificate_ragged_safe_i64",
    "certified_fp32_filter_i64",
    "refine_ambiguous_fp64_i64",
)
EXPECTED_SCREEN_AUDIT_SHA256 = (
    "a19142fe8259cd63db343233835daf189a62a95be4e0ee73debd2e107bfe3542"
)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


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


def exactly_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    screen_path = PROJECT / "results/g4c_screen_runtime_audit.json"
    if sha256_file(screen_path) != EXPECTED_SCREEN_AUDIT_SHA256:
        raise RuntimeError("Screen runtime-audit hash mismatch")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not screen.get("runtime_binary_gate_pass"):
        raise RuntimeError("Screen runtime audit did not pass")
    datasets: dict[str, object] = {}
    all_pass = True
    for dataset in DATASETS:
        expected = screen["datasets"][dataset]["process_cubin_hashes"][0]
        processes: list[dict[str, object]] = []
        for process_id, order in enumerate(ORDERS):
            stem = f"g4c_formal_{dataset}_n4096_k64_process_{process_id}_{order}_a0"
            manifest_path = PROJECT / f"results/{stem}_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache = PROJECT / str(manifest["cache_path"])
            found = {
                name: sha256_file(exactly_one(cache, f"{name}.cubin"))
                for name in KERNELS
            }
            gate = bool(manifest.get("admitted") and found == expected)
            processes.append(
                {
                    "process_id": process_id,
                    "order": order.upper(),
                    "manifest": str(manifest_path.relative_to(PROJECT)),
                    "manifest_sha256": sha256_file(manifest_path),
                    "cache_path": manifest["cache_path"],
                    "cubin_hashes": found,
                    "matches_screen_audited_cubins": found == expected,
                    "pass": gate,
                }
            )
        dataset_pass = all(record["pass"] for record in processes)
        all_pass = all_pass and dataset_pass
        datasets[dataset] = {
            "expected_screen_audited_cubin_hashes": expected,
            "processes": processes,
            "dataset_runtime_binary_gate_pass": dataset_pass,
        }
    result = {
        "experiment_id": "tensorjoin_20260903_g4c_public_anchor_formal",
        "measurement_status": "runtime_binary_identity_not_performance",
        "screen_runtime_audit_sha256": sha256_file(screen_path),
        "datasets": datasets,
        "runtime_binary_gate_pass": all_pass,
        "performance_claim_allowed": False,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4C_FORMAL.md"),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
