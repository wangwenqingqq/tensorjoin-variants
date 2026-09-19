#!/usr/bin/env python3
"""Summarize the frozen G2A correctness/capacity admission evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT / "results"
RAW_DIR = PROJECT / "raw"
OUTPUT = RESULT_DIR / "g2a_summary.json"
RECEIPT = PROJECT / "receipts/g2a_summary_sha256.txt"
METHODS = ("gds", "mistic", "tensorjoin")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT} or {RECEIPT}")
    prepared = load(RESULT_DIR / "g2a_prepare.json")
    oracle_count = int(prepared["oracle_pair_count"])
    oracle_hash = prepared["artifact_files"]["oracle_pairs_u64_le.bin"]["sha256"]
    sources: dict[str, str] = {}
    methods: dict[str, object] = {}
    all_pass = True

    for method in METHODS:
        rows: list[dict[str, object]] = []
        for run_id in range(2):
            path = RESULT_DIR / f"g2a_{method}_process_{run_id}.json"
            row = load(path)
            occupancy_path = RAW_DIR / f"g2a_{method}_process_{run_id}_occupancy.log"
            preflight_path = RAW_DIR / f"g2a_{method}_process_{run_id}_preflight.log"
            occupancy = occupancy_path.read_text(encoding="utf-8")
            preflight = preflight_path.read_text(encoding="utf-8")
            isolation_pass = (
                "python_exit=0 contamination=0" in occupancy
                and "FOREIGN" not in occupancy
                and "quiescence_pass=" in preflight
                and "FOREIGN" not in preflight
            )
            row_pass = (
                bool(row["exact_match"])
                and int(row["canonical_pair_count"]) == oracle_count
                and row["canonical_raw_u64_sha256"] == oracle_hash
                and isolation_pass
            )
            all_pass &= row_pass
            sources[str(path.relative_to(PROJECT))] = sha256_file(path)
            sources[str(occupancy_path.relative_to(PROJECT))] = sha256_file(occupancy_path)
            sources[str(preflight_path.relative_to(PROJECT))] = sha256_file(preflight_path)
            rows.append(
                {
                    "run_id": run_id,
                    "pass": row_pass,
                    "isolation_pass": isolation_pass,
                    "pair_count": int(row["canonical_pair_count"]),
                    "pair_hash": row["canonical_raw_u64_sha256"],
                    "diagnostic_wall_s": float(
                        row.get("diagnostic_wall_s", row.get("diagnostic_total_wall_s"))
                    ),
                }
            )

        repeated_hash_pass = len({row["pair_hash"] for row in rows}) == 1
        method_pass = all(bool(row["pass"]) for row in rows) and repeated_hash_pass
        all_pass &= method_pass
        method_summary: dict[str, object] = {
            "pass": method_pass,
            "repeated_hash_pass": repeated_hash_pass,
            "runs": rows,
            "diagnostic_only_no_performance_claim": True,
        }
        if method == "tensorjoin":
            detailed = [
                load(RESULT_DIR / f"g2a_tensorjoin_process_{run_id}.json")
                for run_id in range(2)
            ]
            work_fields = (
                "accepted_capacity",
                "capacity_attempts",
                "direct_accept_pairs",
                "ambiguous_pairs",
                "fp32_accept_pairs",
                "fp32_reject_pairs",
                "fp64_refine_pairs",
            )
            deterministic_work_pass = all(
                detailed[0][field] == detailed[1][field] for field in work_fields
            )
            all_pass &= deterministic_work_pass
            method_summary.update(
                {
                    "deterministic_work_pass": deterministic_work_pass,
                    **{field: detailed[0][field] for field in work_fields},
                }
            )
        methods[method] = method_summary

    source_build_receipts = {
        "gds": "receipts/g2a_gds_adapter_build.json",
        "mistic": "receipts/g2a_mistic_adapter_build.json",
    }
    for path_string in source_build_receipts.values():
        path = PROJECT / path_string
        sources[path_string] = sha256_file(path)
    sources["results/g2a_prepare.json"] = sha256_file(RESULT_DIR / "g2a_prepare.json")
    sources["src/summarize_g2a.py"] = sha256_file(Path(__file__).resolve())

    summary = {
        "experiment_id": "tensorjoin_20260903_external_exact_selfjoin_g2a",
        "decision": "accepted_for_g2b_admission" if all_pass else "failed",
        "gate_pass": all_pass,
        "scope": "4096x4096x512 Cifar60K subset canonical-ID correctness and scalable-capacity admission",
        "performance_claim_allowed": False,
        "oracle": {
            "pair_count": oracle_count,
            "raw_u64_sha256": oracle_hash,
            "epsilon": prepared["radius"]["epsilon"],
            "effective_epsilon_d2": prepared["radius"]["effective_epsilon_d2"],
        },
        "methods": methods,
        "source_build_receipts": source_build_receipts,
        "source_hashes": sources,
    }
    atomic_text(OUTPUT, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(
        RECEIPT,
        f"{sha256_file(OUTPUT)}  results/g2a_summary.json\n"
        f"{sha256_file(Path(__file__).resolve())}  src/summarize_g2a.py\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
