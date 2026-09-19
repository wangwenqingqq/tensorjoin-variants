#!/usr/bin/env python3
"""Apply the frozen aggregate G4B public-breadth opportunity gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("sift128", "cifar_gist512", "fashion784")
NON_CIFAR = ("sift128", "fashion784")
RESULT = PROJECT / "results/g4b_opportunity_summary.json"


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


def main() -> int:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    results: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    evidence: list[dict[str, str]] = []
    for dataset in DATASETS:
        result_path = PROJECT / f"results/g4b_opportunity_{dataset}.json"
        manifest_path = PROJECT / f"results/g4b_opportunity_{dataset}_manifest.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result.get("dataset_id") != dataset or manifest.get("dataset_id") != dataset:
            raise RuntimeError("Dataset identity mismatch")
        if manifest.get("result_sha256") != sha256_file(result_path):
            raise RuntimeError("Isolation manifest result hash mismatch")
        results.append(result)
        manifests.append(manifest)
        evidence.extend(
            [
                {"path": str(result_path.relative_to(PROJECT)), "sha256": sha256_file(result_path)},
                {"path": str(manifest_path.relative_to(PROJECT)), "sha256": sha256_file(manifest_path)},
            ]
        )

    cells = [cell for result in results for cell in result["cells"]]
    identity = {
        (cell["dataset_id"], int(cell["n"]), int(cell["target_average_directed_nonself_degree"]))
        for cell in cells
    }
    expected_identity = {
        (dataset, n, k)
        for dataset in DATASETS
        for n in (1_024, 2_048, 4_096)
        for k in (1, 16, 64)
    }
    all_exact = bool(
        identity == expected_identity
        and len(cells) == 27
        and all(result.get("process_gate_pass") for result in results)
        and all(manifest.get("admitted") for manifest in manifests)
        and all(cell.get("cell_gate_pass") for cell in cells)
    )
    required_hash_keys = (
        "vector_prefix_raw_sha256",
        "source_row_prefix_raw_sha256",
        "oracle_metadata_sha256",
        "oracle_file_sha256",
        "oracle_upper_raw_u64_sha256",
        "accepted_upper_raw_u64_sha256",
    )
    all_hashes_recorded = bool(
        len(cells) == 27
        and all(
            isinstance(cell.get(key), str) and len(cell[key]) == 64
            for cell in cells
            for key in required_hash_keys
        )
    )
    residual_gate: dict[str, object] = {}
    ambiguity_gate: dict[str, object] = {}
    for dataset in NON_CIFAR:
        n4096 = [
            cell
            for cell in cells
            if cell["dataset_id"] == dataset and int(cell["n"]) == 4_096
        ]
        residual_passing = [
            int(cell["target_average_directed_nonself_degree"])
            for cell in n4096
            if float(cell["fp64_refine_fraction_of_g3b_ambiguity"]) <= 0.25
        ]
        low_medium_passing = [
            int(cell["target_average_directed_nonself_degree"])
            for cell in n4096
            if int(cell["target_average_directed_nonself_degree"]) in (1, 16)
            and float(cell["g3b_ambiguity_fraction_of_all_upper_pairs"]) <= 0.25
        ]
        residual_gate[dataset] = {
            "passing_degrees": residual_passing,
            "pass": len(residual_passing) >= 2,
        }
        ambiguity_gate[dataset] = {
            "passing_low_or_medium_degrees": low_medium_passing,
            "pass": bool(low_medium_passing),
        }
    cross_dataset_residual_pass = all(
        value["pass"] for value in residual_gate.values()
    )
    cross_dataset_ambiguity_pass = all(
        value["pass"] for value in ambiguity_gate.values()
    )
    opportunity_gate_pass = bool(
        all_exact
        and all_hashes_recorded
        and cross_dataset_residual_pass
        and cross_dataset_ambiguity_pass
    )
    summary = {
        "experiment_id": "tensorjoin_20260903_g4b_public_breadth_opportunity",
        "measurement_status": "correctness_and_routing_geometry_not_performance",
        "cell_count": len(cells),
        "all_27_cells_exact_and_structurally_valid": all_exact,
        "all_required_hashes_recorded": all_hashes_recorded,
        "n4096_non_cifar_fp64_residual_gate": residual_gate,
        "n4096_non_cifar_g3b_ambiguity_gate": ambiguity_gate,
        "cross_dataset_fp64_residual_gate_pass": cross_dataset_residual_pass,
        "cross_dataset_g3b_ambiguity_gate_pass": cross_dataset_ambiguity_pass,
        "opportunity_gate_pass": opportunity_gate_pass,
        "generic_timing_implementation_allowed": opportunity_gate_pass,
        "performance_claim_allowed": False,
        "cells": sorted(
            cells,
            key=lambda cell: (
                cell["dataset_id"],
                int(cell["n"]),
                int(cell["target_average_directed_nonself_degree"]),
            ),
        ),
        "evidence": evidence,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G4B_OPPORTUNITY.md"),
    }
    atomic_json(RESULT, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if opportunity_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
