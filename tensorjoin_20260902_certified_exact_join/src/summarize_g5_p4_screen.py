#!/usr/bin/env python3
"""Apply the predeclared G5 P4 external-system cheap-screen gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "results/g5_p4_screen_manifest.json"
SUMMARY = PROJECT / "results/g5_p4_screen_summary.json"
P1 = PROJECT / "results/g5_compatibility_tensorjoin_p1_r1_a0.json"
EXPECTED_P1_SHA256 = "7fcdd2786db413ee9f56347c27eceb9542e41e8fc644e9dd3a55d9ec59bff2a5"
EXPECTED_CANDIDATE_RUNNER = "a8653469a0e5bae82b64cd3d9a82c476df741783735c1b9d699e2f27b2db13a5"
EXPECTED_GDS_LIBRARY = "e0d31f51a8a2ed844b988b2c2b1c08ed9757a55dfceda338c787d999b34b54bb"
EXPECTED_MISTIC_BINARY = "20739abefe2fc21cfb93ec70b0c9cddbde20b76dc896243c72336981c402326a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
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
    if sha256_file(P1) != EXPECTED_P1_SHA256:
        raise RuntimeError("G5 P1 result hash mismatch")
    p1 = json.loads(P1.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest["records"]
    by_round: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in records:
        by_round[int(record["round"])][str(record["method"])] = record
    complete = bool(
        len(records) == 6
        and set(by_round) == {0, 1}
        and all(set(methods) == {"gds", "mistic", "tensorjoin"} for methods in by_round.values())
    )
    exact_all = bool(
        complete
        and all(
            record.get("guard_admitted")
            and record.get("exact_contract_pass")
            and not record.get("foreign_rows")
            and not record.get("postflight_compute_rows")
            for record in records
        )
    )
    samples = {
        method: [float(by_round[r][method]["public_seconds"]) for r in (0, 1)]
        for method in ("gds", "mistic", "tensorjoin")
    }
    keeper_medians = {method: float(np.median(samples[method])) for method in ("gds", "mistic")}
    faster_keeper = min(keeper_medians, key=keeper_medians.get)
    paired_speedups = [
        samples[faster_keeper][r] / samples["tensorjoin"][r] for r in (0, 1)
    ]
    paired_geomean = math.exp(float(np.mean(np.log(paired_speedups))))

    p1_counts = {
        key: p1[key]
        for key in (
            "g3b_direct_accept_upper_pairs",
            "g3b_ambiguous_upper_pairs",
            "fp32_direct_accept_upper_pairs",
            "fp32_direct_reject_upper_pairs",
            "fp32_bitwise_equal_upper_pairs",
            "fp64_refined_upper_pairs",
            "accepted_upper_pairs",
            "overflow_events",
        )
    }
    p1_cache = p1["selected_cache_artifacts_after_timer"]
    candidate_records = [record for record in records if record["method"] == "tensorjoin"]
    candidate_identity = bool(
        len(candidate_records) == 2
        and all(
            record.get("candidate_runner_sha256") == EXPECTED_CANDIDATE_RUNNER
            and record.get("candidate_stage_counts") == p1_counts
            and all(
                record["candidate_cache_artifacts"][name]["cubin_sha256"]
                == p1_cache[name]["cubin_sha256"]
                and record["candidate_cache_artifacts"][name]["ptx_sha256"]
                == p1_cache[name]["ptx_sha256"]
                for name in p1_cache
            )
            for record in candidate_records
        )
    )
    external_identity = bool(
        all(
            record.get("library_sha256") == EXPECTED_GDS_LIBRARY
            for record in records
            if record["method"] == "gds"
        )
        and all(
            record.get("binary_sha256") == EXPECTED_MISTIC_BINARY
            for record in records
            if record["method"] == "mistic"
        )
    )
    per_round_gate = min(paired_speedups) >= 1.50
    geomean_gate = paired_geomean >= 1.60
    screen_pass = bool(
        complete
        and exact_all
        and candidate_identity
        and external_identity
        and per_round_gate
        and geomean_gate
    )
    result = {
        "experiment_id": "tensorjoin_20260903_g5_unified_public_cifar60k",
        "measurement_status": "diagnostic_public_denominator_cheap_screen",
        "manifest_sha256": sha256_file(MANIFEST),
        "complete_two_rounds": complete,
        "all_six_exact_and_isolated": exact_all,
        "samples_seconds_by_round": samples,
        "marginal_medians_seconds": {
            method: float(np.median(values)) for method, values in samples.items()
        },
        "faster_exact_keeper_by_marginal_median": faster_keeper,
        "paired_speedups_vs_faster_exact_keeper": paired_speedups,
        "minimum_paired_speedup": min(paired_speedups),
        "paired_geometric_mean_speedup": paired_geomean,
        "per_round_gate_1_50": per_round_gate,
        "paired_geomean_gate_1_60": geomean_gate,
        "candidate_same_source_stage_counts_and_cubins_as_p1": candidate_identity,
        "external_binary_identity_pass": external_identity,
        "screen_pass": screen_pass,
        "formal_campaign_allowed": screen_pass,
        "performance_claim_allowed": False,
        "next_action": (
            "freeze_and_run_eight_round_G5_formal_campaign"
            if screen_pass
            else "stop_G5_and_preserve_negative_evidence"
        ),
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROJECT / "PROTOCOL_G5_P4_SCREEN.md"),
    }
    atomic_json(SUMMARY, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if screen_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

