#!/usr/bin/env python3
"""Extract figure/table numbers from frozen TensorJoin evidence JSON files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper_assets/data/paper_numbers.json"
SOURCES = {
    "g2b": ROOT / "results/g2b_public_formal_summary.json",
    "g4c": ROOT / "results/g4c_formal_summary.json",
    "g5_p1": ROOT / "results/g5_compatibility_tensorjoin_p1_r1_a0.json",
    "g5_p4": ROOT / "results/g5_p4_screen_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(SOURCES[name].read_text(encoding="utf-8"))


def main() -> None:
    g2b = load("g2b")
    g4c = load("g4c")
    g5 = load("g5_p1")
    g5_p4 = load("g5_p4")

    assert g2b["formal_acceptance_pass"] is True
    assert g2b["all_exact_outputs_pass"] is True
    assert g4c["formal_gate_pass"] is True
    assert g5["correctness"]["exact_contract_pass"] is True
    assert g5_p4["screen_pass"] is False
    assert g5_p4["formal_campaign_allowed"] is False

    n, d = (int(value) for value in g5["shape"])
    upper_pairs = n * (n + 1) // 2
    direct_accept = int(g5["g3b_direct_accept_upper_pairs"])
    ambiguous = int(g5["g3b_ambiguous_upper_pairs"])
    fp32_accept = int(g5["fp32_direct_accept_upper_pairs"])
    fp32_reject = int(g5["fp32_direct_reject_upper_pairs"])
    fp64_total = int(g5["fp64_refined_upper_pairs"])
    accepted_upper = int(g5["accepted_upper_pairs"])
    direct_reject = upper_pairs - direct_accept - ambiguous
    fp64_accept = accepted_upper - direct_accept - fp32_accept
    fp64_reject = fp64_total - fp64_accept
    assert direct_reject >= 0
    assert fp32_accept + fp32_reject + fp64_total == ambiguous
    assert fp64_accept >= 0 and fp64_reject >= 0

    external = {}
    for method, values in g2b["marginal_distributions_seconds"].items():
        external[method] = {
            "median_s": float(values["median"]),
            "p10_s": float(values["p10"]),
            "p90_s": float(values["p90"]),
            "exact": method != "fasted",
        }
    external["fasted"]["false_negatives"] = int(
        g2b["fasted_context_only"]["exact_only_pairs"][0]
    )
    external["fasted"]["false_positives"] = int(
        g2b["fasted_context_only"]["approximate_only_pairs"][0]
    )
    for keeper in ("mistic", "gds"):
        speed = g2b["paired_tensorjoin_speedup_by_exact_keeper"][keeper]
        external[keeper]["paired_speedup"] = float(speed["paired_geometric_mean"])
        external[keeper]["ci95"] = [float(speed["ci95_lower"]), float(speed["ci95_upper"])]

    dynamic = {}
    for key, label in (
        ("sift128", "SIFT-128"),
        ("cifar_gist512", "CIFAR-GIST-512"),
        ("fashion784", "Fashion-MNIST-784"),
    ):
        values = g4c["datasets"][key]
        dynamic[label] = {
            "d": int({"sift128": 128, "cifar_gist512": 512, "fashion784": 784}[key]),
            "speedup": float(values["paired_geometric_mean_median_speedup"]),
            "ci95": [
                float(values["paired_process_bootstrap"]["ci95_lower"]),
                float(values["paired_process_bootstrap"]["ci95_upper"]),
            ],
            "process_wins": int(values["process_median_wins"]),
            "candidate_median_us": float(
                values["marginal_raw_sample_percentiles_us"]["candidate_three_stage"]["median_us"]
            ),
            "keeper_median_us": float(
                values["marginal_raw_sample_percentiles_us"]["keeper_two_stage"]["median_us"]
            ),
        }

    output = {
        "schema": "tensorjoin-paper-numbers-v1",
        "source_sha256": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in SOURCES.items()
        },
        "g5_precision_funnel": {
            "n": n,
            "d": d,
            "epsilon": float(g5["epsilon"]),
            "upper_pairs": upper_pairs,
            "scheduled_tiles": int(g5["scheduled_tiles"]),
            "tiles_per_batch": int(g5["tiles_per_batch"]),
            "batches": int(g5["batch_count"]),
            "stage1_direct_accept": direct_accept,
            "stage1_direct_reject": direct_reject,
            "stage1_ambiguous": ambiguous,
            "stage1_ambiguous_percent": 100.0 * ambiguous / upper_pairs,
            "stage2_direct_accept": fp32_accept,
            "stage2_direct_reject": fp32_reject,
            "stage3_fp64": fp64_total,
            "stage3_fraction_of_stage2_percent": 100.0 * fp64_total / ambiguous,
            "stage3_fraction_of_all_percent": 100.0 * fp64_total / upper_pairs,
            "stage3_accept": fp64_accept,
            "stage3_reject": fp64_reject,
            "accepted_upper": accepted_upper,
            "directed_output": int(g5["correctness"]["canonical_pair_count"]),
            "directed_output_sha256": g5["correctness"]["canonical_raw_u64_sha256"],
        },
        "g2b_external": {
            "scope": g2b["scope"],
            "methods": external,
            "exact_output_count": int(g5["correctness"]["canonical_pair_count"]),
        },
        "g4c_dynamic": dynamic,
        "g5_boundary": {
            "samples_seconds_by_round": g5_p4["samples_seconds_by_round"],
            "paired_speedups": g5_p4["paired_speedups_vs_faster_exact_keeper"],
            "paired_geometric_mean": g5_p4["paired_geometric_mean_speedup"],
            "screen_pass": g5_p4["screen_pass"],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
