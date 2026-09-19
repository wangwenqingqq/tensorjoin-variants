#!/usr/bin/env python3
"""Summarize all frozen G8 processes without choosing a favorable estimator."""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("g8_timing_p0_a0", "g8_timing_p1_a0")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    processes = []
    for label in LABELS:
        result = json.loads((ROOT / f"results/{label}.json").read_text())
        manifest = json.loads((ROOT / f"results/{label}_manifest.json").read_text())
        assert manifest["admitted"] and result["correctness_pass"]
        processes.append(result)
    safety = {}
    for label in ("g8_check_a2", "g8_memcheck_a0", "g8_synccheck_a0"):
        manifest = json.loads((ROOT / f"results/{label}_manifest.json").read_text())
        assert manifest["admitted"]
        log = (ROOT / f"raw/{label}.log").read_text()
        safety[label] = dict(admitted=True, log_sha256=sha(ROOT / f"raw/{label}.log"))
        if "check_a2" not in label:
            assert "ERROR SUMMARY: 0 errors" in log
    names = [c["name"] for c in processes[0]["cases"]]
    cases = []
    for name in names:
        per_process = []
        for process in processes:
            case = next(c for c in process["cases"] if c["name"] == name)
            rounds = {r: {} for r in range(40)}
            for obs in case["samples"]:
                assert obs["method"] not in rounds[obs["round"]]
                rounds[obs["round"]][obs["method"]] = obs["ms"]
            assert all(set(r) == {"P1", "P16", "T4x4"} for r in rounds.values())
            values = {m: np.array([r[m] for r in rounds.values()]) for m in ("P1", "P16", "T4x4")}
            paired = values["P16"] / values["T4x4"]
            p1p16 = values["P1"] / values["P16"]
            log_median = float(np.median(np.log(paired)))
            per_process.append(dict(label=process["label"], reverse=process["reverse"],
                paired_log_median=log_median, primary_p16_over_tile=math.exp(log_median),
                marginal_median_p16_over_tile=float(np.median(values["P16"]) / np.median(values["T4x4"])),
                paired_ratio_arithmetic_mean=float(np.mean(paired)),
                paired_ratio_geometric_mean=float(np.exp(np.mean(np.log(paired)))),
                paired_tile_round_wins=int(np.count_nonzero(paired > 1)),
                paired_p1_over_p16=float(np.exp(np.median(np.log(p1p16)))),
                methods={m: dict(zip(("p10_ms", "median_ms", "p90_ms"),
                                    np.quantile(v, [.1, .5, .9]).tolist())) for m, v in values.items()},
                sustained_ms_per_call=case["sustained_ms_per_call"],
                sustained_p16_over_tile=case["sustained_ms_per_call"]["P16"] / case["sustained_ms_per_call"]["T4x4"],
                position_medians={m: {str(pos): float(np.median([s["ms"] for s in case["samples"]
                    if s["method"] == m and s["position"] == pos])) for pos in range(3)} for m in values}))
        logs = np.array([p["paired_log_median"] for p in per_process])
        mean = float(logs.mean())
        half = 12.706204736432095 * float(logs.std(ddof=1) / math.sqrt(2))
        real = not name.startswith("synthetic")
        cases.append(dict(name=name, real=real,
            active_tiles=case["active_tiles"], padding_amplification=case["padding_amplification"],
            logical_state_sha256=case["correctness_before"]["P1"]["logical_state_sha256"],
            counts_reject_accept_uncertain=case["correctness_before"]["P1"]["counts"],
            layout_host_seconds=[next(c for c in p["cases"] if c["name"] == name)["layout_host_seconds"] for p in processes],
            planning_host_seconds=[next(c for c in p["cases"] if c["name"] == name)["queue_and_row_copy_host_seconds"] for p in processes],
            processes=per_process, primary_p16_over_tile=math.exp(mean),
            diagnostic_process_t95_interval=[math.exp(mean-half), math.exp(mean+half)],
            interval_caveat="Only two processes; normal log-ratio assumption is untested; not a formal confidence gate.",
            tile_wins_both_processes=sum(p["primary_p16_over_tile"] > 1 for p in per_process) == 2,
            qualifies_tile_win=all(p["primary_p16_over_tile"] >= 1.15 and p["sustained_p16_over_tile"] > 1 for p in per_process),
            qualifies_pair_win=all(p["primary_p16_over_tile"] <= 1/1.15 and p["sustained_p16_over_tile"] < 1 for p in per_process)))
    real = [c for c in cases if c["real"]]
    synthetic = [c for c in cases if not c["real"]]
    crossover = lambda cs: any(c["qualifies_tile_win"] for c in cs) and any(c["qualifies_pair_win"] for c in cs)
    compiled = processes[0]["compiled"]
    all_labels = ("g8_check_a2", "g8_memcheck_a0", "g8_synccheck_a0") + LABELS
    for label in all_labels:
        p = json.loads((ROOT / f"results/{label}.json").read_text())
        for method in compiled:
            assert p["compiled"][method]["cubin_sha256"] == compiled[method]["cubin_sha256"]
    resources = {}
    for method, data in compiled.items():
        sass_path = ROOT / f"artifacts/g8_check_a2/{method}.sass"
        sass = sass_path.read_text()
        instructions = re.findall(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9]+)(?:[.\s;])", sass)
        histogram = {op: instructions.count(op) for op in sorted(set(instructions))}
        resources[method] = dict(n_regs=data["n_regs"], n_spills=data["n_spills"],
            shared_bytes=data["metadata"]["shared"], cubin_sha256=data["cubin_sha256"],
            sass_sha256=sha(sass_path), static_opcode_histogram=histogram,
            static_tensor_instructions=sum(n for op,n in histogram.items() if "MMA" in op),
            resource_text=(ROOT / f"artifacts/g8_check_a2/{method}.resources.txt").read_text())
    result = dict(experiment_id="tensorjoin_20260904_g8_refinement_layout_g2a4096",
        status="partial_component_evidence", cases=cases, safety=safety, resources=resources,
        same_compiled_binaries_across_safety_and_timing=True,
        real_data_crossover_pass=crossover(real), synthetic_crossover_pass=crossover(synthetic),
        online_router_authorized=crossover(real),
        denominator="CUDA-event blocks of 20 Python-submitted component calls; resident GPU; no Graph; includes possible submission gaps",
        excluded="layout/queue planning, first pass, FP64 repair, output compaction/export, ingestion",
        static_ledger_caveat="Static instruction sites are not dynamic executed work or a runtime traffic measurement.")
    out = ROOT / "results/g8_layout_summary_a0.json"
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("real crossover:", result["real_data_crossover_pass"], "synthetic crossover:", result["synthetic_crossover_pass"])
    for c in cases:
        print(c["name"], "P16/T4", c["primary_p16_over_tile"],
              "processes", [p["primary_p16_over_tile"] for p in c["processes"]],
              "sustained", [p["sustained_p16_over_tile"] for p in c["processes"]])


if __name__ == "__main__":
    main()
