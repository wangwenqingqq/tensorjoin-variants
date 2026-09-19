"""CPU-only, post-campaign audit; does not replace the frozen estimator."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
OLD = ROOT / "two_gate_campaign_20260908"
SCHEDULES = ["original", "clustered", "interleaved"]
PLANS = ["F8", "F16"]
HASH = "13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495"
COUNTS = {"F8": [1278504, 1827007, 713664, 1111609, 1734, 1993039],
          "F16": [1950364, 86029, 41804, 42491, 1734, 1993039]}

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def gm(a):
    return float(np.exp(np.mean(np.log(a))))

checked = {}
for name in ["timing_freeze_a0.json", "source_freeze_a0.json"]:
    raw = read(P / "artifacts" / name)
    entries = raw.get("files", raw)
    for rel, value in entries.items():
        h = value["sha256"] if isinstance(value, dict) else value
        assert sha(P / rel) == h, rel
    checked[name] = len(entries)
admission = read(P / "artifacts/admission_a0.json")
assert admission["pass"]
for rel, h in admission["evidence"].items():
    assert sha(P / rel) == h, rel

census = read(P / "results/census_a0.json")
assert census["pass"] and census["manipulation_pass"]
for m in PLANS:
    arrays = [np.load(P / "artifacts" / f"{s}_{m}_census.npz") for s in SCHEDULES]
    for key in ["u1_logical", "u2_logical", "hist1", "hist2"]:
        assert all(np.array_equal(arrays[0][key], a[key]) for a in arrays[1:]), (m, key)
    for a in arrays:
        assert len(a["u1_logical"]) == COUNTS[m][1]
        assert len(a["u2_logical"]) == 1734
        assert len(a["batch_u1"]) == 108
        assert a["batch_u1"].sum() == COUNTS[m][1]
        assert a["batch_u2"].sum() == 1734
for row in census["rows"]:
    assert row["output_hash"] == HASH and row["stage_counts"] == COUNTS[row["method"]]

safety = []
for tool in ["memcheck", "initcheck", "synccheck", "racecheck"]:
    a = read(P / "results" / f"{tool}_a0.json")
    g = read(P / "results" / f"{tool}_a0_guard.json")
    log = (P / "raw" / f"{tool}_a0.log").read_text()
    assert a["pass"] and g["pass"] and g["exit_code"] == 0
    assert len(a["rows"]) == 6 and all(r["pass"] for r in a["rows"])
    assert "ERROR SUMMARY: 0 errors" in log or "RACECHECK SUMMARY: 0 hazards" in log
    safety.append({"tool": tool, "pid": a["pid"], "selected_tiles": sorted({r["tiles"] for r in a["rows"]}), "pass": True})

runs = []
guards = []
for i in range(6):
    a = read(P / "results" / f"p{i:02d}_a0.json")
    g = read(P / "results" / f"p{i:02d}_a0_guard.json")
    assert a["pass"] and g["pass"] and g["exit_code"] == 0
    assert a["index"] == i and a["pid"] == g["pid"]
    assert admission["time"] < g["started"]
    assert len(a["samples"]) == 102
    for s in a["samples"]:
        assert s["hash"] == HASH and s["count"] == 3926078
        assert s["stage_counts"] == COUNTS[s["method"]] and s["batches"] == 108
        assert 0 < s["inner_seconds"] < s["seconds"]
    ct = Counter((s["layout"], s["method"], s["phase"]) for s in a["samples"])
    assert all(ct[s, m, ph] == n for s in SCHEDULES for m in PLANS
               for ph, n in [("warmup", 2), ("retained", 7), ("repeat8", 8)])
    if runs:
        assert a["identities"] == runs[0]["identities"]
        assert a["compiled"] == runs[0]["compiled"]
    runs.append(a)
    guards.append({k: g[k] for k in ["pid", "pass", "started", "ended", "max_device_used_mib"]})

for rel, h in runs[0]["identities"].items():
    assert sha(ROOT / rel) == h, rel
for key, entry in runs[0]["compiled"].items():
    for ext in ["cubin", "ptx", "llir", "ttgir", "ttir"]:
        assert sha(OLD / "artifacts/g2_compiled_a0" / f"{key}.{ext}") == entry[ext]

analysis = read(P / "results/analysis_a0.json")
assert analysis["calls"] == 612 and not analysis["replacement_runs"]
assert not analysis["existence_gate_pass"] and not analysis["practical_selector_gate_pass"]
rng = np.random.default_rng(20260908)
groups = [np.array([0, 2, 4]), np.array([1, 3, 5])]
ix = np.concatenate([rng.choice(g, (10000, 3), replace=True) for g in groups], axis=1)
diagnostics = {}
for phase in ["retained", "repeat8"]:
    t = np.zeros((6, 3, 2))
    for i, a in enumerate(runs):
        for j, sch in enumerate(SCHEDULES):
            for k, m in enumerate(PLANS):
                ss = [s for s in a["samples"] if (s["layout"], s["method"], s["phase"]) == (sch, m, phase)]
                t[i, j, k] = (np.median if phase == "retained" else np.sum)([s["seconds"] for s in ss])
    for j, row in enumerate(analysis["phases"][phase]["layouts"]):
        r = t[:, j, 1] / t[:, j, 0]
        ci = np.quantile(np.exp(np.mean(np.log(r[ix]), axis=1)), [.025, .975])
        assert np.allclose(r, row["process_ratios"], rtol=1e-12)
        assert np.allclose(ci, row["CI95"], rtol=1e-12)
        assert abs(gm(r) - row["F16_over_F8_geometric"]) < 1e-12
    h = np.min(t.sum(axis=1), axis=1) / np.min(t, axis=2).sum(axis=1)
    assert np.allclose(h, analysis["phases"][phase]["oracle_headroom"]["process_values"])
    extra = [s["seconds"] - s["inner_seconds"] for a in runs for s in a["samples"] if s["phase"] == phase]
    diagnostics[phase] = {"common_outer_increment_p10_median_p90_seconds": np.quantile(extra, [.1, .5, .9]).tolist()}

out = {
    "pass": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "cpu_audit_numpy": np.__version__, "source_freezes_checked": checked,
    "calls_checked": 612, "full_output_count_each": 3926078, "full_output_sha256": HASH,
    "all_uncertainty_sets_and_tile_histograms_equal_within_plan": True,
    "runtime_compiled_identities_equal_across_processes_and_retained_files": True,
    "frozen_estimator_independently_recomputed": True,
    "safety": safety, "timing_guards": guards, "diagnostics": diagnostics,
    "order_label_note": "method_order / 8_16 / 16_8 labels denote the primary-phase order group; actual method order reverses in repeat8. No group is discarded.",
    "scope_note": "Post-run CPU audit, not an additional GPU measurement, sanitizer run, or universal numerical proof."
}
(P / "results/closure_audit_a0.json").open("x").write(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
