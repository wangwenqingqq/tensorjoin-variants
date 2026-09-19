#!/usr/bin/env python3
"""Verify frozen G9 receipts without importing CUDA or rerunning experiments."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    slots = []
    for path in sorted(Path("results").glob("g9_*_manifest.json")):
        manifest = json.loads(path.read_text())
        for artifact in manifest["artifacts"].values():
            assert digest(artifact["path"]) == artifact["sha256"], artifact
        if manifest.get("admitted"):
            assert manifest["returncode"] == 0
            assert not manifest["foreign_rows"]
            assert not manifest["postflight_compute_rows"]
            runner = ("src/run_g9_graph_isolated.py" if "graph" in manifest["label"]
                      else "src/run_g9_isolated.py")
            assert digest(runner) == manifest["runner_sha256"], runner
            result = json.loads(Path(manifest["artifacts"]["result"]["path"]).read_text())
            assert result["correctness_pass"]
            for source, sha in result["source_hashes"].items():
                assert digest(source) == sha, source
            for sha, compiled in result["compiled"].items():
                base = Path("artifacts") / manifest["label"] / sha
                assert digest(base.with_suffix(".cubin")) == sha, base
                assert digest(base.with_suffix(".ptx")) == compiled["ptx_sha256"], base
            for key, document in {
                "protocol_sha256": "PROTOCOL_G9_TC_REFINEMENT_A0.md",
                "numerics_sha256": "NUMERICS_G9_INT14.md",
                "gpu_revision_sha256": "PROTOCOL_G9_GPU7_R1.md",
                "graph_protocol_sha256": "PROTOCOL_G9_GRAPH_DIAGNOSTIC_R2.md",
            }.items():
                if key in result:
                    assert digest(document) == result[key], document
        slots.append({"label": manifest["label"],
                      "admitted": manifest.get("admitted", False),
                      "launched": manifest.get("launched", False),
                      "artifact_receipts_verified": len(manifest["artifacts"])})
    assert len(slots) == 13, len(slots)
    assert sum(slot["admitted"] for slot in slots) == 10
    print(json.dumps({"pass": True, "slots": slots}, indent=2))


if __name__ == "__main__":
    main()
