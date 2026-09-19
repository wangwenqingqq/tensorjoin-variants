#!/usr/bin/env python3
"""Fail-closed single-slot G9 runner using existing physical GPU1 locks."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from run_g2b_public_screen import (PYTHON, compute_rows, gpu_rows,
    is_descendant_or_self, preflight_text, terminate_own_group)
from g2b_public_common import PROJECT, sha256_file

UUID = "GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--mode", choices=("check", "timing"), required=True)
    p.add_argument("--sanitizer", choices=("memcheck", "synccheck"))
    p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    assert platform.node() == "gpu-host-8"
    assert a.label.startswith("g9_") and all(c.isalnum() or c == "_" for c in a.label)
    paths = {key: PROJECT / value for key, value in {
        "log": f"raw/{a.label}.log", "preflight": f"raw/{a.label}_preflight.log",
        "occupancy": f"raw/{a.label}_occupancy.jsonl",
        "manifest": f"results/{a.label}_manifest.json",
        "result": f"results/{a.label}.json", "cache": f"artifacts/{a.label}_cache"}.items()}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    command = [PYTHON, str(PROJECT / "src/run_g9_tc_refinement.py"),
               "--label", a.label, "--mode", a.mode]
    if a.reverse:
        command.append("--reverse")
    if a.sanitizer:
        assert a.mode == "check"
        command = ["/usr/local/bin/compute-sanitizer", "--tool", a.sanitizer,
                   "--target-processes", "all", "--error-exitcode", "86"] + command
    manifest = dict(label=a.label, command=command, start_unix=time.time(),
                    gpu_uuid=UUID, status="unvalidated", launched=False,
                    foreign_rows=[], runner_sha256=sha256_file(Path(__file__)))
    locks = []
    try:
        for name in (".tensorjoin_gpu1_campaign.lock", ".tensorjoin_g6_gpu1.lock"):
            f = open(Path("@TENSORJOIN_ROOT@") / name, "a+")
            locks.append(f)
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths["preflight"].write_text(preflight_text(UUID))
        assert gpu_rows()[1]["uuid"] == UUID
        with paths["occupancy"].open("x") as monitor:
            for i in range(6):
                rows = compute_rows(UUID)
                gpu = gpu_rows()[1]
                monitor.write(json.dumps(dict(phase="quiescence", time=time.time(),
                                              rows=rows, gpu=gpu)) + "\n")
                monitor.flush()
                assert not rows and int(gpu["memory.used"]) <= 32 and int(gpu["utilization.gpu"]) == 0, "Target not quiescent"
                if i < 5:
                    time.sleep(1)
            paths["cache"].mkdir()
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=UUID, TRITON_CACHE_DIR=str(paths["cache"]),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            with paths["log"].open("x") as log:
                log.write("COMMAND " + json.dumps(command) + "\n")
                log.flush()
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                manifest.update(launched=True, child_pid=child.pid)
                own = set()
                while child.poll() is None:
                    rows = compute_rows(UUID)
                    foreign = []
                    for row in rows:
                        pid = row["pid"]
                        if pid in own or is_descendant_or_self(pid, child.pid):
                            own.add(pid)
                        else:
                            foreign.append(row)
                    monitor.write(json.dumps(dict(phase="run", time=time.time(), rows=rows,
                        gpu=gpu_rows()[1], own_pids=sorted(own), foreign=foreign)) + "\n")
                    monitor.flush()
                    if foreign:
                        manifest["foreign_rows"].extend(foreign)
                        terminate_own_group(child)
                        break
                    time.sleep(.25)
                manifest.update(returncode=child.wait(), own_pids=sorted(own),
                                postflight_compute_rows=compute_rows(UUID))
        result = json.loads(paths["result"].read_text()) if paths["result"].exists() else {}
        clean = (manifest["returncode"] == 0 and not manifest["foreign_rows"] and
                 not manifest["postflight_compute_rows"] and result.get("correctness_pass", False))
        manifest["status"] = "clean_component_slot" if clean else "failed_or_incomplete"
        manifest["admitted"] = bool(clean)
    except BaseException as error:
        manifest.update(status="failed_or_incomplete", admitted=False, error=repr(error))
        raise
    finally:
        manifest["end_unix"] = time.time()
        manifest["artifacts"] = {k: {"path": str(v.relative_to(PROJECT)), "sha256": sha256_file(v)}
            for k, v in paths.items() if v.is_file() and k != "manifest"}
        paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for f in reversed(locks):
            f.close()
        print(json.dumps(manifest, indent=2), flush=True)
    return 0 if manifest.get("admitted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
