#!/usr/bin/env python3
"""Run one locked upstream sample; do not infer exactness from a count."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603"
LOCK = Path("@TENSORJOIN_ROOT@/.tensorjoin_gpu7_campaign.lock")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "-i", UUID,
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    index, uuid, name, memory, utilization = [x.strip() for x in output.split(",")]
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits"], text=True)
    return dict(index=int(index), uuid=uuid, name=name, memory_mib=int(memory),
                utilization_percent=int(utilization),
                compute_apps=[x for x in apps.splitlines() if x.startswith(UUID)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="a0")
    args = parser.parse_args()
    if not re.fullmatch(r"a[0-9]+", args.attempt):
        raise ValueError("attempt must be a nonnegative integer prefixed by a")
    run_dir = ROOT / f"raw/g7_rthiss_smoke_{args.attempt}"
    run_dir.mkdir(exist_ok=False)
    result = ROOT / f"results/g7_rthiss_smoke_{args.attempt}.json"
    if result.exists():
        raise FileExistsError(result)
    build = ROOT / "adapters/rthiss_g7_a0/build_sample_sm120_a1"
    binary = build / "RT-HiSS"
    sample = ROOT / "adapters/rthiss_g7_a0/sampleDataset/iono_57_1000.txt"
    command = [str(binary), str(sample), "0.01", "highest"]
    record = dict(
        stage="runtime_smoke_only", status="unvalidated", exactness="not_tested",
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        upstream_commit="a42fc69cc4b602dc83071b029d185a41e69a04bd",
        owl_commit="c7c3a3ea35b17b5c096a3802ba74b9d8b4e2772a",
        n=1000, d=2, epsilon_text="0.01", command=command, cwd=str(run_dir),
        binary_sha256=digest(binary), sample_sha256=digest(sample),
        public_time_seconds=None, output_pair_ids_exported=False,
        explanation="Count-only runtime smoke, not oracle equality or latency evidence.")
    lib = build / "OWL/owl/libowl.so"
    record["libowl_sha256"] = digest(lib)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = snapshot()
            record["gpu_before"] = before
            if (before["index"] != 7 or before["uuid"] != UUID or
                    before["memory_mib"] >= 128 or before["utilization_percent"] != 0 or
                    before["compute_apps"]):
                raise RuntimeError("GPU7 isolation preflight failed; no process touched")
            if record["sample_sha256"] != "5c3fa39b0ca0f46c37b7c095e498461b1898164b2f42f9d08af432e00374f49e":
                raise RuntimeError("sample changed")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS="8")
            log_path = run_dir / "stdout_stderr.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=run_dir, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record["pid"] = process.pid
                result.write_text(json.dumps(record, indent=2) + "\n")
                start = time.monotonic()
                try:
                    rc = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record["timeout"] = True
                    rc = process.returncode
                record["operational_process_elapsed_seconds"] = time.monotonic() - start
                record["returncode"] = rc
            text = log_path.read_text(errors="replace")
            upstream_result = run_dir / "results.txt"
            count = None
            if upstream_result.exists():
                match = re.search(r"Total identified neighbors:\s*(\d+)", upstream_result.read_text())
                count = int(match.group(1)) if match else None
            record["reported_neighbor_count"] = count
            record["error_markers"] = re.findall(
                r"[^\n]*(?:OUT OF BOUNDS|CUDA error:|terminate called|Error:)[^\n]*", text)
            passed = rc == 0 and count is not None and not record["error_markers"]
            record["status"] = "sample_smoke_passed" if passed else "runtime_smoke_failed"
            record["gpu_after"] = snapshot()
        except Exception as exc:
            record["error"] = repr(exc)
            record["status"] = "unvalidated"
        finally:
            record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "sample_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
