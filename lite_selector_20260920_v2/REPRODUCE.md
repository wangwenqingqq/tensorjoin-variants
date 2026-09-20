# Reproduce without overwriting the retained attempt

CPU evidence replay needs NumPy. GPU replay additionally needs the exact recorded
CUDA/Python environment, a personally verified idle SM120 GPU, and authorized
access to the original G17 fixture files. Set `TENSORJOIN_SOURCE` to the certified
exact-join source tree described in `../precision_format_routing/README.md`.
The real dataset is not redistributed. Changing architecture/compiler requires
re-deriving the undocumented format fields and the FP64 CPU addition graph.

## 1. Fresh destination and immutable source copy

Run from the repository root. The destination must not already exist. Do not use
a retained results directory as the replay destination.

```sh
export REPLAY_ROOT=/path/to/new-empty-attempt
export PYTHON=/path/to/recorded-cuda-python
export CUDA_TOOLS=/path/to/cuda-13.1/bin
export TENSORJOIN_SOURCE=/path/to/authorized/certified-exact-join
export GPU=7  # example only: recheck current owners/processes before running
export RUN=lite_cpu_order_replay
mkdir "$REPLAY_ROOT"
mkdir "$REPLAY_ROOT/precision_format_routing"
cp -R precision_format_routing/src "$REPLAY_ROOT/precision_format_routing/"
export R="$REPLAY_ROOT/lite_selector_20260920_v2"
mkdir "$R"
cp -R lite_selector_20260920_v2/src "$R/"
cp lite_selector_20260920_v2/CONTRACT.yaml lite_selector_20260920_v2/workloads.jsonl "$R/"
mkdir -p "$REPLAY_ROOT/lite_selector_20260920/results"
cp lite_selector_20260920/results/g0_failure_subset_v1.npy \
   lite_selector_20260920/results/g0_failure_diagnosis_v1.json \
   "$REPLAY_ROOT/lite_selector_20260920/results/"
```

The retained guard pins ptxas to `/usr/local/cuda-13.1/bin/ptxas`. That location
must exist and match the recorded toolchain. If the installation differs, record
an explicit new contract/source amendment rather than silently bypassing it.

## 2. Regenerate synthetic parents, verify every input and threshold

This is CPU-only input construction. It does not expose any held-out GPU result.
All 96 configurations must reproduce; do not replace failed hashes or thresholds.

```sh
"$PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
r = Path(os.environ['R']); sys.path.insert(0, str(r/'src'))
import workloads as w
assert np.__version__ == '2.3.1'
rows = [json.loads(s) for s in (r/'workloads.jsonl').read_text().splitlines()]
(r/'data').mkdir()
for parent in sorted({c['parent_id'] for c in rows}):
    group = [c for c in rows if c['parent_id'] == parent]
    x = w.generate(group[0]['family'], group[0]['seed'])
    assert w.digest(x) == group[0]['parent_sha256']
    for n in (1024,4096):
        thresholds, pair_hash, distance_hash = w.thresholds(x[:n])
        for c in [c for c in group if c['n'] == n]:
            assert w.digest(x[:n]) == c['input_sha256']
            assert thresholds[c['target_k']].hex() == c['threshold_hex']
            assert pair_hash == c['sample_pair_sha256']
            assert distance_hash == c['sample_distance_sha256']
    np.save(r/'data'/(parent+'.npy'), x, allow_pickle=False)
PY
```

## 3. Preflight and guarded, serial G0 -> G1

Inspect hostname, driver/toolkit/library versions, active users, all GPU owners,
selected-GPU UUID, memory and utilization. Do not infer availability from low
utilization. The guard performs five idle samples and monitors only owned child
processes on the selected GPU. It never stops foreign work or changes GPU clocks.
Shared-host CPU activity remains a limitation. The recorded run used GPU 7;
this example does not authorize disrupting a current owner.

```sh
nvidia-smi
"$CUDA_TOOLS/nvcc" --version
"$PYTHON" "$REPLAY_ROOT/precision_format_routing/src/guard.py" \
  --gpu "$GPU" --label "$RUN" --timeout 3600 -- bash -c '
set -e
for mode in reference probe regression validate; do
  "$PYTHON" "$R/src/run.py" --mode "$mode" --label "g0_${mode}_v2"
done
for tool in memcheck racecheck initcheck synccheck; do
  "$CUDA_TOOLS/compute-sanitizer" --tool "$tool" --error-exitcode 3 \
    --log-file "$R/results/g0_${tool}_v2.sanitizer.log" \
    "$PYTHON" "$R/src/run.py" --mode sanitizer --label "g0_${tool}_v2"
done
"$PYTHON" "$R/src/run.py" --mode stress --label g0_stress_v2
for p in 0 1 2 3; do
  "$PYTHON" "$R/src/run.py" --mode bench --label "g1_bench_$p" --process "$p"
done
'
```

Any nonzero stage stops the pipeline; preserve errors and do not continue to
benchmarks. Stop/rollback is termination of this guard's owned command group;
there is no production-path replacement to undo. Never kill unrelated processes.
Logs and the guard result are under `$REPLAY_ROOT/precision_format_routing/`.
Full per-call results and compiled resource identities are under `$R/results/`.

## 4. Recompute the G1 decision, not a selector result

```sh
cp lite_selector_20260920_v2/summarize.py "$R/"
"$PYTHON" "$R/summarize.py"
```

Keep the fresh results separate from this retained campaign. The retained
campaign fails G1 and therefore has no DT1/DT3/RF32 model or test timing. A fresh
run is not permission to alter the preregistered estimator, omit FP16, remove slow
samples, or inspect held-out timings before model freeze. Any follow-up direction
requires its own declared contract. The CPU reference graph qualification is a
bounded gate, not a proof covering arbitrary future layouts or compilers.
