# H1 Execution Card

```text
Host: gpu-host-8 (hostname gpu-host-8)
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Branch/version: no Git repository; immutable source hashes recorded per result
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU/port: H1-R1 physical GPU3 only; CUDA_VISIBLE_DEVICES=3; no port
Dataset: frozen H0 ESC-50/PANNs and UCF101/R3D-18 caches
Model/checkpoint: frozen cached PANNs CNN14 and R3D-18 embeddings; no checkpoint load
Dimension: audio D2048, video D512; 4--7 tokens/object, maximum 8
Command: correctness -> four bounded memchecks -> safety summary -> timing
Log: raw/h1_r1_correctness.log, raw/h1_r1_memcheck_*.log, raw/h1_r1_timing_screen.log
PID: recorded by shell/Compute Sanitizer and visible through nvidia-smi
Do-not-touch: every foreign PID and all GPUs with an existing compute process
Validation: canonical CPU/keeper/candidate IDs, interval containment, memcheck, 50-sample timing
Rollback: no service/process mutation; remove only unpromoted H1 generated results after preserving them under rejected/
```

## Preconditions

1. `nvidia-smi` must show no foreign compute PID on physical GPU3.
2. Hold `@TENSORJOIN_ROOT@/.tensorjoin_gpu3_campaign.lock` with `flock`.
3. The runner repeats the UUID/PID check and refuses a shared launch.
4. Do not proceed to memcheck unless correctness passes; do not proceed to
   timing unless all four memcheck logs summarize to zero errors.

## Commands

```bash
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
PY=@TENSORJOIN_ROOT@/isaacsim6/env/bin/python
LOCK=@TENSORJOIN_ROOT@/.tensorjoin_gpu3_campaign.lock

flock -x "$LOCK" bash -o pipefail -c \
  'CUDA_VISIBLE_DEVICES=3 '"$PY"' src/run_h1_multivector_gpu_screen.py --phase correctness 2>&1 | tee raw/h1_r1_correctness.log'

for dataset in esc50_panns ucf101_r3d18; do
  for variant in keeper candidate; do
    flock -x "$LOCK" bash -o pipefail -c \
      'CUDA_VISIBLE_DEVICES=3 compute-sanitizer --tool memcheck --error-exitcode 86 '"$PY"' src/run_h1_multivector_gpu_screen.py --phase memcheck --dataset '"$dataset"' --variant '"$variant"' 2>&1 | tee raw/h1_r1_memcheck_'"$dataset"'_'"$variant"'.log'
  done
done

$PY src/summarize_h1_memcheck.py

flock -x "$LOCK" bash -o pipefail -c \
  'CUDA_VISIBLE_DEVICES=3 '"$PY"' src/run_h1_multivector_gpu_screen.py --phase timing 2>&1 | tee raw/h1_r1_timing_screen.log'
```
