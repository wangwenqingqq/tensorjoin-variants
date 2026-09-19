# Run Card

Host: `gpu-host-8` (live hostname observed: `gpu-host-8`)

Path: `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`

Branch/version: standalone experiment directory; scripts mirrored from the
local project directory before execution.

Environment: A0 uses
`@TENSORJOIN_ROOT@/wxr_gpu-host-1/GTS/bench_env/bin/python`. B0 uses
`@TENSORJOIN_ROOT@/isaacsim6/env/bin/python` (PyTorch 2.11.0+cu130).

GPU/port: A0 is CPU-only with `CUDA_VISIBLE_DEVICES=""`. B0 uses only physical
GPU 0 after an idle check and holds
`@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`. No port.

Dataset: official ESC-50 archive pinned to commit
`33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.

Model/checkpoint: none; deterministic signal-processing feature.

Dimension: `D=1024`; query/base shape `512 x 4096`.

Command:

```bash
CUDA_VISIBLE_DEVICES="" \
@TENSORJOIN_ROOT@/wxr_gpu-host-1/GTS/bench_env/bin/python \
  @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/src/run_a0_esc50.py \
  --dataset-root @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/raw/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6 \
  --project-root @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
```

Log: `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join/raw/a0.log`

PID: recorded in `raw/a0.pid` while active.

Do-not-touch: all GPU processes and all processes outside the experiment path.

Validation: checksum and source receipt; frozen shape/seed/features; zero
containment and classification violations; ambiguity gates in `PROTOCOL_A0.md`.

Rollback: stop only the PID recorded in `raw/a0.pid`; preserve partial downloads
and logs for diagnosis; remove nothing outside the experiment directory.

## C0B formal campaign

Environment: `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`, PyTorch
2.11.0+cu130, Triton 3.6.0.

GPU: physical GPU0 only, UUID
`GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1`; verified idle before every fresh
process while holding `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`.

Contract: resident `512x4096x1024`, midpoint radius yielding one exact
result/query; guarded FP32 keeper versus fused INT8 compact-and-FP64-refine
candidate; orders `AB,BA` repeated four times; 20 warmups and 100 observations.

Logs/results: `raw/c0b_process_*.log`, `results/c0b_process_*.json`, and
`results/c0b_summary.json`. Safety evidence is in `raw/c0b_memcheck.log` and
`results/c0b_stress1000.json`.

Do-not-touch: all non-campaign processes and all GPUs other than physical GPU0.
No clock or power-limit changes.
