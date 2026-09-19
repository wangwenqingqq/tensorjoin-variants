# G5 Run Card

Date: 2026-09-03

```text
Host: gpu-host-8 (gpu-host-8)
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Branch/version: no Git repository; SHA-256 manifest controls identity
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU/port: physical GPU2 (R1); no network service
Dataset: data/g2b_cifar60000/vectors_f32.npy, CIFAR-GIST 60000x512
Model/checkpoint: not applicable
Dimension: N=60000, D=512, 64x64x64 stage-1 tiling
Command: generated per G5 phase after implementation freeze
Log: raw/g5_*
PID: recorded by guarded orchestration
Do-not-touch: physical GPU0 and all non-G5 processes
Validation: exact 3,926,078 directed IDs; SHA-256 13cae87e...5963495
Rollback: stop only the G5-owned process group; preserve all partial evidence
```

Live preflight at 2026-09-03T12:24:26Z found GPU0 occupied by external PID
20277 and GPU1--GPU7 idle.  GPU1 later acquired a foreign process before the
first candidate launched, so the active R1 contract is pinned to identical
GPU2.  See `PROTOCOL_G5_P1_R1_GPU2.md`.
