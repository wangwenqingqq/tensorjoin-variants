# Run Card G2

```text
Host: gpu-host-8 (live hostname and hardware receipt required before execution)
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Branch/version: directory is not a Git repository; use source/archive/binary SHA-256 receipts
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python; CUDA 13.1 nvcc by absolute path
GPU/port: physical GPU0 only; no service port
Dataset: Cifar60K base fvecs; deterministic 4,096-row G2A subset before full 60,000-row G2B
Model/checkpoint: none
Dimension: N=4,096 or 60,000; D=512; exact float32-source -> float64-oracle Euclidean self-join
Command: freeze after adapters and runner are hashed; never time an ad-hoc command
Log: raw/g2_*.log; structured outputs results/g2_*.json
PID: capture each fresh process PID and campaign lock holder
Do-not-touch: every foreign GPU process; GPU clocks/power settings; upstream source snapshots
Validation: canonical directed pair IDs incl. self; zero missing/extra/duplicate/overflow; repeated hash
Rollback: adapters live outside external/ snapshots; abort on occupancy change; retain failed attempts
```

Current checkpoint:

- G2A is accepted: the direct FP64 oracle, GDS-Join FP64, MiSTIC FP64, and
  TensorJoin agree on 262,144 canonical directed IDs in two isolated runs.
- The G2A summary hash is
  `92b7975a5b4d7a8a7862d2797a636385318cf3e2822f5726e7daccde05f25c76`.
- Cifar60K and exact upstream source archives are present locally and remotely
  with matching hashes; this is historical state and must be rechecked before
  G2B execution.
- `DESIGN_G2B.md` freezes an upper-triangle, per-tile-batch worst-case capacity
  proof. Its new path must pass G2A2 before a full-scale smoke.
- FaSTED's synthetic SM120 smoke remains compatibility evidence only.
