# Run Card G6

```text
Host: gpu-host-8; live hostname gpu-host-8 (recheck at execution)
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Branch/version: no Git repository; GTS 3bac1b725e92e98e3b69d5cb79ad9c56ccb5e639; cuvs-cu13 26.8.1
Env: dedicated project envs/cuvs_py312; CUDA 13.1.115; GCC 13.3; CMake 3.28.3
GPU/port: physical GPU1 only under .tensorjoin_g6_gpu1.lock; no service port
Dataset: deterministic G2A 4,096-row subset, then G2B CIFAR-GIST 60,000
Model/checkpoint: none
Dimension: D=512; exact float32-source -> float64-oracle Euclidean self-join
Command: frozen adapter runners only; no ad-hoc timed command
Log: raw/g6_*.log; results/g6_*.json; artifacts/g6_*
PID: record lock holder, launched process, and pre/post GPU process list
Do-not-touch: GPU0 PID 366149; GPU3 PID 448038; every foreign process; upstream GTS snapshot
Validation: sorted canonical directed uint64 IDs incl. self; exact count/hash; no full-k saturation
Rollback: remove only project-local env/build/adapter outputs; preserve failed attempts and upstream clone
```

## Frozen checkpoint

- Physical GPU0 was occupied by PID 366149 (approximately 81 GiB) and GPU3 by
  PID 448038 (approximately 18.5 GiB) at the planning checkpoint.  These are
  observations, not ownership claims; re-identify all live processes before
  every execution.
- GPUs 1, 2, and 4--7 were idle at that checkpoint.  G6 chooses physical GPU1
  and must abort rather than evict a later occupant.
- The untouched upstream GTS configure succeeds when nvcc and SM120 are explicit,
  but compilation with upstream C++14 fails because CUDA 13.1 CCCL requires
  C++17.  This is a toolchain compatibility failure, not performance evidence.
- RT-HiSS remains the newest direct exact baseline, but no runnable artifact was
  found.  G6 must not claim a measured RT-HiSS comparison.

