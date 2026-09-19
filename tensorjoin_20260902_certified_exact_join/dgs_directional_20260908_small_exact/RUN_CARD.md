# Execution card / replay

Host: gpu-host-8 -> gpu-host-8, via existing tiaoban SSH ProxyCommand.
Workspace: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
Experiment directory: dgs_directional_20260908_small_exact.
Python: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python.
Branch: project root is not a Git repository; frozen source hashes used.
CPU: Intel Xeon Gold6530, NumPy2.3.1, FP64, max4 BLAS/OpenMP threads.
GPU: onlyGPU2, same RTX PRO6000Server as the admitted upstream run.
Data: data/g2b_cifar60000/vectors_f32.npy; shape60000x512 FP32.
Quality: complete original FP64-predicate output; T=25921/65536 inclusive.
Do not touch: other jobs, power/clock settings, original kernels/manifests/PPT.
Rollback: no production path changed; retain this isolated experiment directory.
Final status: CPU and reference processes exited; GPU2 idle in final_environment.

Executed from the experiment directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 @TENSORJOIN_ROOT@/isaacsim6/env/bin/python -u src/screen.py

PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=$PWD/artifacts/reference_triton_cache \
 @TENSORJOIN_ROOT@/isaacsim6/env/bin/python src/guard_r2.py \
 --label reference_a0 --gpu 2 --target 8p_gpu2 -- \
 @TENSORJOIN_ROOT@/isaacsim6/env/bin/python -u src/obtain_reference.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 @TENSORJOIN_ROOT@/isaacsim6/env/bin/python -u src/audit.py
```

For a replay, make a new sibling directory under the workspace, copy only src,
PROTOCOL.md, NUMERICS.md, targets_r2.json, and create raw/results/artifacts there.
Do not overwrite this run. Sources derive the parent workspace from their path.
The unchanged prior two_gate_campaign_20260908 sources and frozen artifact
manifests must be present; reference generation fails on differing identities.
The guard verifies the host/GPU UUID, holds both existing GPU2 locks, waits30s
idle, and records every observed GPU process. It never terminates foreign jobs.

The delivery manifest excludes only itself and the private reference JIT cache.
The prior matched compiled programs are identified in results/reference_a0.json
and remain in the owning campaign. Output ID data are internal experiment
artifacts, not an authorization to redistribute the underlying dataset.
