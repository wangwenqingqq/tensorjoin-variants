# Protocol G5: Unified-Artifact Compatibility and Cheap Screen

Date: 2026-09-03

Experiment ID: `tensorjoin_20260903_g5_unified_public_cifar60k`

## Preconditions

- Follow `DESIGN_G5_UNIFIED_ARTIFACT.md` exactly.
- Run only on physical GPU2 of `gpu-host-8` while it is quiescent, per the R1
  amendment in `PROTOCOL_G5_P1_R1_GPU2.md`.
- Hold `/tmp/tensorjoin_g5_gpu2.lock` across every candidate, sanitizer, and
  timing process.
- Before each process, require a 30-second empty-GPU interval and save the full
  `nvidia-smi` preflight.  During each process, abort only the G5-owned process
  group if any foreign GPU2 process appears.  Never signal the foreign process.
- Use `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python` and a fresh
  process-specific `TRITON_CACHE_DIR`.
- Refuse to overwrite any result, raw log, artifact, or cache.

## Phase P0: implementation freeze

Create the batched G5 TensorJoin public runner and GPU2-compatible exact
external wrappers.  Record SHA-256 for every runner, imported arithmetic
source, external binary/library, protocol, and dataset artifact.  No timing
sample is admissible if any frozen upstream arithmetic source differs from the
hashes in the design.

## Phase P1: full compatibility

Run one candidate process over CIFAR-GIST-512 60K.  The public timer begins
after input file loading and candidate specialization compilation, and ends
after the canonical host output is sorted.  Retain:

- complete batch stage counts;
- exact canonical count/hash and structural checks;
- peak GPU/host memory;
- cache tree and exact cubin/PTX hashes before and after the timer;
- GPU preflight, occupancy trace, command, stdout/stderr, and postflight.

P1 passes only on exact output, zero overflow, and no new specialization
appearing during the public timer.

## Phase P2: selected-function audit

Audit the exact P1 cubins with CUDA 13.1 `cuobjdump`.  Save full SASS,
normalized selected-function SASS, resources, PTX, and opcode ledgers.  Repeat
the candidate in a second fresh cache no later than its cheap-screen slot and
require byte-stable cubins for the same three functions.

P2 is mechanism evidence, not public latency evidence.

## Phase P3: same-specialization safety and stress

Select a small set of 64 x 64 full-N tile coordinates from P1 that collectively
exercise stage-1 direct acceptance, stage-1 ambiguity, FP32 direct decisions,
and residual FP64 refinement.  Keep `N_=60000`, `K=512`,
`CAPACITY=16777216`, and all block/warp/stage constants identical to P1.

Run:

1. `compute-sanitizer --tool memcheck --target-processes all
   --error-exitcode 99 --leak-check full` for two iterations;
2. a fresh-process 1,000-iteration two-buffer stress.

Both use an independent CPU FP64 oracle over the selected tile pairs.  Require
zero sanitizer errors, zero mismatches, zero overflow, one stable stage-count
signature, all three stages exercised, and the same three cubin hashes as P1.

## Phase P4: two-round diagnostic cheap screen

Frozen method orders:

1. `gds, mistic, tensorjoin`
2. `tensorjoin, mistic, gds`

Each slot is a fresh process and each TensorJoin slot uses a fresh Triton cache.
No stale G2B time is reused.  Save the original order, result JSON, output hash,
GPU trace, and process tree.  A contaminated slot may be retried up to three
times; retain every rejected attempt.

P4 passes only when all six outputs are exact, the TensorJoin cubins are stable
with P1/P3, TensorJoin is at least 1.50x faster than the faster exact keeper in
each matched round, and the paired geometric-mean speedup is at least 1.60x.

## Decision boundary

- P4 pass: freeze the eight-round G5 formal protocol; do not draft paper prose
  yet.
- P4 fail: retain the negative evidence, stop G5, and do not repair the thesis
  with multi-vector, tree, modality, or packaging claims.
