# Protocol: H2B-P2 Runtime cuBLAS Kernel and SASS Audit

Date: 2026-09-03

## Admission dependencies

P2 may run only with the accepted P1 artifacts and exact source/library hashes:

- P1 correctness `68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99`;
- P1 repeats `6970b41c794ec787e880778f9a501de61331a7eb4cda5303d85130db4215fc27` /
  `75d22742a4a952aa19bfb8b2a53bd36508b45bc9211a71fe1e51cb84b4edda30`;
- H2B kernel/runner `33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43` /
  `d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1`;
- cuBLAS container `e70f38efabe986acd5eb683497c62f0f1730a6176ee291d9d24c6e339d1fbf86`.

Physical GPU1, its campaign lock, foreign-PID refusal, FP32 A/B/C,
`CUBLAS_PEDANTIC_MATH`, `CUBLAS_COMPUTE_32F_PEDANTIC`, default algorithm, and
`NVIDIA_TF32_OVERRIDE=0` remain frozen.

## Trace contract

Run one full token-dot call for each exact shape:

- audio: M=640, N=2560, K=2048;
- video: M=662, N=2684, K=512.

Use NVTX ranges and an Nsight Systems CUDA/cuBLAS launch trace to bind API
calls to runtime kernel names, grid/block geometry, shared memory, correlation
IDs, and module IDs when exposed.  Then use Nsight Compute on one isolated
launch per selected shape to export selected-function SASS and resources.

Profiler collection is diagnostic-only.  Its duration may not enter the later
P4 public outer-wall comparison.

## Required provenance

For each shape retain:

- semantic contract and input/output layout mapping;
- cuBLAS version and full library hash;
- profiler commands and tool versions;
- exact runtime kernel name and launch geometry;
- module kind (`static`, runtime-loaded, JIT, or unresolved);
- selected-function raw/normalized SASS hash and instruction count when
  acquisition succeeds;
- registers, stack/local/shared memory;
- full opcode histogram and explicit counts for TF32 conversion/MMA,
  half/BF16/FP8/int MMA, FP32 FMA, loads, stores, and barriers.

## Pass/stop rule

P2 passes only if both shapes are bound to the actual runtime launch and the
selected dot-product function contains no TF32 or lower-precision MMA path.
If exact selected-function bytes cannot be acquired, mark P2 unresolved rather
than treating a similar static symbol as proof.  Any observed reduced-precision
path rejects the pedantic keeper configuration and blocks P3/P4 until a
separate corrected implementation passes P1 again.

Output paths are `raw/h2b_p2_*` plus an immutable
`results/h2b_p2_cublas_sass_audit.json` summary.

## Current state

Completed and accepted by `DECISION_H2B_P2.md`.  The two exact
runtime-selected functions were bound and exported by NCU; both contain FP32
`FFMA` and zero TF32 conversion or MMA-family instruction.  Parent static/JIT
image linkage remains unresolved and is retained as a provenance limitation.
