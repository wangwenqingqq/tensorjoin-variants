# Protocol: H2B-P1 Direct-cuBLAS Correctness Gate

Date: 2026-09-03

## Frozen target and dependencies

- Host/device: `gpu-host-8`, physical GPU3, SM120, one visible GPU, GPU3
  campaign lock, and foreign-PID refusal before execution.
- Environment must set `NVIDIA_TF32_OVERRIDE=0` before CUDA initialization.
- H1 runner/kernel and H2A correctness/safety/timing hashes are the immutable
  values in `DESIGN_H2B_PEDANTIC_SGEMM.md`.
- H2B-P0 result SHA-256:
  `7b514ab72359320dd51384d9a791e3467be6f02417932ce0125fe0e0128c21e4`.

## Direct cuBLAS contract

The runner uses PyTorch only for device memory, stream ownership, and events.
It calls `libcublas.so.13` through a typed FFI binding and must check every
status code.  Before GEMM it must:

1. obtain the current PyTorch cuBLAS handle and stream;
2. call `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`;
3. query `cublasGetMathMode` and require the returned value to equal pedantic;
4. call `cublasGemmEx` with FP32 A/B/C,
   `CUBLAS_COMPUTE_32F_PEDANTIC`, and `CUBLAS_GEMM_DEFAULT`;
5. query and record the cuBLAS runtime version.

The row-major token matrices are passed through an explicitly documented
column-major transpose mapping.  A deterministic small rectangular smoke must
match direct CPU FP64 dot products within the frozen radius before full data.

## Numerical implementation

- Precompute token squared-norm centers, norm error radii, and L2 upper bounds
  in FP64 outside the operator denominator.
- Materialize the cuBLAS FP32 token dot matrix.
- A dedicated validation kernel writes token D2 lower/upper bounds; a separate
  production object kernel recomputes the identical formula while aggregating
  8x8 object panels.  The production kernel may not consume host-computed
  decisions.
- Exact object repair is byte-for-byte the accepted H1 direct-FP64 panel path.
- Constants are frozen by `DESIGN_H2B_PEDANTIC_SGEMM.md`; no post-run widening.

## P1 validation matrix

1. Four full H2A cells: every token dot radius, token D2 interval, object
   interval, direct decision, and final canonical ID is checked against an
   independent direct-FP64 CPU oracle.
2. Deterministic same-dimension adversarial/metamorphic cases: all zero,
   identical, sign-flipped, cancellation-heavy alternating signs, one-hot,
   small normal scale, and ragged 4--7-token objects.
3. Repeated output/hash check in two isolated processes after the single-run
   proof passes.

## Stop rule and evidence boundary

Any cuBLAS status/math-mode failure, dot-radius violation, token/object
containment violation, unsafe direct decision, duplicate/overflow/nonfinite
value, or final-ID mismatch rejects P1.  Compilation or an API query alone is
not a pass.  P1 supplies correctness evidence only; P2 generated-code identity,
P3 safety/stress, and P4 timing remain blocked until P1 passes.

Raw failed attempts are append-only.  The accepted result path is
`results/h2b_p1_pedantic_correctness.json`; stdout is
`raw/h2b_p1_pedantic_correctness.log`.

## Current state

Frozen before H2B-P1 implementation or GPU execution.
