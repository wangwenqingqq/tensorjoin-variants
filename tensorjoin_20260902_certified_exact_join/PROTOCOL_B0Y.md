# Protocol B0Y: Guarded FP32 Exact-Output Keeper

Experiment ID: `tensorjoin_20260902_fp32_guarded_exact_b0y`

## Motivation

B0X showed that a raw FP32 exhaustive join is fast but differs from the FP64
oracle by one pair at two radii. B0Y strengthens that keeper rather than
discarding it: perform the dense scan in FP32, classify pairs far from the
threshold directly, and refine a conservative threshold band in FP64.

## Frozen keeper

1. Disable TF32 and use PyTorch FP32 `torch.mm` plus an FP32 distance epilogue.
2. Use a fixed squared-distance guard of `1e-3` around the FP64 midpoint
   threshold.
3. Directly accept values at or below `threshold-1e-3`, directly reject values
   above `threshold+1e-3`, and recompute every remaining pair directly in FP64.
4. Materialize direct and refined accepts as pair IDs.

For normalized vectors, `1e-3` exceeds `8*gamma_(2D)` at `D=1024`, where
`gamma_n=n*u/(1-n*u)` and FP32 unit roundoff is `u=2^-24`. This is a deliberately
loose arithmetic screen. Because the selected cuBLAS reduction algorithm is not
resolved at B0Y, exact oracle equality remains mandatory and the guard is not
promoted as a universal formal proof.

## Other contract fields

Hardware, GPU 0 lock and idle checks, resident inputs, ESC-50 cache and split,
shape `512x4096x1024`, three midpoint radii, unchanged INT8 candidate, CUDA-event
scope, 20 warmups, 100 observations, eight fresh processes, AB/BA schedule,
bootstrap estimator, raw-sample retention, and correctness checks are identical
to `PROTOCOL_B0.md`.

## Gate

1. Both guarded-FP32 keeper and INT8 candidate must emit exactly the FP64-oracle
   pair IDs at every radius in all eight processes.
2. The INT8 candidate must have a 95% bootstrap lower speedup bound >=1.5x and
   at least 7/8 wins against guarded FP32 at every radius to pass unchanged.

If the eager candidate fails but correctness, A0 ambiguity, and the B0 raw
INT8-vs-FP32 dense gate remain valid, the result may admit exactly one fused
CUDA prototype at 1 result/query. Its design must eliminate the materialized
INT32 score matrix and FP64 intermediate bound matrices. The fused prototype
gets a separate >=1.5x same-keeper gate; failure stops the performance thesis.

