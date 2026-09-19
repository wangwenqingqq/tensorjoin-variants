# Protocol B0X: Strong Same-Output FP32 Keeper

Experiment ID: `tensorjoin_20260902_fp32_strong_baseline_b0x`

## Reason for this gate

B0 passed against exhaustive FP64, but a database/system result must also face
the fastest straightforward implementation that produces the same pair IDs on
the frozen workload. B0X therefore replaces the B0 end-to-end keeper with a
non-TF32 FP32 exhaustive join and verifies its output against the FP64 oracle.
This is a new frozen experiment; it does not rewrite B0.

## Contract

- Hardware, GPU isolation, feature cache, shape, seed, midpoint thresholds,
  resident-input scope, output materialization, warmups, observations, process
  count, order schedule, estimator, and bootstrap procedure are identical to
  `PROTOCOL_B0.md`.
- Keeper: FP32 `torch.mm`, FP32 norm/threshold epilogue, and materialized pair
  IDs. TF32 is disabled and float32 matmul precision is `highest`.
- Candidate: the unchanged B0 INT8 + FP64 bound + FP64 ambiguous-refinement
  eager PyTorch pipeline.
- Correctness: both variants must emit exactly the FP64-oracle pair-ID set at
  every radius in every process.

## Decision rule

The eager candidate passes the strong-comparator gate only if its end-to-end
speedup over the FP32 keeper has a 95% bootstrap lower bound >=1.5x and at least
7/8 process wins at every radius.

Failure rejects the eager PyTorch implementation, not automatically the fused
mechanism. A single low-selectivity fused CUDA prototype remains admissible
only if all of the following already-measured facts hold:

1. A0 maximum ambiguity remains <=5% with exact final output.
2. B0 dense INT8 versus FP32 lower confidence bound remains >=1.5x.
3. B0X FP32 keeper itself matches the FP64 oracle in all processes.
4. The B0 candidate scan is dominated by materialized FP64 epilogue/framework
   work rather than INT8 GEMM, so a fused design can specifically remove the
   INT32 matrix store/reload and intermediate distance/bound matrices.

That prototype is admitted only for the 1-result/query radius first. It must
beat the B0X FP32 keeper by >=1.5x under a new correctness/sanitizer/paired
timing contract before extension to other selectivities. If it fails, stop the
performance thesis; do not rescue it with the FP64-only comparison.
