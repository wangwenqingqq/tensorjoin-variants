# Protocol Revision: H2B-P3-R3 Device-Access Safety

Date: 2026-09-03

## Reason for revision

P3-R2 completed the full audio target-8 operator with exact P1 output, then
full leak checking reported three allocations totaling 8,520,704 bytes.  All
three allocation backtraces originate in `cublasCreate_v2` through PyTorch's
process-owned current cuBLAS handle.  Disabling PyTorch CUDA allocation
caching and explicitly clearing experiment tensors and cuBLAS workspaces
removed the other nine/144.7 MiB outstanding allocations, but cannot destroy a
handle owned by PyTorch without changing the accepted P1 API implementation.

The R2 run is rejected by its frozen leak-free rule and retained at
`raw/h2b_p3_memcheck_esc50_panns_r2.log`, SHA-256
`6ae8d7e7d28961e0b68f13c0952f5a60c408fdae8abe95ec566240fa04a1f11c`.
It reports no invalid/misaligned/out-of-bounds access before the three leak
records.  No later R2 subgate was launched.

## R3 contract

R3 narrows the memcheck claim from **leak-free process shutdown** to
**device-memory access safety of the full operator**:

- run Compute Sanitizer `memcheck` with `--leak-check no`, while retaining
  `--error-exitcode 86` for access errors;
- keep `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and explicit state/workspace cleanup;
- retain the rejected full-leak R2 log as the resource-lifetime diagnostic;
- do not use wording such as leak-free, resource-clean, or proof of absence of
  all memory bugs;
- keep both full actual datasets, `synccheck`, exact outputs, 4,000 two-buffer
  stress invocations, physical GPU4, and every other P3-R2 rule unchanged.

R3 writes new append-only `*_r3.log` files.  The runner is unchanged from R2,
SHA-256 `63b19aab011f8a4e51f34c739011897ea5fc2a3afed99040622dd3c444d945ba`.

## Pass/stop rule

R3 passes only if both memchecks and both syncchecks report
`ERROR SUMMARY: 0 errors`, every checked result matches P1, all four stress
cells retain one exact signature across 1,000 invocations, and GPU4 is idle at
postflight.  The separate three-allocation cuBLAS-handle limitation remains
material caveat even if R3 passes.

## Current state

Completed and accepted by `DECISION_H2B_P3_R3.md`: four/four full-operator
access/synchronization checks are clean and four/four 1,000-invocation
two-buffer stress cells retain one exact signature.  The separate
three-allocation cuBLAS-handle lifetime limitation remains recorded; R3 is not
a leak-free result.
