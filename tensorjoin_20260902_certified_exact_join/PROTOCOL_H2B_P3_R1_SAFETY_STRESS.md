# Protocol Revision: H2B-P3-R1 Framework-Clean Safety and Stability

Date: 2026-09-03

## Reason for revision

The first P3 command completed the full audio target-8 operator with the exact
P1 output hash/count and no reported out-of-bounds access.  However,
`memcheck --leak-check full` reported 12 outstanding CUDA allocations totaling
153,224,192 bytes from the PyTorch caching allocator and its cuBLAS workspace
at interpreter shutdown.  Because the original protocol declared any
sanitizer error a failure, that attempt is rejected rather than reclassified.

Rejected evidence:

- `raw/h2b_p3_memcheck_esc50_panns.log`, SHA-256
  `60eb6d140957e85bd0c070220e5b3c8b3b3f78d7ea63086f1d32e6965a9a8610`;
- exact output before rejection: 1,024 IDs, ambiguity count 9, SHA-256
  `60f74a68af4b2eb4cd723db53c8ea3c2f2fa3ef22134dc87137145d88dec3a74`;
- sanitizer summary: 12 errors, all classified as leaks, 153,224,192 bytes.

The original command aborted the enclosing shell after this nonzero exit;
video memcheck, both syncchecks, and stress were not launched.

## R1 change

R1 changes only the safety harness and sanitizer environment:

1. retain `PYTORCH_NO_CUDA_MEMORY_CACHING=1` for instrumented processes;
2. after copying the checked output to host, release the device state, run
   Python garbage collection, clear PyTorch cuBLAS workspaces, empty the CUDA
   cache, and synchronize before interpreter exit;
3. retain `--leak-check full`; no leak category or error is suppressed;
4. write new append-only `*_r1.log` files.

The cuBLAS API/precision settings, data, shapes, thresholds, certificate,
dynamic count, FP64 repair, output checks, GPU, and stress contract are
unchanged.  The revised runner SHA-256 is
`5624418bfec918c98b98df90c369383760c765e5c9e3ba0b8f05d795104fa441`.

## R1 commands and gate

The four instrumented commands are the original commands with
`PYTORCH_NO_CUDA_MEMORY_CACHING=1` added.  They still use full leak checking
and nonzero error exit codes.  Stress uses the revised runner but does not set
the allocator diagnostic variable.

R1 passes only if both full actual-data memchecks and syncchecks report
`ERROR SUMMARY: 0 errors`, all four checked outputs match P1, the 4,000
two-buffer stress invocations have one invariant signature per cell, and the
GPU1 postflight is idle.  Any failure blocks P4.

## Current state

Stopped at preflight without GPU execution because physical GPU1 was occupied
by an external training process.  The process was not modified.  The
device-only revision to idle, same-model GPU4 is frozen in
`PROTOCOL_H2B_P3_R2_GPU4.md`.
