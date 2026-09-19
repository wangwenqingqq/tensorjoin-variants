# Protocol C0A: Fused INT8 MMA-to-Status Scan

Experiment ID: `tensorjoin_20260902_triton_fused_status_c0a`

## Scope

C0A tests one mechanism only: fuse the INT8 dot product, dequantization, and
three-way certificate classification so the scan writes one status byte per
pair instead of an INT32 score matrix plus FP64 distance/bound matrices. It is
a diagnostic scan-stage gate at 1 result/query, not end-to-end acceptance.

## Frozen workload and candidate

- ESC-50 cache, seed, disjoint split, midpoint threshold, and shape
  `M=512, N=4096, K=1024` are unchanged from B0Y.
- Inputs: row-major query INT8 codes and a pretransposed contiguous `KxN` base
  code matrix; per-vector FP32 scales, reconstructed squared norms, and
  residual norms.
- Kernel: Triton 3.6.0, tile `64x64x64`, four warps, three stages, one program
  per output tile. INT8 `tl.dot` accumulates in INT32.
- Epilogue: FP32 reconstructed distance and triangle interval. Add `1e-4` to
  the residual-radius sum as a C0A numerical pad.
- Output: `uint8` status only: 0 reject, 1 direct accept, 2 ambiguous.
- No INT32 dot matrix or FP32/FP64 distance matrix may be stored globally.
- Inputs and output are resident; CUDA-event kernel timing excludes allocation
  and all later compaction/refinement.

## Correctness and mechanism gates

1. Zero direct false accepts and zero direct false rejects against the NumPy
   FP64 oracle.
2. Refining every status-2 pair in FP64 yields the exact oracle pair-ID set.
3. Ambiguous fraction <=0.2% at 1 result/query (A0 was 0.0765%).
4. Extracted SASS contains native integer matrix-multiply instructions; source
   and artifact inspection confirm no global INT32 score matrix.
5. `compute-sanitizer --tool memcheck` reports zero errors for the frozen
   launch before any promotion.

## Performance gate

- One process diagnostic screen, 20 warmups and 200 retained CUDA-event
  observations under the same GPU 0 lock and idle check.
- Record p10/median/p90 and every sample.
- Pass if median fused scan latency <=100 microseconds and p90 <=110
  microseconds. This leaves at least about 120 microseconds for compaction and
  refinement under the future 1.5x end-to-end target relative to the measured
  ~331 microsecond guarded-FP32 keeper.

If any gate fails, stop C0 and the current performance thesis. If all pass,
design C0B for fused/streaming compaction and ambiguous refinement under a new
eight-process end-to-end contract.

