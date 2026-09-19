# C0A Design Card

## Target and floor hypothesis

- Target: RTX PRO 6000 Blackwell Server Edition, compute capability 12.0.
- Frozen GPU: physical 0, UUID `GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1`.
- B0Y keeper median: ~331 microseconds end to end at 1 result/query.
- Required future candidate budget for 1.5x: ~221 microseconds.
- C0A scan budget: median <=100 microseconds, leaving ~121 microseconds for
  compaction and 1,605-pair refinement.

## Contract

- Dtype/layout: signed INT8 row-major `A[M,K]`; signed INT8 contiguous
  transposed base `B[K,N]`; INT32 accumulation; FP32 scale/bound epilogue;
  `uint8 status[M,N]`.
- Shape/dispatch: only `512x4096x1024`; no fallback claim.
- Numerical status: `1e-4` residual-radius pad; exact direct decisions and
  refined output validated against FP64, but bit-level certification remains
  unproven.

## Roles and tile

- One Triton program owns one `64x64` output tile.
- Four warps cooperatively load `64x64` K slices and issue INT8 dot products.
- K advances in 16 fixed `BK=64` steps.
- The same program owns the complete accumulator and status epilogue; there is
  no cross-program reduction or handoff.

## Dataflow and live set

| Phase | Live state | Last use |
|---|---|---|
| K loop | INT32 64x64 logical accumulator, A/B fragments, K pointers | Final dot step |
| Epilogue | INT32 accumulator, row/column scales, reconstructed norms, residual norms | Status predicate |
| Store | One-byte status tile | Global store completion |

The accumulator is never written to global memory. Distance, lower bound, and
upper bound exist only in the epilogue dataflow.

## Ready graph

```text
global A/B loads -> INT8 dot accumulation -> metadata loads
-> reconstructed distance -> interval predicates -> uint8 status store
```

There are no software barriers owned by source code. Triton/compiler-generated
pipeline barriers are audited through the emitted SASS only if the diagnostic
gate passes.

## Useful-work and movement hypothesis

- Useful INT8 MAC count is invariant versus B0's `_int_mm` scan.
- Removed global traffic: one 8 MiB INT32 matrix write and its later read, plus
  materialized FP64 distance/lower/upper intermediates created by eager
  operations.
- Added in-kernel work: scale products, norm combination, sqrt, two interval
  predicates, and a 2 MiB status store.
- Expected bottleneck: accumulator/epilogue register pressure or integer MMA
  feed, not result storage.

## Reject condition and verification ladder

1. Compile and high-entropy real-data correctness.
2. Confirm native integer MMA in SASS and one-byte-only output contract.
3. Memcheck the frozen launch.
4. Run the locked 20/200 CUDA-event diagnostic.
5. Reject on any correctness/mechanism/sanitizer failure, median >100
   microseconds, or p90 >110 microseconds.
