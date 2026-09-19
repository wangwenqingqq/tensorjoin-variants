# G2 TensorJoin scalable-core safety gate

Date: 2026-09-03

Parent contract: `PROTOCOL_G2.md`.

Status: frozen before execution.

## Purpose and scope

This gate validates memory safety and sustained deterministic correctness of the
same triangular INT8-certificate, guarded FP32, and exact FP64 refinement kernels
used by the accepted full-scale path. It is correctness evidence, not latency
evidence.

## Deterministic stress workload

- Source: frozen CIFAR-10-GIST 60,000 x 512 float32 array.
- Radius: the frozen G2B threshold, squared distance
  0.3955230712890625.
- Safety subset: 256 original rows with the highest directed degree in the
  frozen accepted full-scale canonical output. Ties are resolved by original row
  ID. This deliberately creates a dense, high-output workload rather than an
  easy mostly-self subset.
- Oracle: filter the accepted full-scale canonical pair file to the selected
  original IDs, remap to local IDs, and sort. The oracle is independent of the
  stress executions and inherits the full three-way exactness audit.
- Kernel path: all upper-triangular 64 x 64 tiles, fixed worst-case capacity,
  INT8 certification, FP32 guarded filter, FP64 refinement, host normalization,
  and exact canonical comparison.
- Pointer churn: alternate two independent result/ambiguity/refinement/counter
  buffer sets on successive iterations.

## Gates

1. **Memcheck:** two complete iterations (both buffer sets) under
   `compute-sanitizer --tool memcheck --target-processes all
   --error-exitcode 99`; zero reported errors and exact oracle output in both.
2. **Sustained stability:** 1,000 complete iterations in one process; all 1,000
   outputs match the same oracle count/hash and stage counts remain stable;
   zero capacity overflow.
3. Each process runs on physical GPU0 under the shared campaign lock after 30
   continuous seconds with no GPU0 compute process. Foreign processes are never
   terminated.

Passing this gate permits the formal eight-round public-denominator campaign.
It does not validate formal timing or establish a performance claim.

