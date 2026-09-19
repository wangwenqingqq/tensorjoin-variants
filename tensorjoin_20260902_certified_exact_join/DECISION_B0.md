# B0 Decision

Date: 2026-09-02

## Conclusion

**PASS under the frozen FP64 comparator, but do not promote the thesis yet.**
The hardware and exact-output feasibility are real. The comparison is not yet
strong enough for a database/system claim because an FP32 implementation may
emit the same result on the midpoint-threshold workload.

## Evidence

- Correctness: INT8 dot product and final pair sets match the NumPy FP64 oracle
  in all 8 processes and at all 3 radii.
- Dense INT8 versus non-TF32 FP32 GEMM: 2.138x geometric-mean speedup, 95%
  bootstrap interval [2.133x, 2.144x], 8/8 wins.
- End-to-end candidate versus exhaustive FP64:
  - 1 result/query: 6.920x, interval [6.783x, 7.028x], 8/8 wins.
  - 8 results/query: 5.065x, interval [5.015x, 5.111x], 8/8 wins.
  - 64 results/query: 1.897x, interval [1.875x, 1.912x], 8/8 wins.
- Marginal median candidate scan-plus-bound stage: about 220 microseconds;
  raw INT8 GEMM: about 43.7 microseconds. Eager FP64 epilogue/intermediate work,
  not INT8 GEMM, dominates the scan stage.

## Boundary

This is diagnostic PyTorch evidence. The runtime-selected library function,
SASS, sanitizer state, fused dataflow, and sustained larger relation sizes are
unresolved. The next gate is `PROTOCOL_B0X.md`, not a paper draft.
