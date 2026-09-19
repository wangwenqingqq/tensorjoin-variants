# B0Y Decision

Date: 2026-09-02

## Conclusion

**Reject the eager candidate against the strongest current exact-output
keeper.** The FP64-only win is real but insufficient for a systems paper.
Proceed only with the one predeclared fused scan prototype at 1 result/query.

## Evidence

- Both variants exactly match FP64-oracle pair IDs in all eight processes.
- Guarded FP32 refines only 107, 418, and 993 pairs, versus 1,605, 7,300, and
  26,185 pairs for INT8.
- Guarded FP32 / eager INT8 end-to-end ratios:
  - 1 result/query: 0.771x, 95% interval [0.765x, 0.778x], 0/8 wins.
  - 8 results/query: 0.559x, interval [0.556x, 0.562x], 0/8 wins.
  - 64 results/query: 0.207x, interval [0.204x, 0.211x], 0/8 wins.
- Marginal medians at 1 result/query: guarded FP32 331.2 microseconds, eager
  INT8 429.8 microseconds.
- Candidate scan-plus-bounds is 219.1 microseconds, while raw INT8 GEMM is
  43.5 microseconds. Materialized FP64 epilogue/intermediate work dominates.

## Reopen boundary

Only a fused low-selectivity path that avoids materializing the INT32 score
matrix and FP64 bound matrices may reopen performance. It must first produce a
one-byte classification state with zero unsafe direct decisions, native INT8
MMA evidence, and <=100 microsecond scan latency. Failure stops the performance
thesis rather than falling back to the FP64 comparison.

