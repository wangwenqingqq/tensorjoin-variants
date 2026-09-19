# C0B Decision

Date: 2026-09-02

## Conclusion

**PASS the frozen end-to-end gate and advance to breadth and external-baseline
validation.** Fused INT8 certificate scoring, sparse compaction, and exact FP64
refinement beat the same-output guarded-FP32 keeper by more than the required
1.5x margin.

## Evidence

- Correctness: exact FP64-oracle output in 8/8 fresh processes; 512 final pairs,
  zero direct false accepts, mismatches, duplicates, or buffer overflows.
- Selectivity: 81 direct accepts and 1,608/2,097,152 ambiguous pairs (0.0767%).
- Safety: memcheck exited normally with zero errors; 1,000 stress launches
  produced one invariant final-output SHA-256.
- Mechanism: extracted scan SASS contains 16 native
  `IMMA.16832.S8.S8.SAT` instructions; refinement contains FP64 arithmetic;
  output is compact pair IDs and counters rather than a dense score/status
  matrix.
- Performance: guarded FP32 / fused candidate geometric-mean speedup is
  3.476x, with deterministic process-bootstrap 95% interval
  [3.402x, 3.542x] and 8/8 process wins.
- Marginal medians: guarded FP32 338.688 microseconds; candidate 98.576
  microseconds. AB/BA split geomeans are 3.488x/3.464x.

## Claim boundary

This is one in-memory shape, one low-selectivity radius, one GPU, and one
handcrafted ESC-50 representation. It does not establish robustness across
real learned audio/video embeddings, query/base sizes, dimensions,
selectivities, or competitive specialized systems. The `1e-4` FP32 bound pad
is empirically validated but is not yet a bit-level floating-point proof.

## Next gate

Freeze a breadth matrix before further timing: at least two real pretrained
embedding families spanning audio and video, multiple dimensions and
selectivities, and same-contract external comparators. Retain guarded FP32 as
the local exact keeper. A paper direction survives only if exactness and a
material end-to-end advantage persist beyond this single favorable point.
