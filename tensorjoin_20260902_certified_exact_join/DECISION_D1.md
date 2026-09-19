# D1 Decision

Date: 2026-09-03 (Asia/Shanghai; measurements completed 2026-09-02 UTC)

## Conclusion

**PASS the frozen learned-audio breadth gate and advance to video embeddings and
an index-aware/external-baseline stage.** The generalized fused certified join
meets the predeclared exactness, safety, stability, and performance rules at
both tested radii without post-measurement tuning.

## Evidence

- Correctness: candidate and guarded-FP32 keeper exactly match the blocked
  direct-difference FP64 oracle in 8/8 fresh processes at each radius. There are
  zero unsafe direct accepts, final mismatches, duplicates, or overflows.
- Tie-aware workloads: nominal target 1 expands to 3,597 pairs (7.025/query)
  because of exact PANNs embedding ties; nominal target 64 yields 32,778 pairs
  (64.020/query). No tie is discarded to force a label.
- Performance at 7.025/query: 4.044x geometric-mean speedup, deterministic
  20,000-sample process-bootstrap 95% interval [4.035x, 4.051x], and 8/8
  process wins. Marginal medians are 1,430.416 microseconds for guarded FP32
  and 353.824 microseconds for the candidate.
- Performance at 64.020/query: 1.548x speedup, interval [1.536x, 1.561x], and
  8/8 wins. Marginal medians are 403.264 and 260.352 microseconds.
- Order check: AB/BA split geomeans are 4.049x/4.038x and 1.542x/1.553x at
  the two radii.
- Safety/stability: memcheck reports zero errors at both radii; each
  1,000-launch stress test has one invariant final-output hash.
- Mechanism: isolated-cache SASS contains 16 native INT8 IMMA instructions in
  the scan plus FP64 arithmetic in refinement. The candidate emits compact pair
  IDs rather than a dense score or status matrix.

## Strongest caveat

The 64-result point passes narrowly: its lower confidence bound is 1.536x
against a 1.5x gate. This remains one pretrained audio model, one dataset,
one 512x4096x2048 shape, one GPU, and an exhaustive resident-input scan. The
low-radius label is materially tie-expanded, the FP32 certificate padding is
empirically validated rather than bit-level proved, and no tree index or
external specialized system is yet in the denominator.

## Next gate

Freeze, before measurement, (1) an open video-embedding workload with the same
exact-output contract and (2) an index-aware design that uses tree nodes to
prune whole vector blocks before Tensor-Core certification. Compare against a
same-contract external CPU/GPU index where obtainable. The paper thesis should
not be promoted from "promising systems mechanism" until the advantage survives
at least one video representation and demonstrates value beyond exhaustive
scanning.

## Evidence identity

- Formal summary: `results/d1_summary.json`, SHA-256
  `a6f5e1c2976db510282abdbd591ba3adbd5fdc06c543036591ec6400cd7f6606`.
- Candidate source: `src/run_d1_triton.py`, SHA-256
  `a522375f9147a9b4a082b85939c0b100da563401ef34f70b3d32a5309be1e52f`.
- Formal SASS: `artifacts/d1_formal.sass`, SHA-256
  `ea8e0a79fa4caced05087e6e1dab2619d3dc1d491ca74b1bffcbcccb17e7ea4e`.
