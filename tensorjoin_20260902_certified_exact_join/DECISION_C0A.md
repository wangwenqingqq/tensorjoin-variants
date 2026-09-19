# C0A Decision

Date: 2026-09-02

## Conclusion

**PASS to C0B.** Fusing integer MMA and certificate classification removes the
eager scan bottleneck with enough latency budget left for compact output and
exact ambiguous refinement.

## Evidence

- Correctness: 0 false direct accepts, 0 false direct rejects, 0 final
  classification mismatches.
- Ambiguous pairs: 1,608/2,097,152 = 0.0767%, below the 0.2% gate.
- SASS: 16 native `IMMA.16832.S8.S8.SAT` instructions.
- Global stores: only two vector status-store instructions in the compiled
  kernel; no global INT32 score matrix.
- Memcheck: target exits normally with `ERROR SUMMARY: 0 errors`.
- Formal scan latency: p10 38.653, median 39.616, p90 40.768 microseconds.

## Boundary

This result is scan-only, one process, one shape, and one selectivity. It does
not include status compaction or FP64 refinement and cannot be called an
end-to-end speedup. C0B must remeasure the guarded-FP32 keeper in the same
eight-process campaign.

