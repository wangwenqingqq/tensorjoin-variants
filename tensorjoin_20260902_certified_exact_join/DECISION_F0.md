# F0 Precision-Cascade Decision

Date: 2026-09-03

## Conclusion

**REJECT the fixed cascade as a universal six-cell mechanism, while retaining
its high-output opportunity as partial evidence.** Exactness passes everywhere,
but the tie-expanded low-radius audio cell violates the frozen work gate.

## Evidence

- All six cells have zero FP32 false accepts, false rejects, and final pair-ID
  mismatches.
- Video sends 2.13%/3.89% of INT8 ambiguity to FP64 at 1/64 results/query.
- HSI sends 15.88%/8.02% to FP64 at 1/64 results/query.
- Audio target 64 sends 5.22% to FP64, but tie-expanded audio target 1 sends
  7,628/11,225 pairs, or 67.96%, violating the required <=25% universal gate.
- The test is an opportunity simulation only: gathering, queueing, launches,
  and latency are excluded.

## Interpretation

The failure is localized to an almost-zero radius with many bitwise-identical
and near-identical PANNs vectors. It does not justify retuning the fixed
`1e-3` FP32 guard after measurement. The three high-output cells consistently
reduce FP64 work by 12.46x--25.71x, which is sufficient to motivate a distinct
selectivity/tie-aware router gate, but not to claim speed.

## Decision

Do not implement or promote one unconditional three-precision pipeline. A
future candidate must predeclare separate routes for tie/near-zero and ordinary
radius regimes, compare them with external same-contract baselines, and pass a
new frozen latency protocol. F0 itself remains rejected and is not reopened by
the favorable five cells.
