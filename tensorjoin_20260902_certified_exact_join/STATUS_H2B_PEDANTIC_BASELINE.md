# H2B Status: Pedantic Exact Baseline Gate Closed

Date: 2026-09-03

## Final status

**Closed negative.** P0 opportunity, P1 correctness, P2 actual selected-SASS,
and P3-R3 bounded access/stress gates passed. P4 then fired its predeclared
early-stop condition in fresh process 0. The H1 multi-vector candidate does
not have a publishable performance advantage over the strongest admitted
pedantic-cuBLAS exact keeper.

## Controlling evidence

- Exactness: P1 zero failures across four full cells and 14 adversarial
  dimension-cases; two fresh repeat processes.
- Generated code: actual selected functions contain 1,152/576 FP32 `FFMA`
  and zero TF32 conversion or MMA-family instruction.
- Bounded safety/stability: four access/synchronization sanitizer logs with
  zero errors and 4,000 exact two-buffer invocations. This is not leak-free;
  three process-owned cuBLAS-handle allocations remain in the rejected
  full-leak diagnostic.
- Performance rejection: exact outer-wall speedups
  1.169x/0.666x/1.016x/0.750x for audio/video target-1/8. No cell clears 1.25x
  and both target-8 cells favor the keeper.

## Do not run

- P4 slots 1--7 or sustained completion;
- H2B larger streaming;
- a tree/index wrapper intended to rescue H2B;
- more modalities or small kernel tuning under the same thesis.

## Reopen condition

Only a new mechanism that reduces H1 ambiguity/FP64 panel work and first
passes the identical four-cell 1.25x cheap kill can reopen the multi-vector
route. The active paper route is the single-vector exact precision router in
`PAPER_SPINE_LOCK.md`; its highest-value strengthening gate is a unified
same-artifact external comparison.
