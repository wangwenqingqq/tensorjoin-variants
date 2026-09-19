# H2A Certified FP32 Strong-Baseline Screen: Execution Status

## Current conclusion

H2A **passes** on physical GPU3 of `gpu-host-8`.  The exact H1-R1 INT8
candidate is 2.758x--10.455x faster than the certified direct-FP32 baseline
under the frozen outer-wall contract, with 50/50 wins in all four cells.

## Completed gates

- Four-cell exactness and interval containment against the independent CPU
  direct-FP64 oracle: pass.
- Bounded actual-data Compute Sanitizer coverage of fixed D2048 and ragged D512
  baseline paths: two/two clean.
- Alternating 10-warmup/50-observation outer-wall screen: four/four cells pass
  the 1.25x gate.
- Runtime-generated code and static resource audit: complete; the baseline
  uses FP32 arithmetic/shuffle reduction and no Tensor Core MMA instruction.
- Source and evidence hashes: recorded in `DECISION_H2A.md` and
  `receipts/h2a_evidence_sha256.txt`.
- Postflight: physical GPU3 returned to 14 MiB/0% utilization with no compute
  application listed.  No foreign process was stopped or modified.

## Scope boundary and next action

This result eliminates exhaustive-FP64 weakness as the explanation for H1,
but it is still a small resident single-process screen against a custom direct
FP32 kernel.  H2B must compare with a certified pedantic SGEMM/cuBLAS keeper
and larger streamed object matrices before any tree/index contribution or
paper-level upper-thesis promotion.
