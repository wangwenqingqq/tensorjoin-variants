# G5 P4 Decision: Reject Formal Promotion

Date: 2026-09-03

## Conclusion

G5 establishes that one full-scale TensorJoin artifact is exact,
generated-code-audited, and safe/stable on the frozen CIFAR-GIST-512 60K
contract.  It does **not** establish a stable public-denominator performance
advantage.  The predeclared P4 gate fails because the candidate is only
`0.951x` as fast as MiSTIC in round 1, below the required `1.50x` in every
round.  Therefore the eight-round formal campaign is forbidden, and no
SIFT/Fashion full-scale expansion is run.

This decision preserves all six clean screen slots, including the unfavorable
candidate observation.  No observation is reclassified, removed, or replaced.

## Gates completed before timing

| Gate | Result | Evidence |
|---|---|---|
| P1 full compatibility | Pass | 3,926,078 directed IDs; exact hash `13cae87e...5963495`; zero duplicate, invalid, symmetry, or overflow error; result SHA-256 `7fcdd278...ff2a5` |
| P2 selected-function audit | Pass | stage 1 has 16 low-precision MMA instructions; stage 2 has no low-precision MMA or FP64; stage 3 has 27 FP64 arithmetic instructions; all stages have zero stack/local/LDL/STL; audit SHA-256 `ffcfdf9c...482b` |
| P3 safety/stability | Pass | full-leak memcheck reports zero errors and zero bytes leaked; 1,002 complete invocations have zero output/stage/overflow/invalid mismatch; manifest SHA-256 `6c6b0dfc...95d0` |
| Binary identity | Pass | the same three cubin and PTX hashes occur in P1, P3, and both P4 TensorJoin caches |

P1--P3 are correctness, generated-code, safety, and stability evidence.  They
are not substitutes for the failed public performance gate.

## Frozen P4 observations

The faster exact keeper by marginal median is MiSTIC.

| Round and order | GDS-Join (s) | MiSTIC (s) | TensorJoin (s) | TensorJoin speedup over MiSTIC |
|---|---:|---:|---:|---:|
| 0: GDS, MiSTIC, TensorJoin | 10.823552 | 4.822746 | 1.178670 | 4.091684x |
| 1: TensorJoin, MiSTIC, GDS | 11.108409 | 5.024050 | 5.284857 | 0.950650x |

- Paired geometric-mean speedup: `1.972247x`, above the `1.60x` aggregate
  threshold.
- Minimum paired speedup: `0.950650x`, below the mandatory `1.50x` per-round
  threshold.
- All six child results are exact and all six guards are admitted, isolated,
  and clean.
- Candidate source hashes, stage counts, cubins, and PTX are identical to P1;
  external binary/library identity also passes.
- Summary SHA-256: `3cd2a9847a453182e65e1d12245a2cb6e3a6b9af7d15bd85c787bbe9bab736c5`.
- Manifest SHA-256: `f9889a9f28811b31aa298575a0813d20210596d204a25fbd8d93ecc6c82c26d3`.

## Negative-evidence record

- **Target:** replace the older G2B candidate with the audited G4C-derived
  dynamic router under the same full public external-system denominator.
- **Workload and keeper:** CIFAR-GIST-512, 60,000-point exact self-join,
  epsilon `0.62890625`, pageable-host float32 to sorted canonical-host uint64
  IDs; freshly measured exact MiSTIC is the faster keeper.
- **Rejected mechanism at this denominator:** a 108-batch analytic INT8
  certificate -> certified FP32 -> residual FP64 route with dynamic count
  reads, compaction, host accumulation, and canonical sorting.
- **Measured effect:** one clean round wins by `4.092x`; the reversed-order
  clean round loses at `0.951x`.  The candidate public time changes from
  `1.179 s` to `5.285 s`, while MiSTIC changes only from `4.823 s` to
  `5.024 s`.
- **What is established:** output, work counts, source, specialization,
  cubin/PTX identity, and GPU-process isolation do not explain the difference.
- **What is not established:** P4 has no stage-level timing or CPU/NUMA
  isolation, so it cannot localize the variance to GPU execution, host
  scheduling, scalar synchronization, or final host reconstruction/sort.
  Any such attribution is a hypothesis, not a result.

## Decision and reopen condition

1. Mark the G5 public-performance/unification claim rejected at the cheap
   screen.
2. Do not run the formal eight-round campaign, SIFT/Fashion full-scale ports,
   tree/index packaging, or paper-facing G5 speedup prose.
3. Retain the separate G2B external-system claim and G3/G4 generated-code and
   router claims; never imply that one proves the other.
4. Reopen only after a mechanism-level change or a newly frozen diagnostic
   campaign that localizes the full-denominator variance with predeclared
   stage timing and CPU/NUMA controls.  A reopen must retain these unfavorable
   observations, use the same exact output and fresh external keepers, and set
   its gate before seeing new times.  Repetition solely to obtain favorable
   samples is not an admissible reopen.

