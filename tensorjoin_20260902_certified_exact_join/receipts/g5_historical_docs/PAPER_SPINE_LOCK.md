# Paper Spine Lock: Exact GPU Precision Routing for Similarity Join

Date: 2026-09-03

## Status and venue lens

**Recommended route:** a SIGMOD/VLDB-style exact similarity-join operator
paper, with SC/PPoPP as the fallback lens if database novelty is judged too
incremental.  This is an extensibility/systems thesis, not a claim of the first
exact low-precision join.  Annual venue deadlines, formats, and tracks must be
verified before submission planning.

The more novel multi-vector headline is rejected by `DECISION_H2B_P4.md` and
must not be used to inflate the single-vector paper.

## One-sentence thesis

> An exact GPU similarity join can exploit low-precision matrix units without
> fixing one precision globally: certified, output-sensitive routing resolves
> most pairs after an INT8 Tensor-Core stage, escalates only uncertain pairs
> through FP32 and FP64, and yields exact output with measured advantages over
> exact GPU join systems on a frozen full-scale public self-join.

This sentence is locked for planning.  It may be strengthened only by a new
same-artifact external experiment; it may not acquire “first,” “general,” or
“fastest” without new evidence.

## Five-beat paper spine

1. **Problem.** Exact high-dimensional similarity joins remain expensive on
   GPUs because one global arithmetic choice either wastes high precision on
   easy pairs or loses exact threshold decisions.
2. **Strongest existing alternative.** GPU exact join systems execute the
   complete comparison path at a fixed precision, while approximate
   mixed-precision systems can be faster but change the output set.
3. **Insight.** Precision is a per-pair query-plan decision: an interval around
   each low-precision score can safely accept/reject most pairs, and only the
   unresolved boundary population needs a more expensive stage.
4. **Mechanism.** TensorJoin implements a three-stage
   INT8-Tensor-Core/FP32/FP64 router with generated-code-audited certificates,
   compaction, actual dynamic stage extents, ragged-dimension handling, and
   exact output reconstruction.
5. **Result.** On the frozen CIFAR-GIST-512 60K public denominator, TensorJoin
   returns the same 3,926,078 directed IDs and is 5.083x faster than the faster
   exact external keeper; separate public N=4096 anchors attribute
   1.271x--2.329x to dynamic precision routing across D128/D512/D784.

## Paper contribution IDs and evidence map

| Paper ID | Contribution claim | Required evidence type | Frozen supporting evidence | Material boundary |
|---|---|---|---|---|
| P-C1 | A sound three-stage precision certificate converts low-precision matrix-unit scores into exact threshold decisions and refines only unresolved pairs. | Algebra, independent FP64 oracle, adversarial correctness, selected generated code, sanitizer/stress | G3A, G3B-R1, G3C-B-R1; ledger C20--C22 | Tested finite domains and SM120 toolchain, not a theorem over arbitrary float32 inputs; does not retroactively prove G2B code |
| P-C2 | Output-sensitive GPU routing makes precision escalation an executable query plan with actual dynamic counts and ragged native dimensions. | Full-pipeline ablation, actual counter reads/launch extents, cross-dimension timing, exact outputs | G3D, G4A-R1, G4B-R1, G4C; ledger C23--C26 | G4 timing is resident-input/output and uses an internal all-FP64-refinement keeper, not an external end-to-end system |
| P-C3 | The complete system is competitive with exact external GPU joins under a full public denominator while exposing the quality gap of a faster approximate point. | Same-source/end-to-end exact external comparison, process-level uncertainty, safety/stress | G2B; ledger C18--C19 | One full-scale public dataset; G2B binary is not the separately audited G4C artifact |

No contribution may cite H1/H2A speedups as evidence against the strongest
exact baseline.  H2B C34 is the controlling counterevidence for multi-vector
performance.

## Headline result inventory

| Evidence role | Workload and denominator | Strongest allowed number | Allowed interpretation |
|---|---|---:|---|
| External exact comparison | CIFAR-GIST-512, 60K self-join, pageable-host input through sorted host IDs | 0.926 s; 5.083x over MiSTIC FP64, 95% CI [5.030x, 5.133x] | Scoped end-to-end exact-system result |
| Second exact external keeper | Same G2B denominator | 11.783x over GDS-Join FP64, 95% CI [11.716x, 11.846x] | Supporting external comparison |
| Approximate context | Same G2B denominator | FaSTED 0.229 s, but 598 false negatives and 1,130 false positives | Quality/latency context, never an exact baseline |
| Dynamic-router attribution | Public SIFT-128/CIFAR-GIST-512/Fashion-784, N=4096,target-64, resident operator | 1.271x/2.329x/1.710x, all 8/8 process wins | Cross-dimension mechanism evidence |
| Exact selectivity breadth | 27 public dimension-density-scale cells | zero observed unsafe decisions; <=0.2882% residual FP64 on the two non-CIFAR N=4096 sources | Correctness/selectivity, not speed |

## Strongest reviewer attacks already visible

1. **“This is error-bounded filter/refine plus engineering.”**  Accept the
   component lineage; defend the contribution as a generated-code-aware,
   output-sensitive exact GPU query plan with measured external-system impact,
   not a new mathematical family.
2. **“Your strongest external result and strongest proof use different
   implementations.”**  This remains true.  G5 built one full-scale artifact
   that passes exactness, selected-code, and safety gates, but its public
   timing screen failed the frozen per-round gate.  Never imply that G3/G4
   binary evidence proves G2B or replace G2B with the rejected G5 timing.
3. **“The broad result is internal and resident-only.”**  Lead with G2B's
   complete public denominator; use G4 only for attribution and breadth.
4. **“The faster mixed-precision baseline wins.”**  FaSTED is faster but not
   exact under the frozen oracle.  Report its 598 misses and 1,130 additions,
   without judging application acceptability.
5. **“Why no index?”**  Define the work as an exact high-dimensional join
   operator and show where it fits in a database query plan.  Do not append the
   rejected conventional tree as cosmetic packaging.

## Paper-critical G5 gate outcome

G5 is complete and rejected at its predeclared cheap performance screen.  One
CIFAR-GIST-512 60K artifact reproduces the exact output, retains the same three
audited cubins/PTX across compatibility, memcheck/stress, and both timing
slots, and runs in six clean isolated external-comparison slots.  However, its
paired speedups over MiSTIC are `4.092x` and `0.951x`; the second round violates
the mandatory `1.50x` minimum even though the paired geometric mean is
`1.972x`.  Per protocol, no formal eight-round or SIFT/Fashion full-scale work
is admitted.  See `DECISION_G5_P4.md`.

The paper must therefore keep G2B external end-to-end evidence and G3/G4
generated-code/router evidence in separate sentences.  The next action is the
teaser, pipeline figure, and external-results/mechanism tables under that
explicit boundary, not another G5 rerun chosen to wash out the negative round.

## Forbidden scope expansion

- first exact low-precision/Tensor-Core similarity join;
- fastest exact similarity join in general;
- multi-vector performance advantage over pedantic cuBLAS;
- arbitrary-dataset, arbitrary-dimension, multi-GPU, or indexed dominance;
- leak-free H2B process shutdown;
- proof that G3/G4 generated code is the G2B timed binary.
