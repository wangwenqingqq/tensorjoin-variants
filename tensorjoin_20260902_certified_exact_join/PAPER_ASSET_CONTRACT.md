# Paper Asset Contract after G5

Date: 2026-09-03

## Locked thesis

Certified, output-sensitive precision routing turns low-precision Tensor-Core
scores into an exact GPU similarity-join query plan by escalating only
uncertain pairs; the audited router improves an all-FP64-refinement path across
D128--D784, while a separately evaluated scalable implementation outperforms
exact GPU join systems on a frozen public 60K self-join.

## Evidence roles

| Role | Artifact | What it may establish | What it may not imply |
|---|---|---|---|
| Mechanism | G3/G4 and G5 P1--P3 | three-stage certificate, selected generated code, dynamic extents, ragged dimensions, exactness, bounded safety | external end-to-end latency |
| External system | G2B | exact full-public latency against MiSTIC and GDS-Join; approximate FaSTED context | that the G4 audited cubins are the G2B timed binary |
| Failed bridge | G5 P4 | same-artifact diagnostic times and the exact rejection boundary | a formal speedup or permission to discard the unfavorable round |

## Co-located headline-number lock

- Exact directed output: `3,926,078` IDs.
- G2B exact external comparisons: `5.083x` over MiSTIC and `11.783x` over
  GDS-Join.
- FaSTED approximate difference: `598` false negatives and `1,130` false
  positives.
- G5 rejection: paired rounds `4.091684x` and `0.950650x`; geometric mean
  `1.972247x`; failure is controlled by the every-round `1.50x` gate.

## Figure contracts

### Figure 1: evidence-separated teaser

**One message:** precision routing makes exact low-precision execution
mechanistically viable, and a separately scoped system implementation shows
external impact.

- Panel (a) uses only G5 P1 stage counts and exact output.
- Panel (b) uses only G2B formal medians, confidence intervals, and output
  comparison.
- A visible footer states that the panels are separate implementation scopes.
- FaSTED is marked approximate with both 598 false negatives and 1,130 false
  positives; it is not styled as an exact baseline.

### Figure 2: exact three-stage query plan

**One message:** the executable query plan uses actual dynamic counts to shrink
the candidate population before progressively more precise work while
reconstructing the exact canonical output.

- Dataflow follows `src/run_g5_tensorjoin_public.py`.
- Matrix and tile annotations use the actual `60000x512`, `64x64x64`, 440,391
  upper tiles, 4,096 tiles/batch, and 108-batch specialization.
- Stage counts are the exact G5 P1 counts, not illustrative values.
- Stage 1 is the only Tensor-Core stage; FP32 and FP64 stages are CUDA-core
  scalar-per-pair filters/refinement.
- Host count reads, accepted-ID copies, symmetrization, and canonical sort stay
  visible because they belong to the public denominator.

## Table contracts

- Table 1: G2B external full-public result.
- Table 2: G4C resident dynamic-router attribution.
- Table 3: G5 same-artifact rejection; must remain adjacent to the distinction
  between Tables 1 and 2.
- Table 4: H2B multi-vector boundary, included only if the extension appears in
  the paper.

## Prohibited implications

- No figure or caption may state or visually imply that G2B and G4/G5 are one
  timed binary.
- No exactness mark may be attached to FaSTED.
- No G5 speedup may appear as formal or paper-facing performance evidence.
- No conventional tree/index, multi-vector speedup, arbitrary-dimension, or
  multi-GPU claim may be added as packaging.
