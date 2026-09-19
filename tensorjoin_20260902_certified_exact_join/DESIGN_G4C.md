# Design G4C: cross-dataset dynamic-router timing anchors

Date frozen: 2026-09-03

## Thesis-local question

Does the certified FP32 middle stage reduce complete resident-GPU exact-join
latency beyond the frozen CIFAR cell when dataset dimension and ambiguity count
change? G4C does not claim that the three-stage path always wins. A loss on a
low-dimensional anchor is a routing boundary, not permission to censor it.

## Deterministic compact subset

Select `N=4096,k=64` for all three G4B-R1 datasets. This rule is independent of
timing: it takes the largest scale and largest output density on every dataset,
maximizing validated fallback pressure while retaining one native-dimensional
anchor per public source.

| Dataset | D | G3B ambiguity | Candidate FP64 residual | Final upper IDs |
|---|---:|---:|---:|---:|
| SIFT | 128 | 39,810 | 83 | 135,172 |
| CIFAR-GIST | 512 | 109,499 | 117 | 135,168 |
| Fashion-MNIST | 784 | 27,349 | 63 | 135,168 |

The count table is opportunity evidence, not performance evidence.

## Frozen variants and denominator

- Keeper: ragged-safe G3B certificate, read the actual ambiguity count, launch
  direct FP64 for exactly that count, then read the final output count.
- Candidate: identical G3B certificate, read the actual ambiguity count, launch
  certified FP32 for exactly that count, read its actual residual count, launch
  FP64 for exactly that residual, then read the final output count.
- Both use resident input, quantization metadata, schedules, and output buffers.
- Host `perf_counter` starts before counter zeroing and ends after the D2H final
  count. It includes dynamic count transfers, their stream synchronizations,
  every stage launch, and final completion. It excludes ingest, quantization,
  resident setup, output copy, and sorting identically.

Every retained process validates both variants against the independent upper-ID
oracle before and after timing.

## Work, movement, and control ledgers

| Ledger | Keeper | Candidate | Invariant/delta |
|---|---|---|---|
| Dense Tensor-Core work | one triangular G3B pass | same | invariant |
| Pairwise FP32 work | none | G3B ambiguity count | added |
| Pairwise FP64 work | G3B ambiguity count | G3C residual count | removed almost entirely |
| Dynamic D2H counts | ambiguity + final | ambiguity + residual + final | one added synchronization |
| Stage launches | G3B + FP64 | G3B + FP32 + optional FP64 | one added launch |
| Resident source/metadata | identical | identical | invariant |

## Lower-bound and boundary hypothesis

The shared G3B pass provides a common floor. Above it, the keeper prices FP64
work proportional to `ambiguity*D`; the candidate substitutes FP32 work at that
shape and adds one host synchronization plus one launch. The candidate should
benefit most when FP64 arithmetic dominates and may lose when low D or a small
ambiguity makes control/launch overhead dominant. This is a hypothesis, not a
claim, and all three anchors remain in formal execution if the screen gate
passes.

## Ownership, live set, and ready graph

Kernel ownership, tile shape, accumulator state, and ragged masking are exactly
`DESIGN_G4B_R1.md`. Each variant owns distinct result, ambiguity, FP64, and
counter buffers. No buffer is shared across variants.

```text
resident source + quantization ready
-> zero variant counters
-> common G3B completes
-> D2H ambiguity count ready
-> keeper FP64 OR candidate FP32 completes
-> candidate D2H residual count ready
-> candidate FP64 completes when residual > 0
-> D2H final count ready and timed call ends
```

Every `.item()` is an intentional dependency/synchronization edge. No overlap is
claimed. The process retains raw timing order so direction and drift can be
audited.

## Screen and promotion ladder

1. Six clean screen processes: two balanced orders per dataset, 10 warmups, 50
   retained calls, and 200-call sustained loops per variant.
2. All six must pass exact pre/post validation and physical-GPU1 isolation.
3. At least two datasets must exceed 1.10x on both the geometric mean of process
   medians and the geometric mean of sustained ratios.
4. If admitted, audit the exact runtime-selected G3B/G3C/FP64 cubins and freeze
   eight balanced formal processes for all three anchors. Screen numbers are
   never paper claims.

Formal promotion will require paired process-level confidence intervals,
correctness, clean isolation, generated-code identity/resource evidence, and
retention of any losing anchor.
