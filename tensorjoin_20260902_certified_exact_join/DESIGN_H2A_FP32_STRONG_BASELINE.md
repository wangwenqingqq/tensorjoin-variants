# H2A Design Card: Certified FP32 Strong-Baseline Kill Test

## Decision question

Does H1-R1's exact INT8 object-first operator retain a decisive advantage over
a substantially stronger exact GPU baseline that scans every token cell by
direct FP32 differences, certifies the resulting squared-distance interval,
and sends only ambiguous object panels to the identical direct-FP64 repair?

H2A is the first strong-baseline screen. It does not substitute for a later
pedantic SGEMM/cuBLAS or external-system comparison.

## Frozen target and contract

- Host/device: `gpu-host-8`, physical GPU3, RTX PRO 6000 Blackwell Server
  Edition, SM120; separate GPU3 lock and foreign-PID refusal.
- Data, deterministic split, normalization, symmetric-Chamfer-squared score,
  thresholds, canonical IDs, and 128x512 object matrices are byte-for-byte H1.
- Candidate is the accepted H1-R1 source hashes: runner
  `11aaa724...0713e`, kernel `c24d6e94...b0db`.
- New baseline and candidate both include dynamic count synchronization,
  direct-FP64 repair, final ID transfer, and canonical host sort.

## Baseline numerical contract

For every token pair, the baseline directly evaluates in FP32:

```text
s = sum_k fl((x_k - y_k)^2)
m = sum_k fl((|x_k| + |y_k|)^2)
r = 2^-14 |s| + 2^-22 |m| + 4096*tiny32
r = r + 2^-22 (|s| + r) + 4096*tiny32
[L,U] = [max(s-r,0), s+r]
```

These are the already-audited G3C-B-R1 conservative constants, now applied to
all rectangular token cells. Object aggregation lifts the squared-distance
interval directly through row/column `min` and directed means in FP64, with an
additional absolute outward guard `1e-10`. Ambiguous object panels use the
unchanged H1 direct-FP64 kernels. Any observed containment violation rejects
the baseline rather than widening constants after measurement.

## Kernel/role and live-set table

| Phase | Owner/grid | Live state | Output |
|---|---|---|---|
| B0 FP32 token interval | one CTA/8x8 token tile, 4 warps, K-step 32 | 8x8 FP32 distance/magnitude accumulators plus one K slab | full FP32 lower/upper D2 matrices |
| B1 object decision | one program/object pair | padded 8x8 FP64 interval panel | direct IDs and ambiguous IDs |
| B2/B3 repair | unchanged H1 kernels | padded 8x8 direct-FP64 panel | appended exact IDs |
| C0--C3 | unchanged H1 candidate | accepted H1 live sets | exact IDs |

Each matrix/output scalar has one CTA owner. Only compaction positions use
atomics. There is no barrier, cluster, or cross-CTA handoff.

## Ready graph and timing scope

Resident inputs/workspaces and one-time compilation are outside timing.
Counter reset -> B0 -> B1 -> charged host ambiguity-count read -> B2 -> B3 ->
charged final count/ID copy and sort. Candidate uses the identical H1 sequence.
The outer monotonic wall timer is the screen denominator; CUDA events are
retained only as diagnostic secondary timing.

## Keeper/candidate and gates

- Strong baseline: certified exhaustive direct-FP32 token scan plus exact FP64
  ambiguous-object repair.
- Candidate: H1-R1 INT8 Tensor Core certificate plus exact FP64 repair.
- Independent oracle: direct-FP64 CPU token differences/object aggregation.
- Correctness: zero token/object containment, unsafe decisions, output
  mismatch, duplicates, nonfinite values, or overflow in all four cells.
- Safety: bounded actual-data memcheck for the new baseline on both D2048 fixed
  and D512 ragged paths; accepted H1 candidate safety must remain hash-valid.
- Performance: 10 warmups and 50 alternating baseline/candidate observations
  per cell. H2A passes only if candidate median is at least 1.25x faster in all
  four cells and wins all 50 paired observation slots per cell.

If H2A fails, close the upper performance thesis before adding a tree. If it
passes, admit H2B pedantic SGEMM/FP32 and larger-scale streaming tests.

