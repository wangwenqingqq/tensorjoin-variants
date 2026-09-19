# C0B Design Card

## Objective and latency budget

- Keeper: ~331 microseconds median from B0Y.
- Candidate target for 1.5x: <=221 microseconds.
- Scan: C0A measured 39.6 microseconds before compaction.
- Remaining budget: ~181 microseconds for counter reset, sparse atomics,
  count synchronization, 1,608-pair FP64 refinement, and result append.

## Ownership

- Scan program: owns one `64x64` dot tile and its certificate decisions.
- Global result counter: allocates slots for direct accepts and refined accepts.
- Global ambiguity counter: allocates slots in the ambiguity buffer.
- Refinement program: owns one ambiguous pair and its full `K=1024` FP64
  reduction. It appends only when the exact predicate passes.
- Overflow counter: records any out-of-capacity append attempt.

## Persistent buffers

| Buffer | Dtype/capacity | Owner | Last use |
|---|---|---|---|
| Counters | 3 x INT32 | reset/scan/refine | final result count |
| Result IDs | 8,192 x INT32 | scan/refine | consumer |
| Ambiguous IDs | 8,192 x INT32 | scan | refinement load |

No dense INT32 score, FP32 distance, FP32 bound, or status matrix exists.

## Ready graph

```text
counter reset -> scan MMA/certificate -> ambiguous count visible to host
-> exact-size refinement launch -> result count/output ready
```

The host count synchronization is intentional and included in timing. It avoids
launching thousands of inactive FP64 programs and provides a clear phase edge.

## Expected deltas versus eager candidate

- Remove 8 MiB INT32 score output and all dense bound intermediates.
- Remove two dense `nonzero` scans over 2,097,152 statuses.
- Add about 1,689 sparse atomics in the scan at the frozen workload.
- Refine exactly 1,608 pairs, with one program per pair.
- Store about 2,120 compact IDs total rather than a 2 MiB status matrix.

## Verification ladder and reject boundary

Compile -> oracle equality -> SASS -> memcheck -> 1,000-launch stress -> locked
eight-process A/B. Any overflow, unsafe direct decision, duplicate/missing ID,
dense intermediate, sanitizer error, stress drift, or lower confidence bound
below 1.5x rejects C0B.
