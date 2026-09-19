# Decision G2A: External Exact and Scalable-Output Admission

Date: 2026-09-03

## Decision

**ACCEPT G2A and admit G2B.** On the frozen 4,096x4,096x512 Cifar60K
subset, GDS-Join, MiSTIC, and the scalable TensorJoin cascade each returned the
same 262,144 sorted directed pair IDs as the independent direct-difference FP64
oracle in two isolated executions.

This is a correctness/capacity result only. It does not establish a performance
advantage or a full-scale systems claim.

## Frozen oracle

- Seeded subset: 4,096 source rows, seed `20260903`.
- Radius: `epsilon = 0.7541135250198396`;
  `effective_epsilon_d2 = 0.5686872086178483`.
- Strict squared-distance gap: `4.6179517532163317e-7`; no tie expansion.
- Directed output including self: 262,144 IDs, exactly 64 per point on average.
- Raw little-endian uint64 pair hash:
  `da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`.

The neighbor distribution is skewed (minimum 1, median 9, maximum 873), so the
average selectivity must not be described as a per-point cap.

## Exact external keepers

| Method | Runs | Exact IDs | Duplicates/invalid | Repeated hash | Isolation |
|---|---:|---:|---:|---:|---:|
| GDS-Join FP64 | 2/2 | yes | 0 | yes | pass |
| MiSTIC FP64 | 2/2 | yes | 0 | yes | pass |

The GDS-Join adapter changes only the compile-time dimension/toolchain target
and uses the upstream Python result API. The MiSTIC adapter selects GPU0, fixes
construction of `std::vector`-containing neighbor-table objects, and serializes
the existing kernel output after inverting its point permutation; it does not
change the kernel or distance predicate. MiSTIC's raw kernel already emitted
all 4,096 self-pairs, so the declared self-normalization path inserted zero IDs.

GDS-Join compressed the subset into one non-empty grid cell, and MiSTIC's final
graph contained one node with 16,777,216 candidate calculations. These are
measured mechanism diagnostics, not general statements about either system.

## TensorJoin scalable candidate

Both candidate executions reproduced the oracle hash and the same work counts:

- direct INT8-certified accepts: 177,618;
- INT8-ambiguous pairs: 204,555 (1.2192% of all 16,777,216 pairs);
- FP32 direct accepts/rejects: 81,380 / 116,825;
- FP64 refinements: 6,350 (3.1043% of INT8 ambiguity);
- all unsafe decisions, final mismatches, duplicates, invalid IDs, and final
  overflows: zero.

The forced geometric-growth path started at 32,768 slots and deterministically
traversed 32,768 -> 65,536 -> 131,072 -> 262,144. The final accepted capacity
was 262,144 with zero overflow. Pair IDs are int64 on device, so the full
60,000x60,000 ID range is representable. No all-pairs status tensor was used.

## Material caveat

The G2A candidate intentionally repeats the dense scan during capacity growth;
its first process also includes Triton compilation. Therefore the recorded
wall times are diagnostic and are forbidden from any comparison. G2B needs a
separately frozen, one-successful-scan capacity estimator and the common
host-array-to-host-pairs denominator before speed can be evaluated.

## Evidence

- `results/g2a_summary.json`, SHA-256
  `92b7975a5b4d7a8a7862d2797a636385318cf3e2822f5726e7daccde05f25c76`.
- `results/g2a_{gds,mistic,tensorjoin}_process_{0,1}.json`.
- `raw/g2a_*_process_{0,1}{,_preflight,_occupancy}.log`.
- `receipts/g2a_artifacts_sha256.json`.
- `receipts/g2a_{gds,mistic}_adapter_build.json` and retained adapter patches.
- TensorJoin runner SHA-256
  `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8`.

## Next admitted action

Freeze a G2B capacity sampler and end-to-end harness, then run one full-scale
correctness/resource smoke per method before any eight-round timing campaign.
If the candidate cannot complete exact full output or the capacity estimator
requires a retry, stop before formal timing and retain the failure.
