# Decision G2B Smoke: Full-Scale Exactness and Resource Admission

Date: 2026-09-03

## Decision

**ACCEPT the full Cifar60K exactness/resource smoke; do not yet claim
performance.** GDS-Join FP64, MiSTIC FP64, and the triangular TensorJoin cascade
returned the same 3,926,078 sorted directed IDs, byte for byte, on the frozen
60,000x512 float32 source at epsilon 0.62890625.

Common raw little-endian uint64 SHA-256:
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.

Formal timing remains forbidden until method-specific public adapters implement
the same host-float32-array to sorted-host-ID denominator.

## Exactness evidence

| Method | Directed IDs | Duplicates | Self IDs | Missing reverse IDs | Common hash |
|---|---:|---:|---:|---:|---:|
| GDS-Join FP64 | 3,926,078 | 0 | 60,000 | 0 | yes |
| MiSTIC FP64 | 3,926,078 | 0 | 60,000 | 0 | yes |
| TensorJoin cascade | 3,926,078 | 0 | 60,000 | 0 | yes |

The three systems use different index/search or precision-routing paths. MiSTIC
directly compared its serialized IDs to both retained comparator files and
reported zero missing or extra IDs.

An additional CPU audit recomputed direct float64 squared distances for every
one of the 1,993,039 accepted upper-triangle pairs. It found zero accepted pairs
outside the frozen predicate. The closest accepted distance remained inside by
`3.978849999475287e-8` in squared-distance units. This audit checks false
positives; three-way complete-ID agreement supplies the full-set evidence.

## TensorJoin capacity and work evidence

- Unique upper-triangle comparisons: 1,800,030,000.
- Scheduled 64x64 upper tiles: 440,391 in 108 bounded batches.
- Per-stage capacity: 16,777,216 IDs for each full batch, derived from its
  4,096 x 64 x 64 worst-case element count.
- Direct INT8-certified accepts: 1,277,154 upper pairs.
- INT8 ambiguity: 1,831,461 upper pairs, 0.101746% of the upper triangle.
- FP32 accepts: 681,650 upper pairs.
- FP64 refinements: 69,358, 3.7870% of ambiguity.
- Overflows, invalid/lower-triangle IDs, duplicates, and missing symmetry: zero.
- Torch peak allocated/reserved device memory: 593,544,704 / 612,368,384 bytes.

These are mechanism/resource counts, not timing claims.

## Historical 3,926,074 discrepancy

The pinned FaSTED accuracy JSON reports 3,926,074 FP64-GDS truth pairs, four
fewer than all three live implementations on the frozen fvec source. The cause
is unresolved; an input serialization, source version, or historical execution
contract difference is plausible but not established.

`PROTOCOL_G2.md` explicitly states that FaSTED's published pair count is not an
oracle. Therefore the discrepancy is retained as counterevidence but does not
override three complete, source-specific, byte-identical outputs plus the CPU
accepted-set audit. It must not be silently described as a typo or rounding
error.

The first GDS and TensorJoin smoke runners mistakenly used this diagnostic
historical count in their local exit condition and returned nonzero after
successfully materializing complete outputs. Their JSON/logs are not rewritten.
`results/g2b_smoke_summary.json` performs the protocol-level adjudication and
records this exception.

## Execution integrity

- Host: live `gpu-host-8` through `gpu-host-8`; physical GPU0 only.
- Every admitted GPU execution held the campaign lock and passed a 30-second
  empty-GPU preflight.
- No admitted run observed a foreign GPU0 process, and postflight GPU0 was
  empty.
- A TensorJoin resource-API failure occurred before kernel launch and is
  retained as attempt 0.
- The first MiSTIC attempt was terminated because the monitor misclassified the
  runner's own child binary as foreign. Its temporary output is retained but is
  excluded. The clean rerun admitted only the root PID and descendants.

## Evidence

- `results/g2b_smoke_summary.json`, SHA-256
  `b68f1d56e45d550b2cea60029e21a818c0cbd9b3dd3d84e39fbac27b561a101c`.
- `results/g2b_{gds,tensorjoin,mistic}_smoke.json`.
- `artifacts/g2b/{gds,tensorjoin,mistic}_pairs_u64_le.bin`.
- `results/g2b_accepted_cpu_fp64_audit.json`.
- `raw/g2b_*_smoke{,_preflight,_occupancy}.log` and retained attempt-0 logs.
- `DESIGN_G2B.md` and `PROTOCOL_G2.md`.

## Next admitted action

Build and validate equal-denominator public adapters, then run a cheap
direction-balanced screen. Only if that screen is compatible with the frozen
1.50x lower-bound target should the eight-process formal campaign proceed.
Sanitizer and sustained hash-stability gates remain mandatory before a positive
paper claim.

