# Directional-envelope small experiment: closed

## Conclusion

**Reject this fixed directional-envelope frontend before a new GPU filter is
implemented. Preserve a separate positive result: deterministic reordering
concentrates the answer into far fewer 64x64 tiles.** The current bound, not the
existence of empty tiles, is the primary structural limitation in this screen.
No novelty or end-to-end speed claim is admitted.

## Full-workload results

Data: 60000 x512 original FP32 vectors, inclusive squared threshold25921/65536.
All440391 upper-triangle 64x64 scheduled tiles, including diagonal and ragged
last32 rows. Fractions below use scheduled tile count, not distance-kernel time.

| Frozen order | Empty tiles from full answer IDs (ideal ceiling) | Tiles certified skippable by best declared envelope | Skip fraction |
|---|---:|---:|---:|
| Original | 29176 (6.6250%) | 0 | 0% |
| Raw-coordinate balanced tree | 307696 (69.8688%) | 0 | 0% |
| Hadamard-coordinate balanced tree | 393406 (89.3311%) | 10655 | 2.4194% |

The ideal ceiling uses ground truth after execution. It is NOT an available
filter, a measured speedup, or a proof that the empty tiles can be cheaply found.
It shows that this reordering changes answer locality substantially without
changing input coordinates, the threshold, or the set of required output IDs.

Hadamard-tree/Hadamard-envelope counts by fixed coordinate prefix:
16:10653 tiles;64:10655 tiles;512:10655 tiles. Combining raw and transformed
bounds by max adds no tiles. These are27 predetermined order/space/prefix rows,
not27 independent processes. Increasing the prefix from64 to512 did not repair
this particular bound. The best mask eliminates43538432 valid upper-triangle
pairs out of1800030000; padded tile-cell savings are2.4194%.

The predeclared admission rule required >=20% tile skips and zero false
negatives. Correctness passed for this workload; selectivity failed. Therefore
no new GPU scheduling/filter implementation, autotuning, or end-to-end timing
was launched. Even free screening would remove only2.4194% of this schedule's
blocks. A uniform-cost work model would give1.02479x for the scan alone; this
is a derived model, NOT a strict latency bound or observed GPU acceleration.

## Evidence and correctness

- CPU structural run: PID642288 on gpu-host-8,46.8883s for the whole screen.
- Unit tests: separable blocks, threshold equality and adjacent FP32 values,
  wide exponents, and Hadamard distance consistency passed.
- Envelope membership checked for all vectors under all three orders and both
  coordinate spaces.49152 random actual member-pair lower-bound checks passed.
- Independent rational replay:32 actual transformed coordinates, all contained
  within the computed outward error envelope (observed error0 in these samples).
- Full admitted output regenerated with unchanged prior F16 operator onGPU2,
  PID642675; guard passed, peak device memory1264MiB. The original frozen
  numerical predicate and compiled-program identities were retained.
- Output3926078 directed IDs (including self), SHA256
  13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495,
  identical to the prior complete-output contract.
- After each permutation's inverse ID mapping, all27 masks remove zero accepted
  output IDs. This is full finite-workload coverage against the admitted result;
  it is not a new independent exhaustive FP64 calculation of every rejected pair.
- Numerical assumptions and explicit guard construction: NUMERICS.md.

## Cost observations and limits

All screening costs below are CPU diagnostic wall times, not candidate GPU
latencies and not comparative performance observations. The implementation
computes512-coordinate summaries and cumulative bounds for all three prefixes
jointly; these are NOT timings of a separately optimized16-coordinate filter.
Hadamard transform:2.0494s; Hadamard tree:1.3072s; transformed block summaries:
2.2506s; transformed block-pair filtering:4.7159s. Full512-coordinate FP64
low/high summary uses7684096bytes per coordinate space/order. Tree construction,
permutations and full transformed vectors incur additional storage and traffic.
Do not divide these CPU costs by previous warmed GPU timing numbers. The
reference regeneration had a fresh JIT/runtime path and was not an A/B run.

## Interpretation and next decision

Measured positive: the chosen direction-space ordering raises ideal empty-tile
fraction from6.63% to89.33%. Measured negative: the interval envelope identifies
only2.42%. Inference: independent coordinate intervals discard within-block
cross-coordinate relationships and are too permissive for this workload.
This is not proof that every deterministic bound, tree, or DGS-derived approach
fails. PathWeaver's approximate direction selection remains a different quality
contract and was not reproduced or benchmarked here.

Do not revive this fixed frontend by adding dimensions or tuning the same tree
on this measured data. Reopen only with a genuinely tighter, independently
motivated safe representation that preserves relevant joint structure, or a
justified changed application predicate. First establish distinctness from
existing approximation/metric-tree filtering; then freeze another structural
screen. Do not claim the89.33% oracle ceiling as implementable savings.

## Artifacts

- PROTOCOL.md (frozen before implementation) and artifacts/source_freeze_a0.json
- src/screen.py, src/obtain_reference.py, src/audit.py
- raw/structural_a0.log, results/structural_a0.json
- raw/reference_a0.log, raw/reference_a0_occupancy.jsonl,
  results/reference_a0_guard.json, results/reference_a0.json
- raw/audit_a0.log, results/audit_a0.json
- artifacts/orders.npz, axes.npz, trees.json, *_envelopes.npz,
  lower_bounds.npz, masks.npz, oracle_occupancy.npz, reference_ids_u64.bin
- CLAIM_EVIDENCE.md, delivery_manifest.json, results/preservation_a0.json

No paper/Overleaf changes; no original PPT changes; no publication or dataset
redistribution. Existing remote data stay in the user's experimental workspace.
