# Claim-evidence ledger

Experiment ID: tensorjoin_20260908_directional_envelopes_small_exact.
All rows concern one60Kx512 FP32 dataset, one threshold, one block size, and the
three predeclared orders. Last verified2026-09-08 Asia/Shanghai; hostgpu-host-8.
Sources and bounds belong to PROTOCOL.md / NUMERICS.md, not a novelty claim.

| ID | Claim | State | Evidence | Counterevidence / allowed wording |
|---|---|---|---|---|
| DGS-S1 | Hadamard reordering concentrates the admitted answers into46985 of440391 tiles | measured | results/audit_a0.json; artifacts/orders.npz; oracle_occupancy.npz | Ground-truth-only empty fraction89.3311%; not a deployable filter |
| DGS-S2 | Directional envelopes safely reject10655 tiles on this finite workload | measured | results/structural_a0.json; audit_a0.json; masks.npz | 2.4194% of scheduled tiles, zero lost admitted IDs; no universal exact-real claim |
| DGS-S3 | This frontend provides enough pruning for GPU prototyping | rejected | Frozen20% rule in PROTOCOL.md vs2.4194% observed | Stop the fixed frontend; preserve nonzero pruning and different-bound reopen condition |
| DGS-S4 | Filtering/indexing is net faster than strong fused FP16 | unknown | no new A/B performance evidence | CPU costs and old GPU results cannot establish a same-contract speedup |
| DGS-S5 | Cross-coordinate information loss limits these boxes | inferred | 89.3311% ideal empty vs2.4194% certified, with64=512 prefix counts | Diagnosis consistent with interval relaxations; not proof that all geometry-based summaries fail |
| DGS-S6 | A new publication-level mechanism has been established | unknown | no evidence; VA-file/GTS/PathWeaver are known relevant precedents | Neither a structural ceiling nor ordinary reordering+filtering admits novelty |

## Negative-evidence record

Target: complete fixed-FP64-predicate similarity join. Keeper: unchanged admitted
fused FP16 output contract. Rejected mechanism: per-coordinate interval
summaries, raw/Hadamard spaces with fixed variance-prefix filtering and64-row
axis-tree layout. Added costs: transform, tree construction, summary generation,
block-pair bounds and eventual gathering; measured count benefit only2.4194%.
Scope: this data/threshold/layout family only. Implementation status: CPU built,
unit/envelope/random-pair/rational/full-ID coverage passed. GPU filter status:
not implemented, no sanitizer/SASS/Graph/stress/performance promotion gates.
Thesis impact: the fixed frontend is not an admitted rescue; stronger joint
summaries remain untested and need their own novelty kill and frozen experiment.
