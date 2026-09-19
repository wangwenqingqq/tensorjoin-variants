# Precision-routing problem-existence test: no material plan-switch evidence

**Decision:** the frozen schedule-only experiment did not establish a material
INT8-versus-FP16 plan reversal, despite a strong, verified intervention in batch
uncertainty distribution. A dynamic selector is not justified by this result.
This is a bounded negative result, not a universal rejection of precision routing.

Experiment ID: `tensorjoin_20260908_precision_routing_schedule_existence`.
Predecessor: `../precision_routing_20260908_layout_existence/REPORT.md`.
Status: completed, independently re-audited from raw artifacts, not paper-promoted.

## What was tested

Question: for an unchanged exact-answer task and two fixed precision plans,
can uncertainty grouping cause robust opposite winning plans, leaving material
headroom over always using the best fixed plan?

The original 60,000 x 512 finite-FP32 CIFAR-GIST input and threshold
`T = 25921/65536` were unchanged. The reference is the original fixed FP64
direct-difference predicate, not an unrestricted exact-real arithmetic claim.
All 440,391 upper-triangular 64 x 64 tiles were computed once, with 4,096 tiles
per dispatch batch. No candidate pruning or new CUDA program was introduced.

F8 uses the retained INT8-first path; F16 is the strong matched fused FP16-first
control. Both use the same downstream FP32/FP64 refinement and output contract.
Each call freshly prepares its precision metadata and produces the complete
canonical directed output including self pairs.

Only the visit order and batch grouping of identical physical tiles changed:
original order, uncertainty-clustered order, and uncertainty-interleaved order.
Row positions, vector values, pair orientation, per-tile arithmetic and kernels
were unchanged. These schedules use post-execution INT8 uncertainty counts;
construction cost is excluded. They are diagnostic oracle interventions, not a
deployable scheduler or neutral learned policy.

## The intervention was real; quality stayed fixed

| Metric | Original | Clustered | Interleaved |
|---|---:|---:|---:|
| INT8 U1 count CV across full batches | 0.130726 | 0.668967 | 0.000465 |
| FP16 U1 count CV across full batches | 0.132839 | 0.662593 | 0.032780 |

The INT8 CV range is 0.668502, comfortably passing the predeclared manipulation
gate. Within each plan, complete uncertainty-ID sets and physical-tile histograms
are exactly identical across schedules, not merely equal in cardinality.
F8 U1 = 1,827,007; F16 U1 = 86,029; both U2 sets contain the same 1,734 pairs.
Thus, a small terminal FP64 queue remains common to both plans.

All 612 timed/warmup calls returned 3,926,078 canonical IDs with reference SHA256
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
This full-scale answer check is distinct from the sanitizer check: memcheck,
initcheck, synccheck and racecheck each passed six 175-tile composites covering
all schedule/plan cells, with full-size input metadata and retained ragged tiles.
The sanitizer runs do not constitute full-scale exhaustive sanitizer coverage.

## Frozen performance result

Ratio R = T(F16) / T(F8); values above 1 favor INT8. Primary estimates use each
process/cell's median of seven retained calls. Repeat estimates use the sum of
eight complete calls per cell with both visit and method orders reversed.
Intervals are 95% order-stratified process-bootstrap intervals, as frozen.

| Schedule | Primary R [95% CI] | INT8 process wins | Repeat-8 R [95% CI] | INT8 process wins |
|---|---:|---:|---:|---:|
| Original | 1.013562 [1.011979, 1.015273] | 6/6 | 1.010369 [1.007032, 1.013286] | 6/6 |
| Clustered | 1.049323 [1.013281, 1.116849] | 6/6 | 1.011255 [1.008456, 1.014061] | 6/6 |
| Interleaved | 1.011049 [1.010487, 1.011840] | 6/6 | 1.002831 [0.975696, 1.030721] | 4/6 |

The predeclared existence gate required one schedule with CI lower > 1.05 and
another with CI upper < 1/1.05, at least five of six consistent process wins,
consistent order groups, and the same reversal in repeat-8. **It failed.**
No schedule passed either material-winner condition.

The clustered primary phase contains a retained slow FP16 process (p00 ratio
1.222517). Interleaved repeat-8 has opposite order-group signs, with geometric
ratios 1.040886 and 0.966167 for the two primary-order groups. These observations
were not removed, replaced or averaged away into a stable reversal. The actual
method order reverses in repeat-8; stored group labels identify primary order.
The present logs do not establish the cause of these tails.

| Empirical free-selector headroom over best fixed plan | Ratio [95% CI] |
|---|---:|
| Primary, equal weighting of the three schedules | 1.000000 [1.000000, 1.000000] |
| Repeat-8, same weighting | 1.002848 [1.000000, 1.004596] |

This selector is a post-hoc empirical oracle over two plans, not a theoretical
upper bound over all plans or a held-out online planner. The measured headroom
does not meet the separate 5% practical gate.

### Timing denominator and material limitation

The primary timer is complete warmed prearranged-host-input to canonical-host-
output wall time: allocation, tile-index creation/reordering, fresh precision
metadata, H2D, scan, queues/refinement, counters, D2H and sorting are included.
File I/O, context/JIT, experimental schedule construction and validation hashing
are excluded. This is not kernel-only or GPU-resident throughput.

The frozen wrapper also includes a common, redundant identity-ID mapping and
second sort inherited from the failed row-layout experiment. Its outer-minus-
inner increment has median 85.068 ms in the primary phase and 84.971 ms in
repeat-8. This dilutes hardware-only differences and is not required by a
minimal identity-layout production operator. It was not removed after seeing
results. Inner diagnostic geometric ratios also favor INT8 on all three
schedules in both phases, but are not substituted for the frozen denominator.

## Interpretation and next falsifiable step

**Measured:** batch uncertainty distribution changed substantially while the
answer and per-plan uncertainty sets stayed fixed; no robust material winner
reversal emerged in the frozen complete-operator comparison.

**Inference:** do not build or advertise an uncertainty-distribution-based
INT8/FP16 selector on the strength of this workload. The negative result is
stronger than a failed weak intervention, but it covers only two fixed plans,
one dataset, one threshold, three schedules and one GPU.

**Not established:** uncertainty alone causes latency changes; precision routing
is generally unnecessary; an online planner is novel; tensor-core routing has a
new external-system speedup. Scheduling jointly changes locality, queue order
and output layout. This campaign does not compare against a non-routing plan,
change uncertainty population size, or test other precision-plan families.

**Reopen condition:** a separately frozen, cheap threshold/boundary-density
screen must exhibit reproducible material opposite winners under an independently
checked equal-answer contract. Only then budget a planner or broad evaluation.
Do not reinterpret mere row reordering, a smaller FP64 queue, or a favorable
single-process tail as proof of the missing plan-selection problem. A new timing
scope must be declared as a separate experiment, never substituted retroactively.

## Execution and provenance

- Host: `gpu-host-8`, verified hostname `gpu-host-8`; existing research-user workspace.
- GPU 2: RTX PRO 6000 Blackwell Server Edition, SM120, UUID
  `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245`; driver 590.48.01; existing 600 W cap.
- Python 3.12.3, NumPy 2.3.1, PyTorch 2.11.0+cu130, Triton 3.6.0;
  Compute Sanitizer from CUDA 13.1. No clock/power changes or foreign-job actions.
- Six formal process PIDs: 661278, 661982, 662715, 663400, 664084, 664807.
  All guards passed; 72 warmups + 252 primary + 288 repeat calls; zero replacements.
- Both established GPU locks, 30 s idle admission per process, 4 GiB cap,
  30 minute timeout and descendant-aware occupancy guard were retained.
- At closure, GPU 2 was idle with 14 MiB used; all listed jobs had exited.
  See `raw/post_campaign_host.txt` (2026-09-07 18:44:15 UTC).
- No new Graph, profiler, generalized stress, paper correctness proof, external
  baseline result or production path is promoted. Existing generated CUDA
  artifacts were verified unchanged; only the host visit order changed.

Primary evidence: `PROTOCOL.md`, `results/census_a0.json`,
`results/analysis_a0.json`, `results/samples.csv`, all six `results/p*_a0.json`
and guards, sanitizer logs, source/timing freezes, and
`results/closure_audit_a0.json`. The CPU-only closure audit independently checks
all 612 answers/counts, set and histogram equality, retained compiled identities,
admission hashes and the frozen bootstrap estimator. The predecessor's failed
invariance experiment remains retained separately and is not relabeled a pass.
