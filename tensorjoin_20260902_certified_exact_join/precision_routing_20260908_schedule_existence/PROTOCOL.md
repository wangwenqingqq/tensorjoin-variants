# Precision-routing existence: schedule-only intervention R1

Experiment: tensorjoin_20260908_precision_routing_schedule_existence.
Frozen before implementation and timing. Independent successor, not a relaxed
pass for the prior layout experiment. Its full negative/invariance diagnosis
remains in ../precision_routing_20260908_layout_existence.

## Why a different intervention

Raw-vector permutation changed3 INT8 uncertain-pair memberships via operand-
orientation-dependent FP32 scaled-dot rounding; complete outputs were identical.
The failed fixed-membership gate remains failed. Here input coordinates, row
positions, within-tile mapping and pair orientation remain EXACTLY unchanged.
Only the visit order/grouping of the same440391 physical64x64 tiles changes.
No CUDA source/binary is edited; one host run wrapper changes tile-index order.

## Task, plans and census

Use the same original60Kx512 FP32 file (SHA95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c),
T=25921/65536 inclusive, fixed original FP64-predicate logical directed ID set:
3926078 IDs, SHA13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495.
Plans: unchanged previous F8/F16 kernels, metadata and common FP32/FP64 stages.
All440391 triangular tiles are computed once;4096 tiles/batch,108 batches,
last2119 tiles. No geometric pruning, precision tuning or changed threshold.
F8 U1/U2=1827007/1734; F16=86029/1734. Require exact logical membership equality
for each plan across all schedules, not just counts.

Schedules are determined before timing using the original-order F8 U1 tile
histogram already captured by the failed experiment's passing original control:
O=original triangular order; C=stable descending tile U1 count; I=interleaved.
Construct I by splitting C into4096 nearly equal contiguous groups, then taking
one element per group each round. This fills107 full dispatch batches and one
2119-tile tail with broadly distributed uncertainty densities. Use the SAME
schedule for F8 and F16. The F8-derived schedules are intentionally diagnostic,
not neutral learned routers; report that bias and do not claim online selection.
Schedules use post-execution information at zero construction cost. This is an
optimistic controlled intervention, NOT a deployable scheduling optimization.

Record exact U1/U2 sets, physical-tile histograms and per-dispatch counts for
all6 schedule/plan cells. Tile histograms must also be identical within each
plan. Manipulation passes if the per-full-batch U1 coefficient of variation
changes by>=2x with an absolute CV change>=0.10, or the top10%-batch share of
U1 mass changes by>=0.10, for at least one plan. Exclude the ragged last batch
only from CV, retain it everywhere else. This verifies changed grouping, not
changed geometry or a stronger numerical filter.

## Timing contract, hardware and safety

Host gpu-host-8/gpu-host-8; existing research-user workspace; GPU2 UUID
GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245, RTX PRO6000BlackwellServer.
Driver590.48.01, Py3.12.3 NumPy2.3.1 torch2.11.0+cu130. Existing two locks,
30s idle,4GiB cap,30min timeout. No foreign jobs or power/clocks touched.
Verify prior full admission and unchanged generated artifacts. Full output/set
census and four changed-schedule selected-tile sanitizer composites precede
formal timing. Existing SASS/safety evidence applies to unchanged programs;
new wrapper source is frozen. No new Graph support or public performance claim.

Complete warmed prearranged-host-input to canonical-host-output outer wall
scope: allocations, tile index creation/reordering, precision metadata, H2D,
full scan, queues, refinements, counters, D2H, physical sort, unconditional
logical ID identity mapping and final sort INCLUDED. JIT/context, data file IO,
experimental schedule construction and validation hashing EXCLUDED. Each call
rebuilds input-dependent precision metadata. Save inner scan-pipeline wall time
as diagnostic only; outer complete wall time determines the decision.

Six fresh processes, all schedule orders: [O,C,I],[C,I,O],[I,O,C],[I,C,O],
[O,I,C],[C,O,I]. Even index F8->F16, odd reverse. Each cell2 warmups+7 retained;
repeat phase reverse both orders,8 complete calls/cell.612 calls including72
warmups,252 primary and288 repeat. Validate every output, no replacements.

## Predeclared decisions

Same thresholds as failed predecessor; do not lower them after seeing data.
R=T(F16)/T(F8), from per-process cell medians (repeat:sum8). Six-process
geometric/arithmetic ratios, order-subgroup ratios, wins, marginal quantiles;
10000 order-stratified process bootstraps seed20260908, jointly resampling cells.

Existence: one schedule CI lower(R)>1.05 and another CI upper(R)<1/1.05;
>=5/6 wins in each indicated direction, both order subgroups consistent, same
opposite schedules pass repeat8. Quality and manipulation must pass first.
Material selector headroom is separate: H=min(sum_s T8,sum_s T16)/sum_s min(T8,T16)
on equally weighted schedules; require CI lower>1.05 in both phases. This is
an empirical zero-cost oracle, not a held-out or online planner result.
No reversal means only 'no material plan-switch evidence for these two plans,
this workload/threshold and these schedules'. It is not universal disproof.

This intervention jointly affects batch uncertainty, source-vector locality,
queue order and output layout. Positive timing would need further attribution
before saying uncertainty distribution alone caused it. Standard adaptive
filters and TC join fusion are prior art; no novelty pass follows from a switch.
