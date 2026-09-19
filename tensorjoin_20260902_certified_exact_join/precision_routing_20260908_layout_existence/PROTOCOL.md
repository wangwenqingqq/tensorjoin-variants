# Precision-routing problem existence: layout-controlled kill test

Experiment: tensorjoin_20260908_precision_routing_layout_existence.
Frozen before new code, census, or timing. This is a bounded existence test, not
a proposed novel planner and not permission to resurrect the rejected interval
frontend. Adaptive floating-point predicates and TC join fusion have established
antecedents; see ../two_gate_campaign_20260908/SOURCES_GATE0_SOURCE_CHECK.md.
No routing algorithm or publication-level novelty is admitted by this screen.

## Hypotheses

H+: at fixed data, threshold, answer and per-plan uncertainty population,
changing physical pair locality produces a reproducible reversal of the faster
complete precision plan. H-: one plan remains preferable, or crossings are too
small/noisy to justify planning. Evaluate both symmetrically.

## Frozen task / controls

- Original N60000,D512 finite FP32, abs(x)<=1, inclusive T25921/65536.
- NPY SHA95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c.
- Same fixed FP64 predicate and logical directed uint64 IDs including self;
  3926078 IDs with SHA13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495.
- Layouts: original, raw_tree, hadamard_tree, exactly the three already frozen
  permutations in ../dgs_directional_20260908_small_exact/artifacts/orders.npz.
  Only row permutation; do not rotate/quantize actual input coordinates or
  change threshold/dimension. The tree is NOT traversed in the timed operator.
- Plans: unchanged admitted F8 and F16 from the previous 2x2 campaign. Both use
  64x64 tiles, all440391 triangular tiles,4096 tiles/batch,108 batches; their
  corresponding interval rules and native metadata plus common FP32/FP64
  downstream remain unchanged. No plan autotuning or source/kernel changes.
- The direct comparison is best of these TWO fixed plans, not best of every
  possible GPU algorithm. F16 is a conventional strong matched adaptive control.
- Expected U1/U2 totals: F8=1827007/1734, F16=86029/1734. Reordering must preserve
  not merely counts but the logical unordered uncertain-ID sets for each plan.
  If it does not, stop this causal test and diagnose; do not call it locality.
- Capture U1/U2 physical tile histograms, occupancy, top10%-tile share, per-batch
  counts, logical set SHA256. This is a separate census, never timed as a plan.

## Hardware / ownership

Host gpu-host-8 (gpu-host-8), existing research-user TensorJoin workspace, no Git
repository at its root. Python @TENSORJOIN_ROOT@/isaacsim6/env/bin/python.
GPU2 RTX PRO6000 Blackwell Server, UUID GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245;
driver590.48.01, torch2.11.0+cu130, NumPy2.3.1, Python3.12.3.
Both existing GPU2 locks;30s idle before every process;4GiB device-memory cap;
30min process timeout; no foreign termination, clock/power changes or old edits.
New files only. Verify old binary/source identities and record actual runtime
compiled identities. Existing safety/SASS evidence is inherited only for the
same generated programs; additionally run four sanitizers on changed-layout
bounded composites and full-workload output/invariance admission before timing.

## Timing contract and schedule

Input is already physically arranged pageable host data plus an explicit logical
ID map. All input-dependent precision metadata, H2D, allocations, full scanning,
compaction, refinement, D2H, physical output sorting, logical ID reconstruction
and final sorting are INCLUDED. Same unconditional ID mapping even for original
layout. File IO, generating the experimental layout, JIT/context startup and
correctness hashing are EXCLUDED. Hence this is a complete prearranged-input
operator comparison, NOT net benefit of building a tree/reordering fresh data.
No cached quantized metadata or output is reused. Keep the inner old-operator
wall timer as a diagnostic; outer complete timer is the primary denominator.

Six fresh processes; layout orders are all permutations:
[O,R,H], [R,H,O], [H,O,R], [H,R,O], [O,H,R], [R,O,H].
Even process index: F8 then F16 in every layout; odd: reverse. For each cell,
2 warmups then7 retained calls. Repeat phase reverses both layout and method
orders,8 additional complete calls/cell. Total612 calls including72 warmups;
252 primary and288 repeat observations. Validate every full output outside the
timer. Retain all observations and guard state. No slow-sample replacement.
A failure stops admission/collection; any repair is a separate recorded version.

## Gates / estimator

1. Quality: all exact logical IDs, per-plan U1/U2 logical sets/totals invariant,
   source/binary provenance, four sanitizer results pass.
2. Manipulation: at least one plan's U1 occupied-tile count changes by >=2x
   across layouts OR its top10%-tile mass changes by >=0.20 absolute. Otherwise
   classify this intervention as insufficient, not disproof of the hypothesis.
3. Existence: for R=T(F16)/T(F8), at least one layout has process-geometric
   ratio95% CI lower>1.05 and another has CI upper<1/1.05, each with>=5/6 wins
   in its indicated direction and both plan-order subgroup signs consistent.
   Repeat8 sums must pass the same reversal. Smaller crossings are not a pass.
4. Practical headroom (separate): per process, equal-weight three-layout sum,
   H=min(sum T8,sum T16)/sum min(T8,T16). Report the geometric mean and CI;
   require CI lower>1.05 in both phases to retain a material selector-budget
   hypothesis. This is a zero-selection-cost empirical oracle, not a deployable
   router, unbiased held-out performance, or a novel algorithm.

Use method median within process/layout for primary, sum8 for repeat. Retain
process raw ratios, marginal p10/p50/p90, arithmetic/geometric means, all order
subgroups. Bootstrap10000 process resamples stratified by F8/F16 order, seed
20260908; jointly resample entire process matrices. State CI limits with6
processes. If quality/manipulation passes but reversal fails, conclude only
'no material plan-switch evidence for this fixed workload/layout/two-plan set'.
Do not generalize to all thresholds, uncertainty populations or algorithms.

## Design / gate boundaries

Only host wrapper/census and canonical-ID remapping are new. CUDA owner table,
accumulator live sets, shared layouts and readiness dependencies are unchanged;
see previous DESIGN_G2.md and MECHANISM_EVIDENCE.md. No new MMA/SASS mechanism
is claimed. Predicted count delta: zero per-plan useful dense arithmetic and
U1/U2 membership; distribution/queue locality and output organization may change.
These joint locality effects are NOT isolated to uncertainty alone; a positive
reversal would require a further attribution test before such a causal claim.
No new Graph support, production dispatch, tree-filter, or planner is installed.
