# Gate2: matched precision x fusion falsification

Frozen before candidate implementation, compiler or timing observations.
This is a bounded attribution/novelty kill test, not a presumption of novelty.
Gate1 need not pass to preserve useful negative/attribution evidence here.

## Matrix

- F8: original admitted G16 INT8 fused scan, four frozen cubins.
- S8: same INT8 dot tiles and original interval expression, split by a global
  int32 score tile buffer. Same per-pair compaction/counters and downstream.
- F16: conventional FP16 dot, inherited two-sided directed-FP32 interval,
  fused per-tile classification/compaction; same downstream stage2/terminal.
- S16: same F16 dot and interval, split by an FP32 score tile buffer.

All use the same64x64 triangular tile schedule,4096tiles per dispatch batch,
D512,N60000, original output IDs and threshold,128threads/CTA,3stages and BK64.
Do not change original F8 to make it look better. Within each precision,
preparation/layout, useful dot work, interval rule, compaction, host counters,
refinements, transfers and sorting match. Different native INT8/FP16 geometry,
metadata rules and required code layouts are part of the precision factor, not
silently declared identical. S8 and S16 add one4-byte score per computed cell,
write+read; F8/F16 do not materialize a global score panel. FP16 reads row-major
half operands (no artificial transpose cost); original INT8 keeps q/qt as used
by F8. All input-dependent preparation remains charged.

The library C from Gate1 remains an additional external-to-factorial dense
control. Do not call C->F16 differences 'fusion alone': tile engine, batching,
counter lifecycle and scheduling also change. The S16->F16 comparison isolates
the split within the matched custom engine. A slower S16 than library C does
not prove C is optimally implemented or make F16 the best possible FP16 kernel.
No autotuning or cherry-picked kernel search is allowed in this first screen.

## Admission and invariants

- Source/design freeze before compilation. Actual generated code/resources and
  no local spills required; retain failures and version any repair.
- F8/S8 accepted and ambiguous ID sets must match on fixtures/selected tiles;
  F16/S16 dot bits and classifications must match (dump variant + non-dump).
- Fixed-reference full output must match for all4paths. Their per-precision
  stage counts must match through full execution. No count-only correctness.
- Boundary/zero/dense/far/wide-exponent/cancellation and ragged-last32 tiles,
  metadata/FP16 dot exact-rational checks, all selected-pair terminal oracle.
- Four sanitizers on bounded composites;32alternating-input stress per path;
  immutable consumer Graph with calibrated counts, including dot/split/queue
  kernels for custom paths. No general dynamic-input Graph claim.
- Record actual IMMA/HMMA instruction families, static resources and same-round
  selected-kernel NCU LaunchStats/SourceCounters for a fixed full tile batch.
  SASS/NCU is mechanism evidence only. No shared-GPU parallel collection.

## Sampling and decision

Only after admission:8fresh processes, four-method Williams orders repeated2
cycles: [F8,S8,S16,F16], [S8,F16,F8,S16], [F16,S16,S8,F8], [S16,F8,F16,S8].
Per method2warm calls then9retained complete calls. Reverse method-block order
for16repeated calls per method. Full pageable-host to canonical-host scope
matches Gate1.8*4*27=864calls, all validated outside timer. Same locks/idle/
resource rules as Gate1. No failed or slow result replacement; first failure
stops its stage. A separately versioned repair is not a favorable-data filter.

Predeclare per-process ratios from method medians (repeat phase: sums):
R_fused = T(F16)/T(F8); R_split = T(S16)/T(S8);
G8 = T(S8)/T(F8); G16 = T(S16)/T(F16);
interaction = G8/G16 = R_fused/R_split.
Analyze all ratios,8process geometric/arithmetic means,4order groups, marginal
medians, p10/p50/p90 and10,000bootstrap resamples stratified by Williams order,
seed20260908. Complete-call data, not sums of phase medians, are denominators.

Falsification: if equalized fused FP16 eliminates/reverses the INT8 advantage,
reject the current INT8-specific cost-advantage interpretation for this scope.
If R_fused's CI lower>1.10 with>=7/8wins and all order groups>1, and repeat16
passes the same bounds, retain a material matched-fusion advantage. To retain
an INT8-specific fusion-interaction hypothesis also require interaction CI
lower>1.10 in both phases. A numerical miss is not paper-readiness failure of
all conceivable designs; state the exact rejected implementation/mechanism.

Even a statistical pass is NOT a novelty pass. It must be explained by a
specific instruction/traffic/dependency mechanism and distinguished from the
nearest source implementation. Native lower-bit throughput or ordinary fusion
alone is not new. If only those explain the result, Gate2 novelty remains
unadmitted. No additional datasets or paper drafting to rescue that outcome.
