# Claim-evidence ledger

Campaign: tensorjoin_20260908_precision_routing_schedule_existence.
Last verified: 2026-09-08 Asia/Shanghai. See REPORT.md for full contract and
results/closure_audit_a0.json for exact verification time, source and binary
identity checks. Local and gpu-host-8 copies are private experiment artifacts.

| ID | Exact claim | Scope / denominator | State | Evidence | Counterevidence / limit | Allowed wording |
|---|---|---|---|---|---|---|
| PR1-C1 | Identical uncertainty memberships survive schedule-only intervention | F8/F16, three schedules, original 60K x 512 and threshold | measured | results/census_a0.json; six census NPZ files; closure audit | Row-layout predecessor failed this same invariant | Identical within each fixed plan, not across precision plans |
| PR1-C2 | Batch uncertainty concentration changes substantially | Full-batch U1 count CV; same physical tile histogram | measured | census manipulation; retained batch counts | Oracle counts; locality and queue order also change | Manipulation passed, not uncertainty-only causation |
| PR1-C3 | Complete timed outputs match the fixed reference | All 612 calls, complete canonical directed IDs | measured | p00-p05 results; closure audit; reference SHA | Not exact-real proof, not all thresholds | All tested outputs match the original fixed FP64 predicate reference |
| PR1-C4 | Changed schedules pass new sanitizer checks | Six 175-tile composites per tool, full-size input metadata | partial | memcheck/initcheck/synccheck/racecheck results and logs | Not all 440391 tiles under sanitizer; no new Graph gate | Selected-tile composites passed four sanitizer tools |
| PR1-C5 | Different uncertainty grouping requires material opposite precision-plan winners | Frozen outer wall time, six processes, primary and repeat-8 | rejected | results/analysis_a0.json; all raw timings | No material winner reversal; interleaved repeats have order-dependent tails | No material plan-switch evidence in this exact scope |
| PR1-C6 | A selector has >5% headroom over the best fixed plan | Empirical per-process oracle, equal schedule weights | rejected | Primary H=1; repeat H=1.002848, CI [1,1.004596] | Not a bound on unseen workloads or plan families | This empirical two-plan selector offers negligible measured headroom |
| PR1-C7 | Building the proposed distribution-based selector is currently justified | Decision inferred from C5/C6 | rejected | Frozen gate failures and REPORT.md | Different uncertainty populations remain untested | Pause this planner until a separate same-contract reversal is established |
| PR1-C8 | Precision routing is universally unnecessary or novel | Other plans, thresholds, datasets, resident scopes | unknown | No evidence from this campaign | This is a bounded existence test, not a novelty result | No universal or novelty claim |

## Retained negative record

- Target: unchanged certified similarity-join answer contract on original 8p GPU 2.
- Keeper/comparator: unchanged F8 and matched fused F16, remeasured together.
- Rejected proposition: distribution-only schedule variation establishes a useful
  need to switch between these plans on this workload.
- Expected work delta: zero total tiles, U1 IDs and U2 IDs; changed batch sizes
  for refinement and changed task/locality order. No dynamic-instruction profile
  was collected, so do not assert an instruction-count or memory-traffic delta.
- Added costs: tile schedule indexing and a common identity remap/second sort;
  oracle schedule construction excluded. All included costs remain in the timer.
- Measured effect: all primary process/cells favor F8; no >5% opposite-winner
  gate passes; repeat free-selector geometric ratio 1.002848.
- Diagnosis: successful distribution intervention, insufficient stable crossover
  evidence; tail cause not determined. No universal mechanism limit inferred.
- Reopen: independently validated, separately frozen threshold/boundary-density
  workload exhibits robust material opposite winners, not just better packaging.
