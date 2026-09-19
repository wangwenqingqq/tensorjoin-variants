# G14 decision: retain the static-join positive; target the data path

Date: 2026-09-05. Host: `gpu-host-8`, physical GPU 2.

**Continue from the static-join implementation. The audited G5 artifact again
has positive full-output observations; the current dominant cost is data
preparation, movement/allocation and host canonicalization, not its three
GPU routing stages.** This is a completed diagnostic, not a novelty pass or
formal promotion. The historical 5.285 s observation remains unexplained.

## 1. Completed evidence

All 12 predeclared independent processes completed, with both method orders
and both default and node-0 CPU/memory placement. All emitted the same
3,926,078 sorted directed IDs, with no duplicate, missing-symmetry, invalid-ID
or overflow error. Every guard was clean. All four G5 selected cubin/PTX sets
match retained P4 code; its specialization and stage counts are unchanged.
The MiSTIC binary is unchanged and was measured freshly in all four blocks.
All eight instrumented phase sums close exactly to the recorded public timer.

| Block / placement | G2B (s) | G5 (s) | MiSTIC (s) | MiSTIC/G5 time ratio |
|---|---:|---:|---:|---:|
| 0 / default | 1.093086 | 0.933772 | 5.021749 | 5.377916 |
| 1 / node 0 | 0.927863 | 0.932263 | 5.482189 | 5.880515 |
| 2 / node 0, reversed | 0.921459 | 0.911490 | 5.283864 | 5.796953 |
| 3 / default, reversed | 0.934260 | 0.950725 | 4.742475 | 4.988273 |

These are **instrumented diagnostic ratios**, not a newly admitted production
speedup. Only two observations per method/placement were predeclared; raw
orders, p10/median/p90, marginal ratios, paired geometric means and wins are
retained in `results/summary.json`. There is no confidence interval or claim
of population-level stability. MiSTIC is not claimed to be the strongest
current adaptive-precision baseline.

G5 still sends only 1,827,007 of 1,800,030,000 upper-triangle pairs to the FP32
stage and only 1,734 pairs to the terminal FP64 stage. Thus approximately
99.8985% of upper pairs are resolved by the first stage on this frozen input.
These selective-routing assets survive. This does not establish a new
filtering principle or an all-input exact-real predicate guarantee.

## 2. What the phase evidence actually says

| G5 phase | Observed range across four processes |
|---|---:|
| Analytic quantization and metadata preparation | 546.378–563.696 ms |
| Layout preparation, H2D and device allocation (combined) | 216.768–235.098 ms |
| Final canonicalization and sorting | 96.453–97.891 ms |
| Stage 1 dispatch and blocking count read | 17.574–18.316 ms |
| Stage 2 dispatch and blocking count read | 8.162–8.803 ms |
| Stage 3 dispatch and blocking count read | 6.252–6.975 ms |

The first three rows account for **94.12–94.54%** of each G5 public observation.
Their host resource-accounted CPU time is close to wall time. That is a measured
budget, not yet an attribution of every instruction or transfer. In particular,
the H2D/allocation bucket also includes `codes.T.copy()` and must not be called
PCIe transfer time. CUDA-event spans bracket dispatch and can include host
issue gaps; they are not profiler-isolated kernel durations.

G2B and G5 are close after the first G2B observation. The slower first G2B run
spends 683.640 ms in preprocessing, versus approximately 554–557 ms in its
other three processes. We retain it; no cause is assigned. G5's historical
5.285 s mode does not recur under either placement. Consequently:

- The old slow mode is **not localized** by this campaign.
- There is no basis to call NUMA binding its cure. The bound MiSTIC processes
  are slower than the default ones, and the shared host load also changes.
- The observation that does repeat is the large data-path fraction.
- Micro-optimizing the already-small route kernels is not the first target.

## 3. Decision at three levels

| Level | Decision |
|---|---|
| Implementation | Keep both original runners and every prior result. The additive adapters only instrument and relocate outputs; normalized AST checks and proxy tests pass. No kernel optimization was made. |
| Mechanism | The existing audited route retains exact-oracle agreement and positive external-system observations in this diagnostic. The dominant currently measured cost lies elsewhere in its complete denominator. |
| Paper thesis | Still no new novelty pass, strongest-baseline admission or formal/sustained claim. G10's overlap analysis and G5 P4's failed original gate remain unchanged. Neither automatically erases this positive evidence. |

No manuscript or Overleaf edit, new dataset, threshold sweep, unrelated GPU
job or speculative replacement direction was launched. GPU 1's foreign process
1460887 remained untouched and was present at postflight. GPU 2 returned to
14 MiB, zero utilization, with no compute process.

## 4. Next action, without changing the research problem

1. **Split and reduce measured preparation costs.** First time quantization,
   residual construction, transpose, transfer and allocation separately. A
   cache-sized row-blocked CPU preparation is a cheap initial control before
   attempting GPU preparation. Preserve bitwise codes and conservative metadata
   and charge every preparation/output cost. No such variant was run in G14.
2. **Add the complete adaptive FP32-first control.** Give it the same host-source,
   exact-oracle output, threshold, terminal, compaction, sorting and safety
   contract. Do not credit a generic host optimization as a Tensor-Core-specific
   advantage; compare fair independently optimized data paths.
3. Admit only a separately frozen bounded implementation experiment, with all
   observations retained. If the control removes the comparative advantage,
   retain that result rather than restoring a weaker baseline. If an advantage
   survives, explain the actual removed cost before proposing paper prose.

Data-path engineering is useful but is **not, by itself, a database/systems
novelty claim**. The purpose is to turn the existing positive asset into a
well-attributed system, not to rescue subsumed ideas with packaging.

## Evidence and reproduction

- Frozen scope and schedule: `PROTOCOL.md`, `artifacts/frozen_hashes.json`.
- Exact generated changes: `artifacts/g2b_adapter.diff`, `artifacts/g5_adapter.diff`.
- Process records and raw run: `results/campaign.json`, `raw/campaign.log`.
- Every observation and phase: `OBSERVATIONS.md`, `results/summary.json` and
  per-method child JSON files.
- Independent adapter check: `src/test_adapter_semantics.py`,
  `raw/adapter_semantics_test.log`, `artifacts/adapter_semantics_audit.json`.
- Original sources: `artifacts/source_snapshot/`; referenced raw logs and
  Triton caches remain at their project-relative legacy G5 paths.
- Verified raw receipts: `artifacts/raw_evidence_manifest.json`.

The measurement has already run and refuses overwrite/restart. Reproduction
requires a new labeled, predeclared campaign; do not delete old files to rerun.
The frozen measurement includes five new Python files. The analyzer, test and
archive tools were added afterward and are not imported by the measurement.
Some child/guard fields retain their legacy G2B/G5 experiment labels: the G14
manifest and protocol, not those labels, define this diagnostic's admission.

Negative evidence preserved: G5 P4's failed per-round gate and 5.285 s sample;
no causal NUMA diagnosis; no long-tail/sustained guarantee; no strongest-control
or novelty pass. Reopen the historical variance diagnosis only when its slow
mode is captured with phase evidence or a separately justified distinguishing
test, not by replacing it with favorable new samples.
