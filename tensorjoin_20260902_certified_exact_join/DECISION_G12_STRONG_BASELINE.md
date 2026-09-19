# G12 Decision: Park the Factor-Preservation-Only Thesis

Date: 2026-09-04.

## Conclusion

**Reject "avoid expanding all eligible candidate pairs" as the differentiated
mechanism for the current factor-preserving vector-join proposal.** A standard
streamed factorized comparator already avoids that pre-predicate allocation.
The proposed cost saving therefore does not distinguish our plan from a
properly constructed baseline. No GPU experiment or paper promotion is admitted.

This is not a claim that FFX implements the whole proposed GPU operator, nor
that every relational/vector optimization is impossible. It closes the current
story; a different mechanism needs a new Gate-0 card, not another name.

## Strongest evidence

For a deterministic endpoint-only threshold predicate, the desired result is
the union of the passing endpoint pairs from every factor. Hence an ordinary
plan can retain the factor lists, compute one distance tile at a time, filter
immediately, and deduplicate only passing output pairs. It never needs the
entire pre-predicate eligibility CSR/list. The proof and full cost ledger are
in `DESIGN_G12_STRONG_BASELINE.md`.

The new literature finding strengthens the novelty boundary: FFX is a current
factorized pipeline/consumer prior, not an exact GPU distance implementation.
Read `SOURCES_G12_STRONG_BASELINE.md` and its pinned source evidence.

## CPU executable check

The frozen 96 runs all match an independent materializing Python-integer
direct-distance oracle. They cover four synthetic factor structures, four
thresholds, three tile sizes, and both witness orders. This is exact integer
semantic validation, NOT public-vector performance or arbitrary-FP32 proof.

Example: twelve identical 64x80 rectangles, D=8, tile edge 16:

| Squared threshold T | Unique eligible E | Score evaluations W | Passing occurrences W+ | Unique output Z | Score tile slots high-water |
|---:|---:|---:|---:|---:|---:|
| -1 | 5120 | 61440 | 0 | 0 | 256 |
| 0 | 5120 | 61440 | 12 | 1 | 256 |
| 64 | 5120 | 61440 | 288 | 24 | 256 |
| 4096 | 5120 | 61440 | 61440 | 5120 | 256 |

These counters explicitly preserve the repeated computation: the comparator
does NOT compute only E scores. It retains at most one score panel plus input
tiles and the growing output set. The high-water value is a logical counter,
not RSS or GPU memory. The same Python process also runs a materializing
oracle; no process-wide memory advantage is asserted.

The dense-output row matters: Z=E and repeated output hash attempts remain.
Avoiding pre-predicate materialization does not make output construction free.

## Two shortcut boundaries also checked

1. Bag semantics do not survive DISTINCT: one passing endpoint pair under two
   witnesses represents two bag rows but one set row. This comparator targets
   an existential endpoint-pair result, not path provenance or bag counts.
2. Factor sums are insufficient for exact pair thresholds. Query vectors
   `[(1,0),(-1,0)]` and `[(0,1),(0,-1)]` have the same sum and per-vector norms,
   but against `[(1,0),(-1,0)]` at T=0 produce two versus zero matching pairs.
   This refutes the aggregate-summary shortcut, not richer indexing mechanisms.

## Three separate decisions and retained evidence

| Level | Status | Exact scope / reopen condition |
|---|---|---|
| Implementation | CPU comparator correct on 96 synthetic runs | Not a GPU implementation; no timing, sanitizer, sustained, or real-FP32 admission |
| Mechanism | Claimed necessity of O(E) pre-predicate materialization refuted | Endpoint-only deterministic threshold and set output; other contracts require separate analysis |
| Paper thesis | Factor-preservation-only novelty claim rejected; branch parked | Reopen only for a differentiated cost-removal mechanism beyond this comparator and nearest prior art |

G11's real common-reference representation advantage and dense-factor coverage
remain valid. They describe standard factorization's opportunity, not an
advantage unique to us. G11's failed duplicate-removal gate is not relabeled a
pass. No new OGB sampling, radius tuning, GPU run, or empirical speedup is added.
All older TensorJoin positive/negative results and the numerical infrastructure
remain available and unchanged.

## What the baseline still pays

Factor construction, repeated vector gathering/packing, W score evaluations,
backend tile padding, numeric refinement, W+ output hash attempts, allocation,
sorting and export remain real costs. A future contribution must remove one of
these with a new mechanism, and prove it against both this streamed plan and
deduplicated-CSR / masked-dense controls under the same contract. Missing such
a mechanism is uncertainty; it is not a license to assert every variant slow.

Do not resume by tuning group size, adding conventional trees, converting the
backend to Tensor Cores, widening datasets, or describing an already-absent
candidate buffer as our saving. None reopens the current thesis by itself.

## Recommended next research action

Return to core-problem selection instead of writing another TensorJoin story.
Keep the tested kernels, oracle/data fixtures, and this strong comparator as
infrastructure. A fresh candidate must state the nearest prior art, the cost
that the strong baseline actually incurs, the proposed non-incremental
mechanism, and a cheap falsification test before new GPU work. No next
direction is declared novel or approved in this audit.

## Evidence

- Protocol: `PROTOCOL_G12_FACTORIZED_BASELINE.md`, frozen SHA-256
  `96206c0db252774e0c02de20a19389e118ae192af00f543374b686d2d6c766ef`.
- Code: `src/check_g12_factorized_baseline_cpu.py`, SHA-256
  `41b63bdd991dbe04356ba942bdfb28437e9b77dd055e3e4e214f5ef2656d662f`.
- All 96 runs and negative controls:
  `results/g12_factorized_baseline_cpu_a0.json`.
- Raw log: `raw/g12_factorized_baseline_cpu_a0.log`.
- Source snapshots, G11 input snapshot, state backup and evidence manifest:
  `artifacts/g12_factorized_baseline/`.
- No manuscript or Overleaf changes. Remote mirroring, if verified by its
  receipt, is evidence storage only, not a remote experiment.
