# G12: Constructive Strong-Baseline Kill Test

Date: 2026-09-04. Freeze before implementing or running the CPU reference.

## Question

Does factorized grouped distance evaluation inherently require building an
O(E)-sized eligible-pair list before applying the vector predicate? If an
ordinary streaming plan avoids this, avoiding that list is not a differentiated
mechanism. This is a semantic/representation test, not GPU performance work.

The related-work audit found FFX (SIGMOD 2026), in addition to FDB, Kuzu,
TigerVector and VecFlow. Do not claim that FFX implements an exact GPU distance
join; its semantic extension is an LLM consumer. The baseline below is our
explicit composition of standard operations, not a measured FFX configuration.

## Frozen mathematical contract

- Factors are lists `(w, Qw, Bw)` with unique endpoint IDs inside each list.
  Different witnesses may overlap arbitrarily, including identical rectangles.
- Return each `(q,b)` once iff some witness names it and the endpoint-only,
  deterministic predicate `sum_d (q[d]-b[d])^2 <= T` is true.
- No witness score, bag multiplicity, top-k, witness provenance, side effects,
  NULL, NaN, infinity, or concurrent feature updates are in scope.
- For this CPU fixture only: signed integer coordinates in [-8,8], D=8. All
  distances and norm/dot intermediates fit int64. They are also exactly
  representable as binary floating-point values, but this is NOT a test of
  arbitrary FP32 input or a certification of a GPU numerical implementation.
- Reference keeper: independently materialize eligible endpoint pairs and use
  Python-integer direct squared differences. This is the correctness oracle,
  not the strongest proposed physical execution plan.
- Strong plan: iterate factor tiles, gather only tile inputs, compute norm/dot
  distances, filter immediately, insert only passing endpoint IDs into an
  exact output set. No all-candidate list/bitmap/CSR or reject-cache is allowed
  inside this plan. Count all repeated score evaluations and output insertions.

## Frozen fixtures and execution

NumPy PCG64 seed 20260904; four explicitly synthetic fixtures:

1. Mixed overlapping factors: Q=23, B=29, D=8; 9 random factors, plus a
   repeated rectangle and empty factors. Force one equal vector pair.
2. Twelve identical full rectangles: Q=64, B=80, D=8; one forced equal pair.
3. Ragged disjoint query groups: Q=33, B=41, D=8; nonuniform base subsets and
   one empty factor; one forced equal vector pair.
4. Empty relation: Q=23, B=29, D=8; no witness has both sides nonempty.

All fixtures use thresholds [-1, 0, 64, 4096], tile edges [1, 7, 16], and
forward/reverse factor order: 4 x 4 x 3 x 2 = 96 reference-plan runs in one
process. No favorable fixture/threshold selection after measurement.

For every run verify final sorted IDs, accepted occurrence count, total score
count W, no rejected-pair retention, score-tile high-water <=t^2, and final
output state cardinality Z. The high-water counter measures logical score
slots, NOT process RSS, allocator bytes, or GPU memory. Python/NumPy temporary
arrays must be bounded by tile/input/output size by source inspection.

Two negative controls:

- Bag semantics: multiple witnesses for one passing endpoint pair produce
  multiple rows; DISTINCT changes that output. No bag equivalence claim.
- Factor summaries are insufficient: two query-vector sets with equal sums
  and equal per-vector norms can have different exact matching endpoint sets
  against the same database. Do not replace the pair predicate with a test on
  factor means/sums and claim an exact algorithm.

## Decision rule

If the algebraic identity and all 96 checks hold, reject the necessity of
pre-predicate O(E) pair materialization for this contract. Also reject the
claim that our representation-saving proposal alone beats this baseline by
avoiding such materialization. Do NOT infer a GPU speed, universally best
plan, impossibility of new optimizations, or exact-real support for arbitrary
FP32 inputs. Any mismatch must be preserved and fixed before the conclusion.

Novelty remains a separate gate. A future candidate must remove a cost this
streaming baseline actually pays, rather than comparing against a purposely
materializing strawman. G11's failed gate and positive structure stay intact.

## Resource and evidence

- Host: local Mac. Python/NumPy versions and hashes recorded at runtime.
- No GPU/remote computation, new dataset, timing campaign, warmup, SASS,
  sanitizer, library engine, device lock, or manuscript (all N/A).
- Costs still charged conceptually: factor building, repeated input gathering,
  repeated distances W, padding for a future TC backend, numeric refinement,
  accepted occurrences, output hash/sort, allocation, and final export.
- Scope excludes comparing Python wall times. Source, raw results, fixture
  hashes, all failed checks if any, and decision are retained separately.
