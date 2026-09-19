# Next gate: conservative RT-HiSS reference-contract repair

Design boundary after G17, 2026-09-05. Not an implemented or measured repair,
not a novelty contribution and not permission for a 60K timing campaign.

## Hypotheses and immutable controls

Positive hypothesis: retaining RT-HiSS's FP32 prefix rejection and refining only
numerically undecidable candidates can satisfy the frozen FP64-reference
contract at much lower correction cost than replacing all refinement with FP64.
Negative hypothesis: conservative candidate generation or the numerical repair
costs erase that opportunity. Test both, preserving native G17 output and cost
as a separately labeled native-quality control rather than dropping the system.

Keeper: pinned G17 native adapter/binary and all raw failures. New repair must
live in a separate adapter and use the same default RT/refinement work ownership,
point/dimension permutation, batching, compressed output and original-ID map.
A full-FP64 predicate may be a correctness/attribution control, never the only
performance baseline used to make TensorJoin look favorable.

## Freeze before implementation

1. Keep G17's nine exact stored inputs, reference thresholds and expected ID
   arrays, including the additive zero-endpoint witness. Do not adjust epsilon,
   omit error directions, delete the original fixture, or use output count alone.
2. Write a reviewed numerical bound covering FP32 subtraction, FMA accumulation,
   prefix early termination, subnormals/FTZ policy, threshold conversion and the
   declared original-dimension-order FP64 terminal. Derive outward evaluation
   of the bound; do not choose a tolerance from the four observed disagreements.
3. At each rejection checkpoint, reject only when a conservative lower bound
   exceeds the reference threshold. Accept only when the complete upper bound
   is at most the threshold. Replay undecided candidates in the frozen terminal
   arithmetic. Positive-only reranking cannot recover G17's false negatives.
4. Original dimension order matters for the reference algorithm. An FP64 sum in
   upstream variance order is not automatically the same oracle. Measure any
   inverse-map storage, fallback loads, queue or mask traffic and synchronization.
5. Candidate completeness is a separate gate. G17's full candidate coverage is
   observed only on its matrix. Audit RT sphere/AABB construction, radius and
   FP32 intersection rounding before claiming a general conservative search.
   If a test reveals missing candidates, halt the refinement-only admission and
   predeclare the minimum outward-bound repair. Do not silently disable the
   index or infer arbitrary-input completeness from these dense candidates.

## Design card to complete before new CUDA code

- Hardware/shape: same SM120 device, D512 and N <= 4096 correctness gate first.
- Roles: retain upstream CTA/shared query-candidate staging and mask writer.
  Per-pair FP32 prefix, interval state and terminal are owned by the existing
  pair-testing thread; no cross-thread reduction or new barrier is presumed.
- Precision: state exact operations, rounding, compiler flags, subnormal policy
  and terminal order. Exact-real and frozen-FP64 agreement remain distinct.
- Live state: ledger native distance, interval/radius temporaries, inverse-map
  addresses and FP64 fallback accumulator. Identify last use and peak register
  pressure before changing warp count or layout.
- Ready graph: preserve input stage -> barrier -> pair tests -> mask atomic ->
  existing completion -> compression -> D2H metadata/IDs -> canonicalization.
  Audit every early return for participation in existing shared barriers.
- Expected costs: additional prefix comparisons/bounds, fallback fraction,
  terminal coordinates, register pressure and mask/output overhead. Counters
  are diagnostics, not public timing. No Tensor Core speedup attribution here.

## Admission ladder and stop rules

A. CPU bound falsification and same-launch GPU predicate A/B on the retained
   disputes plus high-entropy and boundary cases. Any uncontained reference or
   false direct decision rejects that bound; preserve the exact witness.
B. Full G17 matrix with independent raw-mask reconstruction and original-ID
   equality. Zero FN/FP, candidate omissions, capacity or permutation failures.
   Record counts of safely rejected, safely accepted and terminal-refined pairs.
C. New-source memcheck/synccheck, bounded repeat/pointer churn, then actual
   runtime-selected SASS/resource identity. G17 safety does not certify new code.
D. Only after A-C: remove correctness-only capture in a separately frozen
   performance adapter, retain complete output validation, and charge packing,
   all candidate construction, allocation, fallback, D2H IDs and canonicalization.
   Re-measure controls together in direction-balanced fresh processes. Do not
   compare G17 instrumented wall times or G16's 60K numbers to a 4096 result.
E. Novelty remains Gate 0 before paper drafting or an expensive broad campaign.
   Conventional adaptive repair is baseline normalization, not TensorJoin's new
   contribution. A non-incremental mechanism still needs its own prior-art kill
   test and same-contract advantage; more engineering alone cannot supply it.

No fixed speedup or low fallback rate is predicted as fact. Missing evidence is
unknown. A slow but valid comparator is retained; a subsumed paper thesis must
be changed rather than rescued by excluding it.
