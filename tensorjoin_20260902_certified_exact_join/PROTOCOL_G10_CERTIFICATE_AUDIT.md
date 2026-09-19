# G10: certificate novelty and semantic-contract audit

Declared 2026-09-04 before the CPU illustration. This is not a GPU performance
campaign, a reopening of G5/G9, or permission to edit paper prose.

## Questions

1. Which existing TensorJoin certificate claims survive direct comparison to
   vector-approximation search, adaptive robust predicates, error-bounded dot
   products, and modern GPU similarity join?
2. Which guarantee is implemented: search completeness, equality to a frozen
   floating-point oracle, or an exact-real threshold predicate on stored FP32
   coordinates?
3. Does the surviving difference identify a non-incremental mechanism, rather
   than a collection of established components and implementation checks?

## Cheap semantic illustration

Experiment ID: `tensorjoin_20260904_g10_fp64_vs_exact_predicate_cpu`.

- Local CPU only; Python standard library; no CUDA import, launch, timing,
  data download, public-dataset resampling, or existing kernel modification.
- Two vectors, D=512. The second vector is zero. The first has only its first
  two coordinates potentially nonzero. Every coordinate is round-tripped
  through IEEE binary32 before evaluation. Squared threshold T=1, inclusive.
- Four predeclared cases: inside (0.5,0), equality (1,0), just outside
  (1,2^-27), clearly outside (1,2^-13). All remaining coordinates are zero.
- Same-process references: exact rational squared-distance sum; ordinary
  binary64 difference/product/sum; outward-rounded endpoints corresponding to
  G9's relative 2^-40 final interval. This models the arithmetic boundary, not
  the Triton reduction implementation or its launch preconditions.
- Expected witness: 1+2^-54 is greater than 1, while its binary64 rounding is
  1. Hence ordinary FP64 acceptance need not imply exact-real acceptance.
- Require exact integer/rational evidence, binary32 representability, interval
  enclosure, correct definitive interval decisions, and preserved unresolved
  boundary states. Record all four cases, source hashes and CPU/runtime state.

The result must NOT be reported as a new GPU reproduction, a measured error
rate on public data, or a failure of the historical FP64-oracle contract.

## Novelty decision rule

Finding established filtering, bounds or precision escalation rejects those
generic standalone novelty claims, not the correctness of the implementation.
An exact match to the entire implementation is not required to expose a
combination-of-known-components risk, but do not claim such a match was found
unless the source actually establishes it.

No GPU expansion or manuscript promotion follows from an unmatched label or a
stricter numerical contract alone. Reopening research requires a one-sentence
mechanism that removes identifiable work beyond an adapted strong comparator,
a same-contract comparison plan, and a cheap falsification test. A conventional
certificate/FP32/exact-fallback control must not be omitted merely because
existing external systems use fixed precision. Missing comparisons are unknown,
not victories or failures.

## Preservation

Keep G2B's external-system result, G3/G4 attribution, G5's failed unification,
G7's pending baseline admission, and both signs of G9 unchanged. Add a separate
audit record and update only the mutable continuation route after backing it up.
