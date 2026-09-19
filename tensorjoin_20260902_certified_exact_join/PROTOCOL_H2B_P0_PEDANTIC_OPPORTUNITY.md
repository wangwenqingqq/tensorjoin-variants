# Protocol: H2B-P0 Pedantic-SGEMM Certificate Opportunity

Date: 2026-09-03

## Purpose

Run the cheapest falsification test for the H2B numerical geometry before
writing or profiling a cuBLAS GPU implementation.  P0 asks only whether the
frozen theoretical dot-error radius can remain selective on the four H2A
cells.  It supplies no generated-code, GPU correctness, safety, or performance
evidence.

## Frozen inputs and computation

- Use the exact H1/H2A datasets, deterministic object split, normalization,
  symmetric-Chamfer-squared definition, and target-1/8 thresholds.
- Recompute every token squared distance and object score in direct FP64.
- Compute exact FP64 token squared norms from the stored FP32 inputs.
- Use the `gamma_(2D+2)` and underflow/FP64 guards from
  `DESIGN_H2B_PEDANTIC_SGEMM.md`.
- Because no actual `p_hat` exists in P0, use the conservative worst-placement
  envelope `[d2-2*r_d2,d2+2*r_d2]`.  This accounts for a permitted midpoint
  displacement plus its enclosing radius and is deliberately pessimistic.
- Lift token envelopes through the exact directed-min/mean object algebra and
  count direct accepts, direct rejects, and ambiguous objects.

## Immutable output and stop rule

Write once to `results/h2b_p0_pedantic_opportunity.json`; preserve stdout in
`raw/h2b_p0_pedantic_opportunity.log`.  Record all input/source hashes and all
per-cell ambiguity counts.

- Any interval containment or direct-decision violation rejects P0.
- P0 admits GPU implementation only if every cell leaves at most 5% of object
  pairs ambiguous and the target-8 cells leave at most 2% ambiguous.
- Failure closes the pedantic-SGEMM implementation path unless a new
  non-incremental numerical mechanism is separately proposed.  Constants and
  thresholds may not be relaxed after observing the output.

## Current state

Frozen before P0 execution.
