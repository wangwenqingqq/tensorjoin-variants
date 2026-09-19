# G2B formal public-denominator campaign supplement

Date: 2026-09-03

Parent frozen protocol: `PROTOCOL_G2.md`.

Status: frozen before formal execution.

This supplement resolves implementation details left open by the parent. It
does not change the public denominator or acceptance thresholds.

## Methods and order

Eight fresh-process rounds use a rotation plus reverse-rotation schedule:

| Round | Position 0 | Position 1 | Position 2 | Position 3 |
|---:|---|---|---|---|
| 0 | GDS | MiSTIC | FaSTED | TensorJoin |
| 1 | MiSTIC | FaSTED | TensorJoin | GDS |
| 2 | FaSTED | TensorJoin | GDS | MiSTIC |
| 3 | TensorJoin | GDS | MiSTIC | FaSTED |
| 4 | TensorJoin | FaSTED | MiSTIC | GDS |
| 5 | FaSTED | MiSTIC | GDS | TensorJoin |
| 6 | MiSTIC | GDS | TensorJoin | FaSTED |
| 7 | GDS | TensorJoin | FaSTED | MiSTIC |

Every method occupies each position twice. TensorJoin appears before and after
each exact keeper equally often. FaSTED is context only and never participates
in the acceptance gate.

## Isolation and attempts

- Hold `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock` for the campaign.
- Before every process, record preflight state and require 30 continuous seconds
  without a compute process on physical GPU0.
- Monitor GPU0 every 0.25 seconds. A PID is allowed only if it is the launched
  process or a descendant.
- If a foreign PID appears, terminate only the launched process group, retain
  the contaminated attempt, wait for a new 30-second quiescence interval, and
  repeat the same method/round/position with an incremented attempt ID.
- At most three contamination attempts are allowed for one slot. A correctness,
  capacity, or non-contamination process failure stops the campaign rather than
  being silently replaced.

## Estimators

For every method, report all eight raw seconds plus linear-interpolated p10,
median, and p90. For each exact keeper independently:

1. compute per-round paired speedup `keeper_seconds / tensorjoin_seconds`;
2. report 8-round process wins and losses;
3. report the geometric mean of the eight paired speedups;
4. bootstrap the eight paired log-speedups by resampling rounds with replacement,
   eight draws per replicate, using NumPy `PCG64` seed 20260903 and 100,000
   replicates;
5. exponentiate the 2.5 and 97.5 percentiles of replicate mean log-speedup for
   the seeded 95% interval.

The faster exact keeper is the one with lower predeclared eight-observation
marginal median. Acceptance requires TensorJoin to win at least 7/8 rounds
against both exact keepers and the bootstrap lower bound to be at least 1.50x
against that faster keeper, in addition to exactness, safety, and capacity gates.

## FaSTED context

FaSTED uses the same public start/end boundary but FP16 input and FP32
accumulation. Report each run's pair count, precision, recall, F1, exact-only
pairs, approximate-only pairs, latency distribution, and source/license caveat.
It remains neither an oracle nor an exact keeper.

