# Protocol G3B-A: Same-shape adversarial and metamorphic correctness

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_gpu_analytic_certificate_g3b_adversarial`

Status: frozen before execution.

## Reason for this supplement

`GATE0_G3.md` requires adversarial correctness before promotion. The original
`PROTOCOL_G3B.md` froze real-data validation, generated-code inspection,
memcheck, and sustained execution but omitted an explicit adversarial step.
This supplement closes that stronger parent-level requirement; it does not
rewrite or replace any completed G3B evidence.

## Frozen candidate and shape

- Candidate source SHA-256:
  `48ca8b1076db31dddf0a60f05564db0a2e1f037a6f007cc988226bce92eff2f6`.
- Physical GPU0, `sm_120`, `4096x512`, identical 64x64x64 upper-tile
  specialization and direct FP64 refinement.
- Expected certificate cubin SHA-256:
  `c80141730aecd650b57be14ecee6422be1477fc704d9a09e0ecc6949a3ad06af`.
- No timing claim is allowed.

## Seven frozen cases

Five metamorphic cases preserve the exact G2A distance relation and derive the
oracle by exact ID mapping:

1. negate every component;
2. seeded row permutation (`20260903`);
3. seeded dimension permutation plus alternating component sign flips;
4. exact power-of-two input/radius scaling by `2^-8`;
5. exact power-of-two input/radius scaling by `2^8`.

Two synthetic cases target arithmetic boundaries while retaining finite normal
stored terms (zero is allowed):

6. all-zero vectors at epsilon `2^-20`, exercising zero/FTZ square-root paths;
7. 2,048 all-`+1` and 2,048 all-`-1` vectors at epsilon zero, exercising
   both `+512*127^2` and `-512*127^2` integer-dot extremes. The exact oracle
   contains precisely the two within-sign complete directed blocks.

## Checks and acceptance

Every case must have finite inputs and finite normal-or-zero scales,
reconstruction norms, and residual bounds. It must report:

- zero duplicate/invalid/overlapping direct or ambiguous IDs;
- zero unsafe direct accepts and zero unsafe direct rejects;
- exact final upper IDs and exact symmetric directed IDs;
- zero capacity/overflow events.

All seven cases must pass in one isolated process after 30 seconds of empty
physical GPU0, with no foreign-process overlap or postflight residue. The
candidate cubin emitted for this process must equal the frozen G3B validation
cubin. Failure is retained and reopens G3B rather than being dismissed as a
dataset artifact.

