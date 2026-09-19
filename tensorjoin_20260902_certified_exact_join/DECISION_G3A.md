# G3A host analytic-certificate opportunity decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_analytic_certificate_g3a`

## Decision

**PASS for implementing a separate GPU proof candidate. This is not yet a GPU
soundness proof or performance evidence.**

On the frozen G2A `4096x512` input, the conservative host analytic model
processed all 8,390,656 upper-triangle pairs with:

- zero reconstructed-expression bound violations;
- zero source-distance interval-containment violations;
- zero unsafe direct accepts and zero unsafe direct rejects;
- 90,991 direct accepts, 8,197,745 direct rejects, and 101,920 ambiguous pairs.

The analytic ambiguity is `0.9965x` the accepted G2A2 fixed-pad count of
102,277, well below the frozen maximum of 127,846. The larger analytic error
radius does not inflate ambiguity because the recomputed reconstruction norm is
tighter than the earlier empirical construction on this cell.

The largest observed float32 reconstructed-expression error was
`1.1288e-6`, only 1.4025% of its declared pairwise bound in the worst case; the
minimum observed bound slack was `3.8019e-6`. These are diagnostic tightness
measurements, not a replacement for the analytic derivation.

## Proven precondition opportunity

For D=512 and code range `[-127,127]`, the maximum possible absolute INT32
accumulator is 8,258,048. It is below `2^24` and `INT32_MAX`; the observed
maximum was 1,492,856. Thus this shape admits exact INT8 MMA accumulation and
exact int32-to-float32 conversion before scale arithmetic.

## Boundary

G3A executes the intended outward model on the host. It does not prove that
Triton emits the assumed operation/rounding sequence, does not include the
documented error allowance for the actual GPU square-root instruction, and
does not repair the FP32 refinement guard. Existing G2B results remain
empirically exact only.

The next admitted work is a separate G3B kernel that:

1. keeps the G2B keeper immutable;
2. incorporates explicit expression, square-root, addition, and residual-sum
   outward margins;
3. refines every unresolved G3B pair in direct FP64 initially, so FP32-stage
   soundness is not conflated with the INT8 certificate gate;
4. validates direct accept/reject containment, exact final IDs, generated
   PTX/SASS, memcheck, repeated runs, and sustained execution before timing.

## Evidence

- Gate and protocol: `GATE0_G3.md`, `PROTOCOL_G3A.md`.
- Result: `results/g3a_analytic_certificate.json` (SHA256
  `4f7b4f9ca366bd89bd9a38b55f71ea052a1e5bed757607a02a2f06e4feedf09e`).
- Raw log: `raw/g3a_analytic_certificate.log` (SHA256
  `feea7b3240492734f8c9e6bede27722c0d27bacfc36419e5eae82c7180621756`).
- Runner SHA256:
  `3bcff75294babd150387a2a3b7aa9e21c83101033e38c11a6563a3537e8f7455`.

