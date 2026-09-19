# Protocol G3A: Host Analytic-Certificate Opportunity Gate

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_analytic_certificate_g3a`

Status: frozen before measurement.

## Scope

- Input: unchanged G2A `4096x512` float32 vectors and canonical FP64 oracle.
- Predicate: unchanged strict G2A squared-distance threshold.
- Quantization: unchanged per-vector symmetric signed INT8 codes and float32
  scales.
- Reference ambiguity: 102,277 upper-triangle pairs from accepted G2A2.
- This is a host analytic model and cheap opportunity test. It is neither a GPU
  implementation nor performance evidence.

## Analytic model

Let `q_i`, `s_i`, and `e_i` be one vector's INT8 codes, float32 scale interpreted
as an exact real number, and an upward reconstruction-error norm. Integer code
norms and pair dots are exact because `D*127^2 < 2^24 < 2^31`.

For the float32 expression

```text
fl(A_i*s_i^2 + A_j*s_j^2 - 2*C_ij*s_i*s_j),
```

G3A applies an absolute outward radius of

```text
E_ij = 2^-17 * (|A_i*s_i^2| + |A_j*s_j^2| + 2*|C_ij*s_i*s_j|)
       + 8 * smallest_normal_float32.
```

`2^-17` is deliberately at least eight times the standard `gamma_16` bound for
round-to-nearest float32 at this operation count. G3A measures the slack against
a higher-precision reconstruction expression but does not infer GPU lowering
from the host result.

Each reconstruction error is recomputed from the actual emitted codes and
float32 scale in float64. Its squared-norm reduction is inflated by
`1/(1-gamma_(2D+2))`, square-rooted, and rounded upward before storage in the
model. The candidate distance interval is

```text
lower = max(sqrt(max(reconstructed_d2_fl - E_ij, 0)) - e_i - e_j, 0)
upper = sqrt(max(reconstructed_d2_fl + E_ij, 0)) + e_i + e_j.
```

GPU square-root and final-addition error are deferred to the next gate; G3A asks
whether this ideal outward model is selective enough to implement.

## Required outputs

- exact input/oracle/source hashes and environment;
- integer overflow and float32 exact-conversion preconditions;
- maximum observed reconstructed-expression error and minimum bound slack;
- interval containment diagnostics against direct float64 source distances;
- direct accept, direct reject, ambiguity, unsafe accept/reject counts;
- ambiguity ratio versus the frozen 102,277-pair G2A2 reference;
- first violating pair IDs and values if any.

## Acceptance

All conditions must hold:

1. source/oracle count and hash match the frozen G2A contract;
2. no INT32 overflow possibility and every integer dot is exactly convertible
   to float32;
3. zero reconstructed-expression bound violations;
4. zero source-distance interval-containment violations;
5. zero unsafe direct accepts and zero unsafe direct rejects;
6. analytic ambiguity is at most `floor(1.25*102277) = 127846` upper pairs.

Failure is retained and stops the analytic proof route. Passing admits only a
separate GPU proof candidate; it does not alter any existing claim.

