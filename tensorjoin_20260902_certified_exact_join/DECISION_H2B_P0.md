# Decision: H2B-P0 Pedantic-SGEMM Certificate Opportunity

Date: 2026-09-03

## Decision

**PASS P0 and admit the H2B GPU proof implementation.**  Even after expanding
the exact token distance by a pessimistic two-radius envelope that charges both
the permitted SGEMM midpoint displacement and the certificate radius around
that midpoint, all four frozen cells remain far below the predeclared
ambiguity limits.

This is a CPU-only geometry result.  It does not show that cuBLAS obeys the
proposed bound on the target binary, that the GPU implementation is correct or
safe, or that it is faster than H1.

## Results

| Dataset | Dimension | Target/query | Ambiguous objects | Fraction | Gate |
|---|---:|---:|---:|---:|---:|
| ESC-50/PANNs | 2048 | 1 | 6 / 65,536 | 0.00916% | <=5% |
| ESC-50/PANNs | 2048 | 8 | 19 / 65,536 | 0.0290% | <=2% |
| UCF101/R3D-18 | 512 | 1 | 0 / 65,536 | 0% | <=5% |
| UCF101/R3D-18 | 512 | 8 | 8 / 65,536 | 0.0122% | <=2% |

The dimension-aware `gamma_(2D+2)` values are
`0.0002443195117330004` for D2048 and `6.115810562469786e-05` for D512.
The maximum pessimistic token-envelope half widths are respectively
`0.0009772781076613465` and `0.0002446324392289422`.  All token/object
containment and direct-decision checks are zero by construction and explicit
validation.

## Interpretation

P0 rejects the hypothesis that a conservative pedantic-FP32 dot bound must
necessarily make the multi-vector certificate unselective on these workloads.
The observed maximum ambiguity is 69x below the stricter 2% target-8 gate.
This supplies headroom for actual cuBLAS midpoint error and implementation
guards, but is not evidence about latency.

## Next gate

Implement H2B-P1 with direct `cublasGemmEx`, queried
`CUBLAS_PEDANTIC_MATH`, explicit `CUBLAS_COMPUTE_32F_PEDANTIC`, exact-FP64
norm preprocessing, and the frozen object certificate.  Require actual-data
and adversarial containment plus exact final IDs before SASS, sanitizer, or
timing promotion.

## Evidence

- Result: `results/h2b_p0_pedantic_opportunity.json`, SHA-256
  `7b514ab72359320dd51384d9a791e3467be6f02417932ce0125fe0e0128c21e4`.
- Raw log: `raw/h2b_p0_pedantic_opportunity.log`, SHA-256
  `7b514ab72359320dd51384d9a791e3467be6f02417932ce0125fe0e0128c21e4`.
- Runner: `src/run_h2b_p0_pedantic_opportunity.py`, SHA-256
  `57c7e9c7733c16768baa0759df7b9ad72522206502c92facba0b12cf93e73676`.
- No GPU work was executed by P0.
