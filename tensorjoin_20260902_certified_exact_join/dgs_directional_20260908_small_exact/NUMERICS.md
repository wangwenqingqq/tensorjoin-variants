# Conservative directional-envelope arithmetic

This is a bounded CPU experiment, not a new general numerical certification
paper. Inputs are the original finite FP32 values widened to binary64; D=512,
abs(x)<=1. The output contract is the existing binary64 direct-difference
predicate, not mathematical-real exactness.

Let u=2^-53 and gamma(k)=ku/(1-ku). All arrays below use binary64, without BLAS
matrix multiply or reduced-precision math. The unnormalized Sylvester
Walsh-Hadamard matrix H has H^T H=512 I, so for any coordinate subset S,
sum_{j in S} ((Hx)_j-(Hy)_j)^2 /512 <= ||x-y||^2. Raw coordinate subsets have
the same property with scale1. These two bounds may be combined by max, not sum.

For a computed transform yhat=fl(Hx), the depth is9 addition/subtraction levels.
An absolute transform error envelope is gamma(9)*sum(abs(x)), enlarged for the
FP64 sum, subnormal losses, and final rounding. screen.py uses
nextUp(gamma(9)*sum64(abs(x))/(1-gamma(512)) + 2^-970).
There is no transform overflow for this bounded domain. Each coordinate interval
is [nextDown(yhat-e), nextUp(yhat+e)]. Raw intervals use e=0. The min/max over a
block encloses every member's exact transformed coordinate, including errors.

For block intervals A,B, the exact coordinate gap is
max(0, lo_A-hi_B, lo_B-hi_A). The implementation nextDowns the rounded
subtractions before clamping at0, nextDowns each squared gap, and divides each
prefix cumulative sum by (1+gamma(512)), followed by nextDown. The resulting
L is a lower bound on every member-pair squared real distance (up to subnormal
absolute losses far below the separately reserved oracle allowance).
Division by512 is a power-of-two scaling; nextDown also covers its rounding.

To preserve the frozen binary64 oracle, real-distance pruning alone is
insufficient at threshold ties. The original oracle loads FP32 as FP64,
subtracts, squares and reduces512 nonnegative terms (two256 reductions plus
addition). Even allowing4096 binary64 roundings, the absolute error is bounded
by 2048*gamma(4096) plus subnormal allowance, below 2^-28. This is deliberately
loose. Prune only when L > nextUp(T+2^-28), T=25921/65536. Thus even a downward
oracle roundoff cannot turn a pruned pair into an accepted result.

The complete-ID audit additionally verifies no accepted original pair is
removed after mapping old IDs through each permutation. This is a full
finite-workload check, not an independent mathematical proof of the admitted
upstream operator. Synthetic separable/boundary/high-entropy fixtures and
random actual member-pair comparisons prevent an all-zero filter from being
mistaken for substantive numerical coverage. An independent Fraction replay
checks selected transform coordinates exactly.
