# G18 numerical contract and outward-cutoff derivation

Pre-implementation design; theory is conditional on the stated arithmetic and
input domain. Target CUDA instructions, finite tests and full pair equality are
separate gates. No exact-real or arbitrary SciPy-build equivalence claim.

Let S_k be the exact sum of k squared differences of stored finite FP32 values,
|x| <= 1, D=512. Let s_k use FP32 RN subtraction and RN FMA in upstream dimension
order, with gradual underflow (`--ftz=false`, no fast math). No overflow occurs:
|difference| <= 2, S_D <= 2048. Let R be original-order sequential FP64 RN
subtraction, separate RN square and RN addition, exactly as the fallback code.

Set u32=2^-24, u64=2^-53, gamma_m(u)=m*u/(1-m*u),
g=gamma_(2D+4)(u32), h=gamma_(3D+4)(u64), B=D*2^-140.
These deliberately conservative constants depend on D/domain/arithmetic, not
on observed errors or a desired fallback rate.

## Bounding s_k

FP32 subtraction obeys relative error <=u32 in this domain. If its exact result
is subnormal, it is a multiple of 2^-149 within the subnormal range and therefore
exactly representable; subtraction does not require an extra underflow term.
Squaring the rounded difference contributes at most two relative-error factors.
The k nonnegative FMA accumulations contribute at most k further factors, plus
at most k*(2^-150)*(1+u32)^k absolute underflow error. For k<=512,
(1+u32)^k<2, so the latter is smaller than B by a wide margin. Standard product
bounds with k+2 factors are enclosed by the larger chosen gamma_(2D+4).
Consequently, for every prefix:

    (1-g)*S_k - B <= s_k <= (1+g)*S_k + B.

Zero/subnormal products are covered by B; we do not assume they survive an FMA.
FTZ is NOT allowed by this proof. Changing flags or architecture reopens the
arithmetic admission gate, even if ordinary inputs still agree.

## Bounding the declared FP64 reference

All nonzero exact differences between stored FP32 values have magnitude at
least 2^-149; their squares are at least 2^-298, far above FP64 underflow.
FP32-to-FP64 conversion is exact. RN subtraction, squaring and accumulation
obey relative-error bounds without overflow/underflow. The chosen h safely
encloses the <=3D+4 factor count, independent of dimension permutation:

    (1-h)*S_D <= R <= (1+h)*S_D.

This compares rounded scores through the exact S, not by pretending that FP64
is exact-real arithmetic. The fallback still executes the specified original
order; the bound alone does not make different FP64 reduction orders identical.

## Precomputed two-sided cuts

For a stored FP64 threshold T>=0, exact-rational values are:

    high_exact = B + (1+g)*T/(1-h)
    low_exact  = (1-g)*T/(1+h) - B.

Let high be high_exact rounded upward to FP32, and low be low_exact rounded
downward. Host code uses Fraction from the exact binary value of T, then checks
and adjusts the final FP32 candidate using exact rational comparisons. No chain
of ordinary floating-point operations is assumed to round outward.

If any prefix s_k > high, then S_D>=S_k>(T/(1-h)), so R>T and rejecting is safe.
If the final s_D<=low, then S_D<=T/(1+h), so R<=T and accepting is safe.
All other candidates, on either side of the native threshold, execute the
original-order FP64 terminal and compare directly with stored T. Negative low
cutoffs at tiny T correctly force fallback rather than discarding self pairs.
The native rounded epsilon is used only by the unchanged RT index, not as the
terminal threshold. Candidate completeness is independently tested and remains
an unproved general boundary.

## Self-review / falsification checklist

- Prefix safety uses a lower bound on full R; never accept from a short prefix.
- Use `>` for safe reject and `<=` for safe accept, preserving inclusive ties.
- CPU rounding checks include subnormal cutoffs, zero, adjacent threshold ulps,
  negative low cuts and values near the D*4 distance bound.
- Compile explicit `__fsub_rn`, `__fmaf_rn`, `__dsub_rn`, `__dmul_rn`, `__dadd_rn`;
  inspect selected code instead of assuming the compiler honors the source.
- Preserve original dimension inverse and strided candidate accesses in fallback.
- Native G17's two FN and two FP must all be fixed; no tolerance fitting or
  emitted-pair-only reranking. Independent strict FP64 and legacy SciPy output
  checks must both pass the frozen full fixture matrix.
- The chosen envelope need not be tight. A high fallback fraction is retained
  adverse evidence, not permission to shrink the bound after measurement.
