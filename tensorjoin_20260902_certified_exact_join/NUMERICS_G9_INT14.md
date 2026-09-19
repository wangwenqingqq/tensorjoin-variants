# G9 INT14 certificate derivation

This is a checked finite-workload contract, not an all-exponent library API.

## Exact integer reconstruction

For each original FP32 vector x, choose a power-of-two s with
q=round_even(x/s), |q_k|<=16319. Decompose q=128h+l with signed INT8
h=floor((q+64)/128), l=q-128h. Explicit range checks precede GPU execution.
Each of four D=512 signed INT8 dot products is accumulated in INT32:
hh<=512*127^2, |hl|,|lh|<=512*127*64, ll<=512*64^2 in absolute value.
All are far below INT32 overflow, regardless of accumulation order. Combining
in INT64 gives q_i dot q_j exactly, magnitude at most 512*16319^2<2^38.
The entire reconstructed vector z=s*q is exactly representable in FP32/FP64
on the admitted exponent range. Quantized norm2 s^2*sum(q^2) and the cross dot
s_i*s_j*dot_q are exact FP64 values: their integer significands have <53 bits
and power-of-two scaling only changes exponents. Overflow/subnormal cases are
rejected by host admission.

## Residual norm upper bounds without empirical calibration

Check every source/reconstructed coordinate is on the 2^-40 grid and scaled
coordinates are exactly convertible to INT64. Then d_k=(x_k-z_k)*2^40 is an
integer. Check D*max|d_k|^2 fits INT64 BEFORE multiplying or summing. The exact
integer R2=sum(d_k^2) represents ||x-z||^2=R2*2^-80.

An FP64 square-root estimate is only a starting value. Convert the proposed
bound r to its exact integer ratio a/b and verify
`a^2 * 2^80 >= R2 * b^2`. Advance r by nextafter(+infinity) until this exact
integer inequality holds. The stored r is therefore an upper bound regardless
of initial sqrt rounding. Obtain upper norm z_bound similarly by comparing its
rational square to the exactly represented reconstructed norm2.

## Pair certificate

Let A=||z_i||^2, B=||z_j||^2 and C=z_i dot z_j; these are exact FP64 inputs.
Using explicit PTX directed arithmetic:

```
Lz = max(0, sub.rm(add.rm(A,B), 2*C))
Uz = max(0, sub.rp(add.rp(A,B), 2*C))
e  = add.rp(r_i,r_j)
h  = add.rp(z_bound_i,z_bound_j)
E  = add.rp(mul.rp(2*h,e), mul.rp(e,e))
L  = max(0, sub.rm(Lz,E))
U  = add.rp(Uz,E)
```

Multiplication by2 is exact in the admitted normal range. Triangle inequality
gives ||z_i-z_j||<=h and ||(x_i-z_i)-(x_j-z_j)||<=e. Expanding the squared
distance gives |Dtrue-Dz|<=2*h*e+e^2<=E. Thus [L,U] contains the original
squared Euclidean distance. Accept only U<=T; reject only L>T; otherwise repair.
All directed operations are explicit inline PTX, without approximate sqrt or
unspecified floating tensor accumulation. Inspect generated code and test
enclosure against a CPU outward reference before timing.

## Repair

FP32 repair retains the conservative G8 direct-difference certificate on the
same admitted D=512 FP32 input, with fusion disabled. Its uncertain subset
goes to FP64 direct differences. The fixed 2^-40 grid and |x|<=1 checks ensure
each FP64 subtraction is exact; products and reductions may round. A relative
radius 2^-40 around the positive FP64 distance sum dominates the D=512 standard
binary64 accumulation/product error bound (<2^-43 here). Use directed mul.rm/
mul.rp for the final endpoints. Zero distance is exactly zero. No subnormal or
overflow arithmetic occurs within the admitted input range.

If this final interval still overlaps T, leave state2 (unresolved) and fail
the main-workload admission. Do NOT silently equate approximate FP64 with an
exact real predicate. Correctness-only adversarial fixtures may intentionally
produce unresolved states at equality; they must never produce a false
definitive decision. The actual main workloads must finish with zero state2.

Numerical references: documented integer MMA and directed arithmetic in
https://docs.nvidia.com/cuda/parallel-thread-execution/ . Integer decomposition
is an established technique, not the claimed novelty of G9.
