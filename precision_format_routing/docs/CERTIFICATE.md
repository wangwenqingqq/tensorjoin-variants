# Numerical contract and implementation boundaries

Let the finite FP32 source rows be x and y, their stored-format reconstructions
(including scales) be z and w, and outward-rounded metadata bounds satisfy
||x-z|| <= ex, ||y-w|| <= ey, ||z|| <= uz, ||w|| <= uw.
The metadata stores intervals for the ORIGINAL squared norms ||x||², ||y||²,
not just reconstructed norms. FP64 reductions plus conservative inflation and
directed FP64->FP32 conversion construct these metadata.

For native scaled dot p, assume the conditional accumulation envelope
|p - z·w| <= gamma * uz * uw + tiny. Then

    |p - x·y| <= gamma*uz*uw + ex*uw + uz*ey + ex*ey + tiny.

The classifier evaluates original_norm_x + original_norm_y - 2*p using
explicit directed FP32 additions/multiplications and outward padding. It accepts
only an upper interval endpoint <= T, rejects only a lower endpoint > T, and
otherwise emits an INT64 pair ID for the retained FP32 distance filter. That
filter emits its unresolved IDs to the retained-arithmetic FP64 terminal.

- INT8's D512 int32 dot cannot overflow (512*127² = 8,258,048) and is also exactly
  representable in FP32. Its certificate charges gamma=2^-21 for scaled-dot
  rounding. FP8/FP16/FP32 use the prior FP16 control's conservative conditional
  gamma=0.00012232370499987155, not an asserted universal vendor guarantee.
- Per-row power-of-two scaling is saturation-safe for E3M4 max-finite 30,
  E4M3FN 448, E5M2 57344. The producer includes ties-even and underflow tests.
- E3M4 decode: sign * mantissa/64 for exponent zero, otherwise
  sign * (16+mantissa)*2^(exponent-7); exponent 7/mantissa 15 is NaN.
- Native E3M4 is experimentally validated on this SM120 binary family. It is
  not an official CUDA dtype or a cross-architecture guarantee. The standalone
  byte quantizer is real GPU work and is included in preparation latency.
- The native patch flips only bits 82 and 84 of each freshly compiled plain
  E4M3 QMMA.16832 instruction in an owned binary copy. A/B documented compiler
  controls, exhaustive codebook execution, anti-fallback values and high-entropy
  D512 dots qualify the format identity. The disassembler does not print a
  recognized E3M4 mnemonic; a zero printed QMMA count is not a fallback proof.
- Raw 0x10 decodes to E4M3FN 0.03125, not 0.125, versus E3M4 0.25. At K64,
  self-dots are respectively 0.0625 and 4.0. This independent oracle corrects
  an inconsistent earlier internal scalar lookup; no inference uses that typo.

## Threshold ABI

All terminal thresholds are explicit binary64 constants. Because Stage 1/2
interval endpoints are binary32, comparison against floor_FP32(T) is exactly
equivalent to comparison against binary64 T for those endpoints. The terminal
comparison must NOT use a runtime Python float inferred as FP32 by Triton.
Three adversarial threshold fixtures cover the exact one-coordinate squared
distance and its immediate binary64 neighbors.

## What the checks establish

Exact-ID equality is measured against a direct FP64 predicate on the frozen
inputs; small fixtures also use independent CPU subtraction/squared-distance
checks. Metadata and interval containment are directly checked on high-entropy,
signed, zero, underflow and endpoint data. These empirical checks do not turn a
conditional dot bound into a formal guarantee for every possible input or
hardware implementation. The result is a research prototype, not a certified
production exact-real join.

Format advantages are distribution-dependent. INT8 has uniform absolute spacing
under its row max scale; E3M4 trades exponent range for mantissa bits. Outliers
can hurt INT8, while broad residual accumulation and FP8 relative precision can
hurt E3M4. Fewer FP32 candidates need not reduce the final FP64 boundary queue.
