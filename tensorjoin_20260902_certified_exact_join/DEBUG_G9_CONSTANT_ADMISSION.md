# G9 A0 numerical admission failure

The first correctness process failed before any TC or performance measurement.
The signed boundary fixture uses T=nextafter(0.5625,-infinity). Its P16 FP32
stage passed definitive-decision checks; the newly written FP64 repair stage
then made a false definitive decision. This localizes the first failure to the
FP64 repair boundary, not quantization, TC multiplication, or data packing.

The emitted PTX compared against `0d3FE2000000000000` (0.5625) instead of the
requested previous binary64 value. The two directed radius multiplications
received `mov.b64 ..., 1065353216`, the FP32 bit pattern 0x3f800000 placed in a
64-bit operand, NOT the binary64 value 1.0. The inline-assembly constraint did
not perform a numeric FP32-to-FP64 conversion. Python floating constexpr values
were materialized at FP32 precision/width before their use in the FP64 operation.

Correction: create the threshold and both relative-radius coefficients with
explicit `tl.full((), value, tl.float64)` constants. Apply the same threshold
fix to the not-yet-run TC certificate. This restores the frozen numerical
contract; it does not relax correctness or change the performance candidate.

Evidence: `raw/g9_check_a0.log`, its failed isolation manifest, original source
in `artifacts/g9_failed_check_a0/`, original cubin/PTX in
`artifacts/g9_check_a0/`, and `raw/g9_a0_constant_lowering_excerpt.log`.
Closure requires fresh boundary/all-workload checks, static constant inspection,
and sanitizers on the corrected binary. No A0 timing exists to discard.

Closure: check A3 on GPU7 passed all fixtures and main workloads. The static
audit verifies exact binary64 threshold and coefficient patterns; dtype checks
reject non-FP64 inline-assembly operands at compile time. Memcheck/synccheck
passed in both eager and Graph modes with the identical corrected cubin set.
