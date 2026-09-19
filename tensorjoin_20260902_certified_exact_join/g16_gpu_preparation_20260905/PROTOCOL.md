# G16: remove the measured host-preparation bottleneck, not the strong control

Declared 2026-09-05 before implementation/measurement. This is a new bounded
data-path candidate under the original static-join problem, not a novelty pass,
new dataset, or a return to a weak baseline. G15's pedantic FP32-first control
and every unfavorable observation remain mandatory and unchanged.

## Admission and hypothesis

G14 measured about 94% of G5 time in preparation/movement/canonicalization.
G15-A's byte-preserving CPU blocking gives only a 1.27–1.33x preparation
improvement. G15-B full compatibility already measures about 0.31 s with the
same complete output; its final paired screen must finish before G16 GPU work.
This creates an explicit attribution question: can transferring the original
FP32 source once and constructing quantized operands/error metadata on the GPU
remove enough *measured* preparation cost to survive the strong control?

This is engineering evidence, not a claim that moving quantization to a GPU
is a new research principle. Do not promote a subsumed thesis by packaging this
change. No formal/multi-dataset expansion is admitted here.

## Design card

- Host/GPU/runtime and locks: same verified 8p GPU 2 and environment as G15.
  GPU 1's foreign job is untouched. No clock/power changes.
- Public source/output/threshold: identical frozen CIFAR-GIST FP32 60000x512,
  epsilon 0.62890625 and D2 threshold 0.3955230712890625; all 3,926,078 sorted
  directed IDs, including self, with the original `13cae87e...5963495` hash.
- Input domain for new preparation: finite FP32 coordinates with abs(x)<=1,
  D=512; other domains fail closed. No new exact-real claim.
- One CTA per vector, 512 lanes of logical data over 4 warps. Each lane owns
  its loaded FP32 coordinate and resulting INT8 code. Max, exact INT32 code
  squared norm, and FP64 residual squared norm are CTA reductions.
- Compute scales with round-to-nearest FP32 division; use libdevice RNE rint.
  INT32 sum is exact: 512*127^2=8,258,048 <2^24 and <2^31. Reconstruct the norm
  in FP64 before FP32 conversion. Form actual reconstruction/residual in FP64.
- Reuse the existing analytic residual formula with gamma=(2D+2)u64/(1-(2D+2)u64),
  RNE FP64 division/sqrt, nextafter toward +infinity, upward FP32 conversion and
  positive-subnormal floor. Explicit FP64 constants/intrinsics are required.
- A per-call device error flag rejects invalid inputs or nonfinite metadata
  before downstream routing. Charge its initialization and scalar read.
- Use a charged ordinary PyTorch transpose-contiguous copy for the INT8
  column-major operand; do not introduce an unvalidated custom transpose.
- Keep all three G5 routing/refinement kernels, launch arguments, capacities,
  tile schedule and host output canonicalization unchanged.

### Ownership / readiness / live state

| Phase | State / ownership | Release boundary |
|---|---|---|
| H2D | Original FP32 input, one contiguous allocation | Retained through all G5 refinement |
| Per-vector max/code | lane x, scalar row max/scale, integer q | Max dies after scale; q stored and consumed in norm/residual |
| Metadata | exact INT32 norm sum, FP64 reconstruction/delta/sum | q/delta dead after reductions; row leader stores metadata |
| Operand transpose | PyTorch stream-ordered copy | Row-major and transposed INT8 operands retained through scan |
| Routing | Unchanged G5 stage owners | Original count-read/overwrite graph |
| Output | Unchanged full host reconstruction/sort | Timer ends only on sorted host IDs |

No custom asynchronous barrier, cluster, TMA, TMEM or persistent work queue.
Triton owns reduction synchronization. Audit registers/shared/local traffic and
generated numerical instructions. An allocation/launch or pointer change does
not waive sanitizer/stress gates.

## Gates and fixed observations

1. Metadata tests: all 60K real vectors plus deterministic signed random,
   zero/one-hot/duplicate, cancellation, tiny-normal/subnormal and mixed-exponent
   fixtures. Check finite metadata, exact INT8 ranges, bitwise scale/code/norm
   agreement where it occurs, and independent CPU residual-bound enclosure.
   Norms must agree with the independently reconstructed GPU-code/scale norm.
   Any real bound or norm violation stops the formula; no post-data widening.
2. Full same-output G5 integration, selected original G5 cubin/PTX identity,
   new metadata code audit, memcheck/synccheck and 1000 alternating-input/pointer
   metadata invocations. Check output counts and new metadata binary stability.
3. Two reversed-order independent-process blocks, GPU-prepared G5 then G15
   FP32-first, followed by FP32-first then GPU-prepared G5. No replacements.
   Charge all source transfer, metadata preparation, transpose, allocations,
   GPU stages, count reads, ID copies and host sorting in both variants.
4. Keep a narrow engineering candidate only if it wins both complete-process
   comparisons by >=1.25x and all preceding gates pass. Two blocks are not
   formal confidence/tail/sustained or generality evidence. Otherwise retain
   the loss and do not restore a weaker comparator.

Warm exact specializations outside timing; retain no prepared source state.
Do not use sanitizer/profiler timing in performance ratios. Keep original G5,
G15 CPU-blocked G5, and the FP32-first sources immutable. Record every attempt,
including failed launches and uncertain metadata behavior.

Any different quantization caused by subnormal execution is not automatically
a predicate failure: the stored reconstruction and conservative residual must
still describe the actual original FP32 vector. Record differences explicitly;
do not inherit old input-byte identity evidence for those cases. The full-input
certificate/output and code/safety gates remain mandatory.
