# G16 pre-screen addendum: equal GPU preprocessing opportunity

Declared before any G16 full-operator compatibility or public timing. The seven
GPU quantization metadata fixtures have passed; no G16 public times are known.
Do not modify the original protocol or four already-frozen sources.

A fair strongest control must not retain a CPU norm-preparation cost while the
candidate receives GPU preprocessing. Add a separate GPU-norm version of the
immutable G15 FP32-first operator. Preserve the original FP32 control as an
attribution arm. Freeze six independent processes: block 0 GPU-G5, CPU-norm
FP32, GPU-norm FP32; block 1 GPU-norm FP32, CPU-norm FP32, GPU-G5. This preserves
the original pair order, adds a stronger control, and permits no replacements.
Require >=1.25x against the faster admitted control in **both** blocks for a
narrow engineering screen pass. No formal, sustained or novelty promotion.

## GPU norm design card

Same host, GPU, D512 finite |x|<=1, contiguous FP32 source, output/threshold,
stream and complete pageable-host denominator as the original G15. Keep the
same direct pedantic cuBLAS binding, classifier, terminal, panels, capacities,
compaction/readback and sorting. Only replace norm-metadata construction and
its host-to-device transfer. Charge source H2D, metadata allocation/launch,
invalid-flag read, and all downstream work. Warm the new specialization outside
timing, retain no source-prepared state, and check no timed JIT.

One CTA/vector, four warps and 512 logical coordinates. Convert x to FP64;
FP32 square products are exactly representable in FP64. Reduce their nonnegative
squares in FP64 with fusion disabled. Use the original conservative
radius gamma=(512*eps64)/(1-512*eps64) times center plus 1024*tiny64.
Form nextafter(sqrt_rn(center+radius)*(1+8*eps64), +infinity) in FP64. Radius
exceeds the standard 511-addition error bound, including radius evaluation.
All constants must be FP64. A flag rejects nonfinite/out-of-domain input or
invalid metadata; do not modify the already frozen classifier radius.

Lanes own x/square until the CTA reduction; only row center, radius and L2 upper
survive after reduction and are stored to three FP64 arrays. No custom shared
handoff, barrier, persistent queue, TMA or async stage. Triton owns reduction
synchronization. Audit generated resources, FP64 operations and local traffic.
Memory/launch dominated preparation is a hypothesis, not a measured claim.

## Additional gates

- Independently enclose norms for all 60K vectors and the same seven metadata
  fixtures, including subnormals; compare against CPU reduction. Record any
  changes, not just output agreement.
- Seven complete reference-oracle join fixtures, rectangular cuBLAS mapping and
  interval containment; full60K canonical count/hash; no timed JIT.
- 1000 alternating-input/pointer-churn subset full calls, full memcheck and
  fixture synccheck. No new Graph or arbitrary-stream support claim.
- Actual runtime-selected cuBLAS trace plus fixed unchanged classifier/terminal
  source and binary identity; new metadata static audit. Profiler times excluded.
- GPU-G5 additionally retains original protocol gates: full output and original
  stage binary/work identity, metadata 1000-call stress, memcheck/synccheck.
