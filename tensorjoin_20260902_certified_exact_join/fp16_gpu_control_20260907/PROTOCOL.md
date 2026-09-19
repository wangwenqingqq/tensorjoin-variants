# Owned FP16 dense GPU control: admission before timing

Experiment tensorjoin_20260907_fp16_gpu_control_d512. Frozen before code changes.
This is comparator admission/attribution, not a new research thesis. Prior
FaSTED/filter/quantization overlap and the FP16 census remain in force.

## Contract and implementation choice

Original gpu-host-8 GPU2/SM120/driver590.48.01, torch2.11cu130 and existing
cuBLAS containers; N60000,D512,T25921/65536, finite FP32 |x|<=1. Same unchanged
FP32 stage2 and selected FP64 terminal as the retained INT8 path. Reference
equivalence, not exact-real arithmetic. Original files and two sealed deliveries
are protected; create this independent directory without modifying old sources.

C: fuse explicit FP16 reconstruction (subnormals set to zero), residual and
norm metadata in one GPU pass. Store outward-rounded FP32 norm endpoints and
upper norms; evaluate the dot/reconstruction/terminal interval with explicit
directed FP32 operations on GPU. No FP64-per-pair classifier handicap. The
conditional dot-model coefficient is the previous g=(1026*2^-23)/(1-1026*2^-23),
rounded upward to FP32, not fitted after a failure. Retain1e-12 absolute padding
and the2^-38 original-input norm terminal allowance. Native FP32 directed
rounding controls classification; a selected-library dot model remains a
separate assumption, not established merely by a TC opcode.

Owned cuBLAS handle, explicit default stream0,8,519,680-byte device workspace,
math16/compute68, FP16 inputs/FP32 output, alpha1,beta0,default algorithm-1.
The owned lifecycle is adapted from admitted G19 source. Alignment/layout,
runtime-selected names, resources, source export and library hashes must be
recorded for512-square and all3 production panel shapes before comparisons.
No borrowed Torch BLAS handle is created by this new control.

## Gates and stop rules

1. Trace owned FP16 GemmEx at512-square and4096x4096/4096x2656/2656x2656,
   D512. Prewarm outside an NVTX capture range. NCU LaunchStats/SourceCounters
   and selected SASS exports are diagnostic only. Check accumulated FP32
   paths/no unexpected reduced-output conversion. Record any acquisition gap.
2. Validate the GPU metadata against exact rational row values and GPU interval
   enclosure against an independently evaluated CPU expression. Validate all
   selected pairs against the fixed terminal, including rejected candidates.
   Reuse the frozen512-row adversarial fixture and three public panels; also
   include fixed-seed wide-exponent/cancellation dot checks. Keep actual score
   samples for exact dot-bound checks, rather than checking output counts only.
3. Audit actual compiled metadata/classifier PTX/SASS and runtime ABI. No FP64
   pair operations, no local spills, and no unexpected .ftz on interval ops.
   Fail-closed on invalid input, capacity/identity errors or unsafe decisions.
4. Memcheck/synccheck/initcheck/racecheck on bounded composite probes; stress
   includes32 alternating fixture/input allocations and immutable-input Graph
   replay of the classification/refinement sequence with calibrated launch
   counts (not a general dynamic Graph API). Full60K canonical output equality
   is a separate gate. Explicit shutdown releases owned handle/workspace/cache.
5. Only after gates1-4 pass may a separate timing freeze be written. Remeasure
   A and C in that campaign under the same warmed complete host-to-host scope;
   do not divide by the old1.518533x denominator or profiler durations.

No speed estimator is selected here; no timed ranking is authorized before
admission. A first failure stops its stage and is retained. Code mistakes may
receive an explicitly versioned repair, never a silent failed->pass rewrite.
Missing proof/selected-code gates do not become passes through finite tests.

## Resources and shared-host safety

Two historical GPU2 locks,30s quiescence,4GiB normal admission memory limit,
30-minute per-process timeout; no clock/power modification. Profiler overhead
may require a separately declared host-memory/resource exception, not a silent
guard relaxation. Initial live users occupy GPU3-6 and GPU7; do not touch them.
No deployment, manuscript or Overleaf changes. No novel-mechanism claim follows
from comparator engineering or conventional outward-rounded arithmetic.
