# Block-scaled E3M4 TensorJoin Stage 1 screen

Pre-execution registration, 2026-09-19. Keeper: repository commit
`2c3ab2875ecbf89fd9a21aeb34a8a6fc6fc0ca92`. No production dispatch is changed.

## Mechanism, novelty and decision

Use the independently verified SM120 QMMA.SF E3M4 format fields in a real,
nonuniform tiled Gram matrix, not the previous uniform-code instruction probe.
One UE8M0 scale per row/block of 32 elements may reduce reconstruction residuals
and hence the FP32 refinement queue. Extra scales, block reductions, FP64
metadata, packing and MMA operand preparation may erase any gain. Public
blackwell-isa/cubit already disclose the hidden selector; format discovery and
generic precision routing are not novelty claims. This is a cheap application
screen, not a paper thesis or external-baseline admission campaign.

Reject this implementation for the tested task if exact canonical IDs differ,
the interval envelope fails, a sanitizer fails, or the full-cost gate fails.
An implementation rejection is not a universal rejection of E3M4 or scaling.

## Frozen task and host

- Host class: eight-GPU RTX PRO 6000 Blackwell Server Edition server; selected
  physical GPU 0, SM120, driver 590.48.01, CUDA 13.1.115.
- PyTorch 2.11.0+cu130, Triton 3.6.0, compiler pinned to CUDA 13.1.
- Existing cooperative GPU guard, five idle preflight samples and continuous
  ownership checks. No clock/power/driver changes; never stop foreign work.
- Input: finite FP32 N x 512 rows, absolute values <= 1. Output: directed,
  sorted canonical pair IDs using the original finite-FP64 distance predicate
  and binary64 threshold. Data and threshold hashes come from the frozen G17
  fixture manifest; no raw source data is published.
- Correctness: the same 14 real/synthetic/boundary fixtures from the prior
  experiment, maximum N=4096. Small N<=32 also use independent CPU FP64.
- Timing: real CIFAR 4096 x 512, synthetic clustered and synthetic outlier
  1024 x 512. Use each fixture's original threshold, unchanged for all methods.
- Methods: per-row INT8, FP16, E4M3, E3M4; block32 E4M3 and E3M4. Re-measure all
  in every process using the same retained cascade. Direct FP64 is the oracle,
  not a promoted low-precision performance baseline. No cold dynamic selector.

## Numerical and implementation contract

- Retain the prior INT8/FP16/per-row producer, classifier, FP32 refinement and
  explicit-binary64 terminal without source changes. Subclass the existing
  engine only for block32 preparation and native scaled dot execution.
- Block scales use saturation-safe power-of-two ceil(max/max_finite), floored
  at 2^-100, with E4M3 max 448 and E3M4 max 30. Blocks below 2^-100 use unit scale and
  charge all lost information to the residual. E3M4 uses nearest-even rounding,
  including codeword midpoints and underflow.
- Q is row-major uint8 N x 512; S is row-major uint8 N x 16 UE8M0. B is Q^T;
  B scales stay in N x (K/32) layout, not transposed. Triton dot_scaled compiles
  the documented E4M3 QMMA.SF form; E3M4 patches private copies only at bits
  82 and 84. Every selected instruction must be SF/16832/F32/E4M3/E4M3/E8,
  with absolute format fields zero before the patch; all other bytes frozen.
- Native codebook and E4M3/E5M2 compiler controls are rechecked on the new kernel.
  Raw 0x10 distinguishes E3M4 from E4M3. Random codes, lane positions and scales
  validate nonuniform dot reconstruction; no FP16 emulation is admitted.
- Reconstruct z elementwise with each element's own block scale. Compute the
  SAME original-row norm interval, upper ||z|| and upper ||x-z|| using FP64
  reductions and outward FP32 conversion. The existing classifier formula is
  unchanged because it depends on reconstruction errors, not scale granularity.
- Retain the conditional FP32 dot envelope gamma=0.00012232370499987155 for
  FP8/FP16. Check it against CPU FP64 decoded dots and actual distance interval
  containment. Empirical checks are not a universal hardware/real-arithmetic
  proof. Native scale application introduces a new path requiring these gates.

## Kernel design / readiness

One four-warp CTA per 32 x 32 output tile, contraction chunks of 64 elements,
two scale bytes per row/chunk, two compiler-managed pipeline stages. One CTA
owns each output tile; lanes retain FP32 accumulators for the full K=512 loop.
Input and scale loads -> synchronous scaled MMA -> accumulator update -> next
chunk -> output store. Compiler-managed shared staging/reuse must be reflected
in generated SASS and resource metadata; there is no custom barrier protocol.
Per-row preparation uses a 16 x 32 logical tile: block max/quantization ->
reconstructed FP64 sums -> row metadata. Expected extra work versus per-row:
16 maxima/scales rather than one, 16 bytes of scales rather than one float,
native scale operand loads/shuffles. Dot useful arithmetic is unchanged.
No register cap, setmaxnreg, cluster, TMEM, or new dispatch heuristic is added.

## Validation before timing

1. Exhaustive FP8 raw-code axes; anti-fallback; documented A/B compiler controls.
2. Nonuniform N=31/32/33/97/129 Gram dots with mixed scales and K=64/512;
   independent decode, norm/residual bounds, full distance intervals, midpoints.
3. Exact IDs on all 14 fixtures and threshold-adjacent CPU checks.
4. Memcheck/racecheck/initcheck/synccheck on CIFAR prefix512 and boundary32.
5. Full-cascade pointer churn: 16 calls per method per fixture, N capped at512;
   explicitly count distinct input addresses. Fixed-input dot Graph replay and
   non-default-stream checks are separate from full-cascade stress.
6. Static compiled binary/resource ledger, including native SF assertions and
   every observed Triton specialization. No profiler duration is a speed claim.

## Measurement and admission

Four fresh processes, method orders forward/reverse/forward/reverse. Two warmups
and five retained observations per method/workload. Exclude compilation and
evidence serialization; include H2D, allocations, producer/scales/metadata,
Stage 1, queue transfers, FP32/FP64 refinement, D2H and CPU canonicalization.
Use synchronized host wall time for the complete CPU/GPU denominator; CUDA
events retain phase attribution, not interchangeable end-to-end estimates.
Additionally, on CIFAR4096, run 32 consecutive complete calls per method per
process with the same alternating order. Retain all individual samples plus
the enclosing batch wall time. This is a bounded sustained screen, not uptime.

Primary estimator: paired process log ratio comparator/candidate with 95%
Student-t interval (df=3); report p10/p50/p90, marginal medians, order splits and
process wins. A screen win requires >1.10x in every process versus BOTH INT8
and FP16, interval lower bound >1, and no regression in the sustained screen.
E3M4-vs-E4M3 at block32 separately attributes the format change. No favorable
subset replaces the predeclared real-data decision. Keep all negative samples.

These are matched research prototypes, NOT a re-benchmark of optimized legacy
INT8/cuBLAS FP16 or a full 60K join. Even a screen win would remain a candidate
pending strongest external-baseline and production qualification.

## Operational record and rollback

New source/artifact directory only; original TensorJoin and prior experiments
remain untouched. Each label is unique, with guard PID, numeric GPU telemetry,
source/cubin hashes and raw results. Stop only owned process groups on failure,
foreign occupancy or timeout; no GPU reset. Existing private GitHub branch
precision-format-routing receives curated task-owned code/evidence after review.

## Pre-timing operational amendment

GPU 0 completed probe_v1 and validate_v1. Before the first sanitizer, the guard
observed a foreign compute process on GPU 0 and refused to launch our child.
That blocked attempt is retained; no foreign work was stopped. Re-preflight
found GPU 1 idle. Move the final campaign to physical GPU 1 and repeat probe,
all-fixture validation, every sanitizer and stress before timing. All four
performance processes and sustained batches must use GPU 1. GPU 0 diagnostics
are not mixed into performance or final admission gates. Host CPU activity is
not isolated; this remains a shared-host end-to-end screen, not a pure GPU
throughput measurement. The original frozen datasets, thresholds, method set,
timing estimator and acceptance threshold are unchanged.

Before timing, a read-only review found that the diagnostic DUMP classifier
was missing from the compiled-specialization registry. Register it explicitly
and repeat all final correctness/sanitizer/stress gates as v3. This changes
evidence collection only, not any kernel or cascade decision. v1/v2 diagnostics
are retained separately; final admission uses v3 gates and the matching-source
benchmarks. The scale-floor wording above describes the unchanged producer.

After v2 finished, a new foreign training process occupied GPU 1. No task-owned
process was running and no foreign process was touched. The final v3 gates
and all four benchmark processes move together to idle physical GPU 7 under
the same guard. This supersedes GPU 1 as the final timing device; GPU 0/1
results remain diagnostics only. The source fingerprint is frozen after the
registry fix and before v3 starts.
