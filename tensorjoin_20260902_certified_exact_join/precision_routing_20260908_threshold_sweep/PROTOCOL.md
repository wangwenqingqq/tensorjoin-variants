# Threshold sweep, frozen before threshold selection and timing

Date: 2026-09-08. Input: original 60000 x 512 finite FP32 CIFAR vectors,
SHA256 95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c.
Original row order, upper triangular tile order, 4096 tiles per batch, existing
fused F8 and F16 paths, GPU2 UUID GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245.
No tuning of tile size, schedule, certificates, queue capacity, or kernels.

Select five thresholds targeting mean nonself degree 4, 16, 64, 256, 1024,
plus the original radius 161/256. Use 2^22 independently sampled nonself
ordered pairs with replacement, NumPy PCG64 seed 2026090801, canonicalize
pair orientation, and CPU FP64 direct squared differences. Choose m/4096,
m in [1,4095], minimizing absolute empirical degree error, ties toward
smaller m. Thus radius and radius squared are exactly representable FP32.
Threshold planning uses no latency results. Report realized full degrees.

Numerical contract: output agrees with the unchanged, SHA-verified retained
FP64 terminal cubin at each new squared threshold. This is not exact-real
membership or a universal FP16 certificate. Inherited F16 bounds remain
conditional on the stated Tensor Core error premise. The interval itself
does not depend on threshold; new F16 constexpr thresholds require individual
binary capture and instruction/resource audit before retained timing.

Build a complete independent reference: enumerate every unordered pair with
self once (1,800,030,000 pairs) and call the frozen FP64 terminal at the
largest threshold. Persist accepted upper IDs. For every smaller threshold,
call the same terminal on that superset; monotonicity makes all larger-threshold
rejects irrelevant. Mirror off-diagonal IDs and sort to canonical directed
uint64 output. Verify coverage, overflow, counts, unique IDs and hashes.
No fast filter is used to exclude pairs from this reference.

Every operator call must match its reference output count and SHA256, and
admission calls also compare arrays. Capture and freeze all binaries, input,
sources, thresholds, and reference evidence before retained timing.

Timing scope: warmed complete host-to-host operator, from before tile-list
construction, input upload and metadata through queue processing, counters,
synchronization, output download, mirroring and sorting. JIT, input disk I/O,
reference/hash verification and warmups are excluded. Both paths use the
same canonical output wrapper. No extra identity remapping from the prior
layout experiment is required because rows are never remapped. Report these
new timings internally; do not directly ratio them against the older wrapper.

Exploration: two independent processes, all six cells, one warmup per path
and three paired retained repetitions per cell. Confirmation is predetermined
for all six cells regardless of exploration: three new processes with six
paired repetitions per cell, followed by three fresh repeat processes with
six pairs per cell. Rotate/reverse threshold order and alternate F8/F16
within pairs. Two warmups per cell/path for confirmation. Fresh processes
have no per-input metadata cache. Separate diagnostic calls after admission
record stage CUDA events and host preparation/output time; diagnostic durations
are never substituted for complete-operator timing.

Ratio R = F16 seconds / F8 seconds; R > 1 favors F8. Material crossing requires
some cell with process-cluster bootstrap 95% CI entirely above 1.05, another
entirely below 1/1.05, consistent direction in every confirmation process and
both independent blocks. Also require >=5% oracle headroom over the best
static path on an explicitly uniform six-cell query mixture, with 95% CI.
Point estimates alone do not pass. Bootstrap 20000 draws, fixed seed, paired
process/cell sampling. Report process medians and both blocks separately.
Without material crossing, stop this two-path planner claim on this sweep.
Even a pass establishes only whole-query threshold routing headroom, not
per-tile routing or a novel planner.

Operational limits: existing exclusive locks and ownership monitoring,
GPU memory <4 GiB, <=1800 seconds per guarded process, 30 seconds idle
before each process, no foreign process termination or GPU clock changes.
If reference construction needs more than one process, persist row-range
shards and resume disjoint coverage; never skip verification to save time.
All failed attempts and protocol amendments remain visible.
