# Block-pruning feasibility, 2026-09-08

Question: do simple layouts plus conservative geometric bounds remove enough
64 x 64 distance tiles to improve the existing complete join? This is a
feasibility screen for learned layouts, not an FM implementation or novelty claim.

Primary input: unchanged Cifar60K binary32 N=60000, D=512, input SHA256 and all
six thresholds/reference arrays from the completed threshold sweep. Secondary
geometry-only screens: the frozen 4096-row SIFT128 and Fashion784 datasets at
their existing k1/k16/k64 thresholds, scaled by exactly 2^-8 (threshold squared
by 2^-16) to meet |x|<=1. A shuffled 4096 x 512 separated-cluster fixture is a
positive control; it is never pooled with real-data performance.

Before viewing geometry results, fix four layouts: original, ascending first
principal-component score, recursively balanced maximum-variance-axis split,
and recursively balanced two-means split (four Lloyd-style balanced updates,
deterministic farthest-point initialization). Every leaf is exactly 64 rows
except the final ragged leaf. Preserve original row IDs and original coordinates.

Bounds: center/radius balls, original-coordinate AABBs, distances to 16
deterministically chosen farthest-point pivots, and AABBs in a 32-dimensional
PCA projection with a verified operator-norm allowance. Use the maximum of
the four conservative squared lower bounds. All metadata and lower-bound
matrices are built on CPU with four BLAS threads. Include this cost in build
time. No tuning after inspecting results; keep all layouts, bounds and cells.

Use complete reference outputs to mark occupied tiles for every layout. This
gives an ideal empty-tile count, not an implementable candidate generator.
Assert that every reference-occupied tile survives each conservative bound.
Only certified geometric masks may enter deployable-path measurements.

Select one reordered layout before GPU timing by highest arithmetic mean
pair-weighted certified pruning over the six primary cells; ties favor
pca_sort, balanced_axis, balanced_2means, in that order. If all reordered
layouts prune zero pairs, select balanced_2means for the negative control.
This selection does not use ideal occupancy or GPU timing.

GPU comparison: original/full, selected-layout/full (isolates permutation
cost), selected-layout/geometric. All use unchanged F8 and F16 arithmetic
kernels and the same admitted CUB complete-output implementation from the
shared output audit. A new integer-only remap restores canonical original
upper IDs before common mirroring/sorting; include it in query timing.
Outputs remain complete directed uint64 IDs sorted in CPU RAM, with self once.

Repeated-query timing starts with reusable CPU layout/bound metadata and
reordered CPU input, but includes threshold mask/list construction, H2D input,
fresh GPU metadata, scan/refinement, remap, output sort and D2H. Build time is
reported separately, and amortized results explicitly modeled as B/Q + Tquery.
Also measure true single-query complete execution including rebuilding the
selected layout/bounds at k4/original/k1024 for both methods.

Admission checks all 36 normal configurations against complete FP64 reference
arrays; check the integer remap separately and retained arithmetic kernel
identities. Three fresh timing processes with one warmup and four paired
retained repetitions per configuration, balanced/reversed orders. Each process
also makes the six complete single-query calls. No timing-based outlier removal.
Report process medians, paired bootstrap intervals (20000 draws, fixed seed),
and all per-process direction checks. A >=5% repeated-query gain with interval
lower bound above 5%, consistent process direction and complete correctness is
only grounds for a larger study, not evidence that FM is required. Report
single-query regressions and amortization break-even explicitly.

Separately time selected-layout/ideal-occupied-mask as an optimistic diagnostic
for k4/original/k1024. Reference lookup/construction is excluded and this is
explicitly NOT a deployable algorithm or strict universal speedup upper bound.

Resource scope: new directory under @TENSORJOIN_ROOT@; do not change prior
experiments. GPU0 UUID GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1, own exclusive
guard with the existing lock paths, 30 seconds idle before each process,
<4 GiB device use and <=1800 seconds per guarded process. Same-GPU internal
comparisons only. Do not terminate foreign processes or alter GPU clocks.
Preserve failures and document amendments before fresh admission.
