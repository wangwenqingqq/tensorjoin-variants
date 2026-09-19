# Multiscale distribution of unresolved pairs

Frozen 2026-09-08 before measurements. This is an input-only projection screen
and complete structural census, not a Kakeya theorem application or a new join
implementation. All prior experiments are read-only.

## Question

Does conservative low-dimensional filtering leave a small set of difficult
pairs concentrated in rows, columns or nested subtiles? Separate loose bounds
from coarse execution granularity using exhaustive frozen FP64 answer sets as
an offline diagnostic only.

## Frozen configurations

- Full CIFAR 60,000 x 512 represented FP32 input; SHA256
  95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c.
- All six inherited thresholds (k4, k16, k64, original, k256, k1024), unchanged.
- Orders: original, inherited PCG64 random, inherited maximum-variance kd tree,
  and stable sort by first PCA score. Orders depend only on input.
- Common nested PCA projections of ranks 16, 32, 64, obtained from the FP64
  centered covariance of the full input. No labels or threshold tuning.
- Nested consecutive blocks of 16, 32, 64, 256 points in each order; tails and
  the upper triangle including self are counted exactly. Native kernel size
  is 64. Finer blocks are a structural diagnostic.

## Bound and numeric scope

For represented projection P use scale2 = 1 + ||P^T P - I||_F + 2^-20 as an
upper operator-norm-squared allowance. Each represented FP64 projected
coordinate has enclosure +/-2^-24, inherited for D <= 1024, |x|,|P| <= 1.
For each rank compute conversion allowance
4*2^-24*sum(span(q) + 2^-24). Compute squared projected distances with FP64
Gram products in three cumulative coordinate bands; subtract the conversion
allowance, gamma_(4r+32)*8*r*max(abs(q))^2, and another 2^-20 absolute margin.
Divide by scale2, clamp below at zero, and reject only if lower > T+2^-20.
Use the rank-64 allowances and scale2 for every prefix, and clamp each computed
band contribution below at zero, so ranks share the same conservative test.
The Gram allowance covers band norms, dot products, and accumulation; this is
a conservative source-level engineering argument, not a formal rounding proof
for arbitrary BLAS. No TF32, FP16 or GPU arithmetic is used.

Every rejected pair must be absent from each exhaustive frozen-terminal
reference. Validate equivalently that every reference-positive pair survives
at every rank/layout/threshold, in the actual streamed masks. Independently
check FP64 direct projected distances and explicit reference lookups for fixed
sample blocks, tails, diagonals and threshold equality controls. Numerical
claims are limited to this input/terminal scope.

## Measurements

For every configuration save exact counts: surviving pairs, false candidates
(survivors minus true hits), empty/occupied blocks, weighted pair capacity of
occupied blocks, and oracle occupancy from true hits. At size 64 save the number
of active rows and columns. Report the distribution among occupied off-diagonal
full tiles, including fractions with <=6/16/64 surviving pairs or <=8/16 active
rows/columns, and distributions restricted to oracle-empty tiles.

Measure 256->64 and 64->16 child occupancy, both raw active-child fractions and
survivor mass in the most populated child. Compare geometric layouts with
original/random, and true-hit structure with conservative-survivor structure.
Record global pair rejection to distinguish a weak bound from scheduling loss.
Define work amplification = pair capacity of occupied tiles / surviving pairs;
this is a count-based diagnostic, not runtime. Projecting, reference census,
mask processing, saving, and ordering CPU times are separated where practical.

## Interpretation fixed before results

- Information-limited: >50% of all pairs still survive even rank 64.
- Sparse exceptions: among occupied full off-diagonal native tiles, >=20% have
  <=64 surviving pairs (<=1.5625%), OR >=20% can be covered by <=8 active rows
  or <=8 active columns. Report both criteria, without tuning cutoffs.
- Subtile concentration opportunity: going from occupied 64 tiles to their
  occupied 16 children reduces evaluated pair capacity by >=20%. For 256->64
  apply the same criterion. Report raw counts and conditionals, since sparsity
  alone can create empty children even under random layout.
- These are internal descriptive screens. Passing any screen does not prove
  novelty, cross-dataset generality or net speedup. No GPU phase in this task;
  a concrete certificate/executor design and cost experiment would be a later
  step, especially because a finer tile can lose Tensor Core reuse.

CPU only, four BLAS/OpenMP threads, one campaign with a 1800-second timeout.
Do not touch running GPU jobs or existing experiment directories. Preserve
sources, hashes, logs, all configurations, independently checked diagnostics,
and a report with clear limits. This dataset is previously explored, not a fresh
holdout; six thresholds are not six independent datasets.
