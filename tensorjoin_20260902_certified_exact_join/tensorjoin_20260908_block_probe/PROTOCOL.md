# Block structure and bound-cost probe

Frozen before measurements, 2026-09-08. This is a bounded mechanism screen,
not a new join implementation or a novelty claim. Existing runs are read-only.

Use the full 60,000 x 512 FP32 CIFAR input and all six previously frozen
thresholds and exhaustive frozen-terminal FP64 reference result sets.
Input SHA256: 95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c.

Compare original order, a fixed PCG64 random permutation (seed 2026090814),
a stable sort by distance to one pivot, and a balanced maximum-variance
coordinate tree with leaves of at most 32 vectors. Test consecutive block
sizes 32, 64, 128; 64 is the existing arithmetic kernel's native block size.
Smaller blocks are structural diagnostics, not immediately skippable native
tiles. All preprocessing uses input only, never result sets or thresholds.

Choose 16 pivots by deterministic farthest-first traversal, starting at a
fixed PCG64 draw (seed 2026090815); compute all point-pivot distances in FP64.
For each block, store coordinate minima/maxima, mean and enclosing radius,
and min/max distances to each pivot. Evaluate three familiar bound families:

- Coordinate box: sum squared coordinate gaps for a lower bound; sum the
  squared largest possible coordinate differences for an upper bound.
- Enclosing sphere: max(0, distance between centers - radii sum)^2 and
  (distance between centers + radii sum)^2.
- Pivot intervals: maximum separation of intervals as a lower distance bound;
  minimum over pivots of the sum of maximum distances as an upper bound.
- Combined: maximum of the three lower bounds and minimum upper bound.

Use FP64 arithmetic and expand squared-distance bounds outward by 2^-20.
This fixed conservative pad is a screening implementation choice, not a new
numerical theorem. Admission requires checking every direct reject/accept
against the full frozen FP64 result sets. Claims remain empirical for this
fixed input; do not call the bounds universally certified. Zero-result blocks
reject iff lower > T, full-result blocks accept iff upper <= T.

For each ordering/block size/threshold, census the exact number of empty,
fully accepted, and mixed blocks from the full reference. This answer-informed
oracle is only a structural ceiling, never a deployable pruning method.
Separately count blocks/pairs resolved by each computable bound. Include ragged
last blocks and upper-triangle/self accounting. Check complete accepted-pair
counts and zero false decisions in every configuration, without sampling.

Record wall times for input-only ordering, pivot preprocessing, block metadata,
each CPU bound family, and reference census separately. Reference/hash work is
excluded from implementation cost. These CPU prototype times are diagnostics,
not GPU timings or end-to-end join speedups. All methods/results are retained.

Stage gate: proceed to a guarded GPU bound-cost microprobe only if some native
64 configuration resolves at least 20% of blocks using the combined bound.
Otherwise stop this fixed set of bound mechanisms, report the structural
gap, and do not optimize kernels with negligible removable work. The threshold
is an internal screening rule, not a publication criterion. No tuning layouts,
pivot count, block sizes, numerical pad or gate on results.

If a GPU phase is admitted, freeze an amendment before its execution, use
GPU2 UUID GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245 and the existing unmodified
30-second-idle / <4-GiB / 1800-second exclusive guard. No foreign process kills.
CPU phase uses four BLAS/OpenMP threads and has a 1800-second process timeout.

Validation includes separated-cluster positive controls, ragged groups,
duplicate points and threshold equality. Preserve source/input/reference
hashes, all raw logs, per-configuration counts, and a final report. Prior E0
pivot-tree failure on audio/video is context, not fresh evidence from this run.
