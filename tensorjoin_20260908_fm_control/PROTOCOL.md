# Actual OT-CFM layout control, 2026-09-08

The user explicitly requested a real FM comparison. Test one specified FM
layout strategy, independently of the unsuccessful previous neural layouts.
All configuration choices below freeze before learning or geometry selection.

## Data, features and reliable predicates

- Unchanged Cifar60000 x 512 binary32 input, six threshold-sweep cells and
  complete FP64 reference outputs. Same original arithmetic kernels, integer
  ID remap, CUB full output and conservative original PCA32 AABB bound.
- Shared FP64 PCA32 computed on all unlabelled rows, with original projection
  padding, norm factor and squared-distance slack. Verify the old PCA masks.
- Reuse split seed 2026090813: 40000 training, 10000 validation, 10000 test.
  This is an adaptive same-dataset follow-up: the test rows have been seen in
  previous reports, and PCA is transductive. Do not call it a new independent
  holdout. No test/full geometry or query timing selects this round's models.
- Model source u = (q - training_mean(q)) / training_std(q[:,0]), using one
  scalar scale to preserve relative Euclidean distances in the PCA subspace.
  Target g is a 32D isotropic Gaussian with variance equal to the mean of the
  32 training-source variances. Gaussianization is a chosen layout hypothesis,
  not a guarantee of better pruning or a direct pruning objective.
- Learned coordinates ONLY select a permutation. All certified decisions use
  the untransformed original PCA coordinates and frozen distance kernels.

## A genuine conditional flow-matching model

For each minibatch, sample 256 training rows u and 256 Gaussian targets g.
Build an FP32 squared-distance cost matrix and solve its balanced one-to-one
assignment with scipy.optimize.linear_sum_assignment on CPU. Use all 256
matched pairs. This is minibatch OT coupling, not a global OT solution or a
correctness predicate.

Draw t uniformly in [0,1] and form z_t=(1-t)u+t*g. Train the time-dependent
velocity v_theta(z_t,t) against g-u with mean squared error / target_variance.
This uses zero conditional-path noise (sigma=0). At deployment actually solve
dz/dt=v_theta(z,t), starting from each observed u at t=0 through t=1.

FM network: 33 -> 128 -> 128 -> 32, SiLU, 24992 parameters. One-pass control:
u + r_theta(u), 32 -> 128 -> 128 -> 32, SiLU, 24864 parameters, trained by MSE
to exactly the same matched Gaussian endpoints. ALL parameters are trained
for both models. Shared hidden weights/biases start identically; FM's extra
time-input column starts at zero; both output layers start at zero. These
initial controls are checked, but only genuinely trained checkpoints are
eligible for the FM/direct comparison. PCA is the separate fallback baseline.

Seeds 2026090831/32/33, 4000 Adam updates per model/seed, lr=0.001, batch=256,
no weight decay, gradient norm clip 10, checkpoints at 1000/2000/4000. Paired
models consume identical OT pairs at every step; separate seeded t draws are
used only by FM. Alternate execution order for cost measurement. Log shared
OT pairing cost and each model's update cost separately; a standalone model
must pay its full pairing cost, not half of it. Count all three seeds' cost.

## Layout and model selection

- Every method may use two predetermined 64-row grouping rules: stable sort
  by the first coordinate, or recursive balanced maximum-variance coordinate
  splitting. The latter uses all 32 layout coordinates. PCA gets both rules
  too; its first-coordinate sort retains the archived FP64 PCA score exactly.
- FM integrates with fixed-step Heun, testing 8/16/32 steps (16/32/64 actual
  vector-field evaluations). Evaluate each trained checkpoint and rule on
  validation certified mean pair-pruning across the same six thresholds.
- Direct has 3 seeds x 3 checkpoints x 2 rules = 18 candidates; FM has 54
  including the three solver budgets. This larger FM selection budget is
  explicit, not presented as equal search cost. Baseline PCA has two rules.
- Select by highest validation pair-pruning, then fewer updates, lower seed,
  fewer FM steps and first-coordinate rule. No query timing is used. Save
  all 18 trained weight files and all candidate evaluations, good or bad.
  Freeze selected model hashes, rule and FM solver budget before admission.
- Evaluate fixed validation regression losses and sliced-W2 diagnostics to
  the Gaussian target (64 fixed directions), plus layer weight changes and
  nonzero displacement. Distribution diagnostics do not select checkpoints.
- After selection, inspect FM endpoints at 8/16/32/64 steps on 1024 fixed
  full-data rows, reverse-integration error and trajectories. Report numerical
  sensitivity; do not quietly substitute a new model/solver after this check.

## Complete execution and cost comparison

Families pca/direct/fm, configurations original_full, pca_geo, direct_geo,
fm_geo, fm_full, each F8/F16 and all six thresholds. Check 36 test/full
reference-occupancy combinations, 60 normal full outputs, 12 FM diagnostics,
three true rebuild-plus-query calls; all admission outputs array-equal to
the archived reference. Verify inherited arithmetic/compiled identities and
freeze every timing dependency.

Three fresh guarded confirmation processes. Each configuration gets one
warmup and four retained repeats, balanced/reversed order, no outlier deletion:
720 retained and 180 warmup calls. Each process also runs 18 actual complete
rebuild-plus-query calls (three families x k4/original/k1024 x F8/F16). Count
all 1029 complete calls including admission; repeated outputs must match
reference count and SHA256. Rebuilds include fresh PCA/features, real model
mapping/ODE, grouping, bounds and original input reorder; weights are already
trained, and training is reported separately.

Each process times one-pass GPU mapping and FM GPU mapping at 8/16/32 steps,
seven retained repetitions after warmup. Host-to-host includes FP64-to-FP32
feature conversion, H2D, normalization, all actual ODE calls and 32D endpoint
D2H; weights are preloaded. Count NFE, total parameters and dense-layer FLOPs.
CPU preprocessing is in actual build timing; no CPU neural mapping claim.

Primary comparisons: FM versus PCA and versus matched direct mapping. Equal
six-query mixture and each cell use 20000 paired hierarchical bootstrap draws
over processes and repetitions (seed 2026090838). A useful regime requires a
>=5% reduction in complete query time at the lower 95% confidence bound, same
positive direction in all three processes and correct outputs. Report mapping,
training, actual build-plus-query and both selected-run/all-seed amortization.
Only three process clusters: intervals describe within-campaign repeatability.
Compare current GPU4 measurements only, never divide by prior GPU0/GPU2 times.

## Resources and implementation evidence

New directory: gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_fm_control.
GPU4 UUID GPU-863c06a5-9f33-0265-b098-013fa840d5db. Same exclusive guard checks,
30-second idle admission, <4 GiB device use, <=1800 seconds per guarded process,
four CPU BLAS/PyTorch threads, deterministic algorithms, TF32 disabled. Do not
terminate foreign tasks, modify clocks, or change prior experiment files.

Check Heun against analytic constant/linear fields, including error reduction
and reverse integration; check balanced grouping and OT assignment on small
known cases before training. Preserve all failure records if any check fails.

Method references (scope and exact formulas in METHODS.md):
- https://arxiv.org/abs/2210.02747
- https://arxiv.org/abs/2302.00482
