# Conditional solution-state FM for TensorJoin, 2026-09-08

The user requested empirical validation of treating an unsolved data-derived
state as the starting point and progressively obtaining the join solution.
This round tests scalar tile-occupancy solution states, followed by a certified
completion mechanism. It does not claim to directly generate every pair ID or
to guarantee exact neural endpoints. Freeze this protocol, all solver sources
and preparation artifacts before training/model selection.

## Shared task and split

Unchanged Cifar 60000 x 512 FP32 input, six archived threshold-sweep queries,
64-row blocks in the exact previous FP64 PCA sort, and original conservative
PCA32 AABB pruning. Bitwise check the prior permutation and lower bounds.
Retain the original arithmetic/refinement, integer ID remap and CUB complete
output implementation. No learned coordinate or prediction changes the final
original-distance predicate. Inherited numerical scope remains explicit.

Split whole blocks, not pairs sharing rows: shuffle blocks 0..936 with seed
2026090901, use first 625 blocks for train (40000 rows), next 156 for validation
(9984), and remaining blocks plus the ragged final block for test (10016).
Train only on pairs whose BOTH blocks belong to train; validation similarly.
PCA is transductive. This dataset and references have appeared in prior rounds;
this is an adaptive within-dataset follow-up, not fresh independent testing.
Before selection aggregate exact reference counts only for train/validation
block pairs. Test/full labels and full certificate outcomes are accessed after
selection. Existing FP64 oracle generation is not free: report its archived
wall time separately from this round's label aggregation and training costs.

## Data-conditioned solution model

For each tile/threshold the endpoint y is -1 if its exact reference count is
zero and +1 otherwise. Twenty query-independent FP32 input summaries describe
PCA center differences, variance sums and bbox gaps in bands [0:1,1:4,4:8,
8:16,16:32], original center distance, mean original pair distance, radius sum,
norm-range gap and original PCA lower bound. Normalize summaries using only
training block pairs. Append normalized threshold for 21 condition features.
These summaries are untrusted guides, not a sufficient statistic theorem.

The coarse squared-distance heuristic is original mean pair distance minus
2*sqrt(2*sum((var_a+var_b)^2)+4*sum((mu_a-mu_b)^2*(var_a+var_b))) in PCA32.
Define initial state a=tanh((T-coarse)/(0.25*T)), computed from input data.
No Gaussian source, no extra path noise and no minibatch OT matching are used.
For t~Uniform(0,1), z_t=(1-t)*a+t*y, train v(c,z_t,t) to y-a with MSE.
At deployment actually integrate dz/dt=v(c,z,t) from a to get occupancy score.

FM has a 21-to-64 affine condition layer, learned 64-vector state and time
weights, SiLU, 64-to-64, SiLU and scalar head: 5761 parameters. Matched direct
residual predictor a+r(c,a) has 5697 parameters, the same structure without a
time vector, and learns y on the SAME training examples by MSE. Both output
heads start at zero; hidden/state weights match and FM time weights start at
zero. All parameters train. Cache only the first affine condition term during
ODE inference because it is independent of state/time, verified against the
uncached nonzero field. Report dense FLOPs using this actual implementation.

Three seeds 2026090911/12/13, 3000 Adam updates per family/seed, lr=.001,
batch=1024, grad clipping=10, foreach=False. Thresholds cycle equally; sample
naturally distributed train pairs surviving the coarse bound, excluding diagonal
tiles. Alternate paired model update order; shared batch preparation is charged
fully to each standalone family. Checkpoints at 750/1500/3000; retain all 18
weight files and all training histories. FM Heun steps 2/4/8 (NFE 4/8/16),
direct NFE 1. Evaluate 27 FM and nine direct validation candidates.

Select each family's trained checkpoint by mean extra certified pair pruning
across six validation queries when checking 25% of coarse survivors. A stronger
certificate is computed on validation block pairs only for this selection.
Ties favor fewer ODE steps, fewer updates, lower seed. No full query timings,
test labels or full certificate outcomes select models. Freeze weights/hashes
and solver steps before admission. Initial zero models are preflight controls,
not eligible fallback selections that could silently replace the actual FM.

## Certified completion and controls

Predictions allocate extra checking effort only. Rank coarse surviving,
off-diagonal tiles by predicted occupancy, lowest first. Verify a fixed fraction
with actual 64-by-64 pair distances in the original PCA32 projection. Direct
FP32 differences/FMA sums use a conservative global numerical allowance and
the inherited projection operator bound; details in CERTIFICATE_SCOPE.md.
Check dimensions in groups of eight, stopping at 8/16/24 if all pairs already
clear the conservative threshold, otherwise using 32. A tile is removed only
when this certificate passes; all other tiles go to the inherited solver.
Models cannot reject tiles or accept output IDs on confidence alone.

Eight configurations: pca (coarse bound only), all (check every survivor),
heuristic25 (coarse heuristic ranking), random25 (fixed seed 2026090920+cell),
direct25, fm05, fm10, fm25. Fractions .05/.10/.25 are of coarse survivors;
ceil to integer. Deterministic partial selection resolves equal scores by tile
ID; execute selected certificates in ascending ID order for every policy.
FM25 versus PCA/heuristic25/direct25 are primary comparisons; all, random25
and FM05/FM10 diagnose the certificate, routing and effort-budget tradeoffs.
Keep all outcomes. Do not present oracle empty-tile counts as executable routing.

## Validation and complete timing

Before training, test Heun on analytic fields, zero-model identity and cached
condition equivalence. Check actual and synthetic GPU projected minima against
CPU FP64 direct sums, original-space distances, near-threshold decisions,
ragged blocks and partial/full certificate agreement. Preserve development
failures and source versions (preflight a0 unsupported break syntax; a1 omitted
valid 24-coordinate early stop from the test enumeration; both before training).

After selection, evaluate raw initial/direct/FM candidate predictions on
validation, test and full data: report false negatives, false positives and
missed reference-entry counts. These raw predictions are diagnostic only.
Save actual selected FM trajectories on 512 fixed full-data tile inputs;
compare 2/4/8/16-step endpoints and reverse integration without reselection.
Check all 48 configuration/threshold masks against full exact occupancy.
Perform 96 normal full-array admission calls, 12 FM25 diagnostic calls and
five actual rebuild-plus-query admissions: 113 complete array comparisons.
Freeze all timing dependencies and admitted masks.

Three fresh guarded confirmation processes. Each configuration/threshold/F8
or F16 gets one warmup and four retained repeats in rotated/reversed order,
no outlier deletion: 288 warmup and 1152 retained calls. Each process also
executes 30 actual rebuild-plus-query calls (pca/all/heuristic25/direct25/fm25,
k4/original/k1024, F8/F16): 90 calls. Total complete calls = 1643 including
admission. Every output count and SHA256 must match the reference, and every
executed mask must equal the admitted mask. Complete query time includes
condition construction, real model/ODE inference, selection, certificate,
input/metadata transfers, arithmetic/refinement, remap, complete CUB sort and
CPU download. Query-independent CPU summaries/index may be reused; no cached
query predictions or certificate results are used by the timed solver.

Each process measures direct and FM2/4/8 inference on all eligible tiles of the
original threshold, seven retained repetitions after warmup. Input host features
and initial states are prepared; include normalization, H2D, all real ODE work,
and endpoint D2H. Chunk size 65536. No CPU-neural throughput claim. Fresh index
builds include PCA/features/bounds, input reorder and only the summaries required
by the policy; label generation/training are accounted separately.

Point estimates average process medians. Use 20000 paired hierarchical bootstrap
draws (process, then repetitions), seed 2026090929, for each cell and the equal
six-query mixture. Useful primary improvement requires >=5% query-time saving
at the lower 95% confidence bound and positive direction in all three processes.
Only three process clusters; intervals describe this campaign's repeatability.
Report all budgets, cost components, and training/index/oracle amortization.
Compare current GPU4 runs only, not old GPU timing ratios.

## Resources and archive

New directory gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_fm_solver.
GPU4 UUID GPU-863c06a5-9f33-0265-b098-013fa840d5db, same exclusive guard,
30-second idle admission, <4096 MiB use, <=1800 seconds per guarded process,
four CPU/PyTorch/BLAS threads, TF32 disabled, deterministic PyTorch operations.
Never modify old experiments, clocks or foreign processes. Archive successful
and failed development checks, frozen inputs, models, compiler captures, raw
runs, complete output audits, CPU/GPU environment, statistics, plots and full
file SHA256 inventory. Method source: METHODS.md.
