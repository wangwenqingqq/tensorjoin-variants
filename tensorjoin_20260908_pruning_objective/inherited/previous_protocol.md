# One-pass learned-layout feasibility, 2026-09-08

Test the immediate proposal in the conversation: can a small one-pass neural
map produce more useful layouts than PCA, with affordable model/build costs?
This round does not implement or claim an advantage for flow matching. It is
a prerequisite/comparator for deciding whether multi-step FM deserves testing.

Freeze before model training or geometry evaluation:

- Primary data and six thresholds: unchanged Cifar60K, 60000 x 512 binary32,
  all full FP64 references from the completed threshold sweep.
- Reuse the four-thread FP64 PCA32 construction and conservative projected
  AABB bound from the previous block screen. Other bounds contributed no
  Cifar pruning and are removed for EVERY layout. Verify the PCA masks against
  the archived previous combined masks, plus complete output correctness.
- PCA uses all unlabelled input coordinates, as a shared transductive index
  preprocessing step. Fixed row permutation seed 2026090813: 40000 gradient
  training rows, 10000 validation rows, 10000 untouched model-selection test
  rows. This is within-dataset holdout, not external-distribution validation.
- Common network inputs are the existing 32 PCA coordinates, normalized with
  training-row mean/std. Outputs are one scalar used only for stable sorting;
  every 64 rows form a tile (last tile ragged). Original coordinates and IDs
  remain the inputs to all reliable distance predicates.
- Controls: ordinary first-PC sort; learned linear residual (33 trainable
  parameters); MLP residual 32 -> 64 -> 64 -> 1, ReLU, 6337 parameters. Both
  learned models start with exactly the first-PC score by zeroing the residual
  output weights. Sort transformations alone cannot authorize pruning.
- Fixed proxy objective: normalized distance stress on 50% local and 50%
  uniformly drawn training-row pairs. Targets are Euclidean distances in raw
  PCA32 coordinates, divided by training std of PC1. Loss is
  mean((abs(score_i-score_j)-distance)^2/(distance+0.1)), with 1e-4 times
  mean squared residual-score regularization. This tests one defined proxy;
  failure is not proof that all neural layout objectives fail.
- Construct local pairs from exact top-8 neighbors in the FP32 GUIDE distance
  computation, excluding self: fixed 8192 training anchors against all 40000
  training rows, tiled GPU computation. This graph is only training data,
  never a candidate generator or correctness predicate. Count graph cost.
- Three seeds 2026090814/15/16 for each learned architecture. Adam lr=0.001,
  batch=1024 pairs, 1200 updates, checkpoints at 300/600/1200, gradient norm
  clip 10. No tuning after results. Select one trained checkpoint per family
  by highest six-threshold mean certified pair-pruning fraction on validation
  rows, ties toward fewer updates and lower seed. Retain all histories and
  candidates, whether or not they beat PCA. Test/full geometry is evaluated
  only after selected model hashes are frozen.

The admission and deployable GPU execution use the same frozen projected
AABB bound for every layout, no Lipschitz claims about a neural network. Test
and complete-data reference occupancy must show no pruned true matches.

Measure GPU host-to-host scalar mapping separately (includes feature H2D and
score D2H, model already loaded), and CPU mapping for context. The deployed
mapping backend is fixed to GPU before timing. Report parameter count, actual
mapping time, batch arithmetic FLOP estimates, shared preprocessing, guide
graph, training time, and experiment/model-selection cost separately.

Complete query configurations: original/full, PCA/geometric, linear/geometric,
MLP/geometric, and MLP/full. Both retained F8 and F16 paths, identical CUB full
output and integer original-ID remap. Repeated-query times reuse CPU layout
and bound matrix; still include mask/list construction, fresh GPU input and
metadata, scans/refinement, remap, complete sort and CPU download.

Admit all 60 configurations against full reference arrays and retained kernel
identities. Freeze timing dependencies and selected models. Three fresh timing
processes, each one warmup and four retained repetitions per configuration,
balanced/reversed model/method/cell order, no timing-based deletion. Per
process also measure 18 complete build-plus-query calls (pretrained PCA/linear/
MLP x k4/original/k1024 x F8/F16); these include rebuilding PCA/features,
mapping, sorting/reordering and conservative bounds. Training is separately
accounted, not mislabeled as included in those calls.

Compare learned layouts primarily with PCA using the SAME bound/backend,
not only original full scan. Report cellwise and equal-six-query-mixture
time savings with 20000 paired hierarchical bootstrap draws (process + paired
repetitions, seed 2026090817). A >=5% query-time reduction vs PCA whose 95%
lower bound exceeds 5%, with same direction in all processes and correctness,
supports only the tested regime. Additional build/training break-even must
be reported. Geometry improvement without net time improvement does not pass.

Resources: new @TENSORJOIN_ROOT@/tensorjoin_20260908_learned_layout. GPU0
UUID GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1; same exclusive guard and locks,
30-second idle, <4 GiB device use, <=1800 seconds per guarded process, four CPU
BLAS threads. No foreign process termination, clock changes, prior-file edits,
or comparisons obtained by dividing new times by old GPU2/GPU0 times.
