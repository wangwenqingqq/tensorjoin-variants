# Direct certified block-pruning objective, 2026-09-08

Test the user's approved next step: optimize the actual number of safely
pruned point pairs after sorting and blocking, and compare complete query
performance with PCA and the preceding distance-stress MLP. This is a bounded
one-pass layout experiment, not flow matching or an ODE solver.

## Frozen data and interpretation

- Same 60000 x 512 Cifar binary32 inputs, original six thresholds and complete
  FP64 reference outputs from the frozen threshold sweep.
- Same transductive FP64 PCA32 preprocessing, 64-row blocks, conservative
  projected AABB bound, original arithmetic kernels, ID remap and CUB output.
  Verify shared feature arrays, row split and PCA masks against the previous
  archives. No modification of any previous experiment directory.
- Reuse 40000 training / 10000 validation / 10000 test rows (seed 2026090813).
  Only training-row coordinates and fixed thresholds drive parameter search;
  validation rows select checkpoints. Freeze selected weights before new
  test/full geometry and all query timings. The test split was already seen
  in the preceding experiment, so this is an adaptive same-dataset follow-up,
  not a fresh independent holdout or an external-distribution claim.

## Model and direct objective

- Keep the 32 -> 64 -> 64 -> 1 ReLU residual MLP (6337 total parameters),
  standardized with training mean/std and initialized to PCA by a zero output
  layer. Fix random hidden layers; search only the 64 final-layer weights.
  Output bias stays zero because constant shifts do not affect sorting.
  This tests a random-feature readout search, not optimization of all weights;
  comparison with the old proxy also changes the optimizer/search capacity.
- Three initialization/order seeds: 2026090821, 2026090822, 2026090823.
- For EVERY candidate, map all 40000 training rows, stable-sort, make 64-row
  blocks, construct the same conservative FP64 AABB lower bounds as deployment,
  and count exactly how many unordered pairs (including self pairs in the
  denominator) are rejected across all six thresholds. Maximize the integer
  sum of those six counts. No nearest-neighbor graph or distance-stress loss.
- Five coordinate-search sweeps with target residual-score standard deviations
  [0.03, 0.01, 0.003, 0.001, 0.0003]. In each sweep visit all 64 output weights
  in a seed-dependent random order. Try both signs, with coefficient step
  equal to target deviation / max(training hidden-feature std, 1e-6).
  Choose among current, plus and minus by highest certified pair count, then
  smallest output-weight squared norm, then original order. This makes the
  training reward nondecreasing; no assumption that validation improves.
- Exactly 640 non-incumbent proposals per seed, 1920 total. Cache the fixed
  hidden activations for training only; candidate final-layer scores must
  equal ordinary complete model forward scores at every checkpoint. Training
  uses GPU mapping plus CPU certified bounds; record wall time including both.
- Save initial zero model and the end of every sweep (18 checkpoints total).
  Select one direct model by six-threshold mean certified validation pruning,
  ties toward fewer proposals then lower seed. The zero/PCA initialization is
  an eligible fallback. Retain all candidates, including rejected proposals.
- Prior proxy comparator is the previous frozen selected MLP checkpoint
  (seed 2026090815, update 300, SHA256
  5019e10e06c2904211dae28faac0c74ba665dd956fe791bce9a7c06e0b20c7eb).
  Do not retrain or reselect it. Its training cost is historical and separate.

## Admission and query measurement

- Families: pca, proxy, direct. Configurations: original_full, pca_geo,
  proxy_geo, direct_geo, direct_full. Both retained F8 and F16 paths, all six
  thresholds, same bounds and complete GPU output pipeline.
- Before timing: test/full reference occupancy checks (36 combinations),
  all 60 query configurations compared against full reference arrays, 12
  diagnostic calls and three actual rebuild-plus-query calls. Verify the
  zero-residual control. Freeze code, models, geometry and inherited kernels.
- Three fresh guarded processes. Each configuration gets one warmup and four
  retained repeats in balanced/reversed order: 720 retained and 180 warmups.
  No timing-based outlier deletion. Each process also runs 18 actual complete
  rebuild-plus-query calls (three families x k4/original/k1024 x F8/F16), and
  seven retained host-to-host model mappings per family/device after warmup.
- Repeated queries reuse CPU layout/bound matrices but include threshold
  mask/tile-list creation, fresh GPU inputs/metadata, all kernels/refinement,
  original-ID restoration, CUB full sorting and full output CPU download.
  Rebuild calls actually repeat PCA/features, mapping, sorting/reordering and
  bounds, with pretrained weights. Training is separately accounted.
- Primary comparison is direct versus PCA. Also report proxy versus PCA and
  direct versus proxy from this round only. Report each threshold and the
  equal-six-query mixture, using 20000 paired hierarchical bootstrap samples
  of process and repetitions, seed 2026090824. A useful gain requires >=5%
  lower query time versus PCA at the lower 95% confidence bound, same positive
  direction in all processes and correct outputs. With only three processes,
  intervals describe within-campaign repeatability only.
- Report total search budget, selected-checkpoint cost, model mapping, actual
  build cost and training/build amortization. Tiny/noisy time differences and
  an identical PCA fallback cannot be called a learned speedup. Record full
  permutation equality, block assignment changes and executed tile counts.

## Resources and preservation

New directory: gpu-host-8:@TENSORJOIN_ROOT@/tensorjoin_20260908_pruning_objective.
GPU4 UUID GPU-863c06a5-9f33-0265-b098-013fa840d5db. Same exclusive guard checks,
30-second idle admission, <4 GiB total device use, <=1800 seconds per guarded
process, four CPU BLAS threads. No foreign-process termination or clock changes.
Retain all attempts and raw records. Compare only this round's GPU timings.

Pre-training hardware amendment: GPU0 became occupied by a foreign job before
any training/measurement began. GPU4 was idle and is selected for this whole
round. Only add its explicit parser target/UUID; all lock, ownership, idle,
memory and timeout checks remain intact. This choice is based on availability,
not measured performance. CPU preprocessing already completed is unchanged.
