# G8 A0: refinement layout diagnostic

Frozen before implementation or measurement, 2026-09-04.
Experiment: `tensorjoin_20260904_g8_refinement_layout_g2a4096`.

## Decision and novelty gate

Hypothesis: for an invariant uncertain-pair set, physical locality changes the
best refinement granularity, independently of ambiguity count. This is a cheap
component falsification test, NOT a new paper claim or a replacement keeper.

Known prior art already covers adaptive precision, GPU uncertain-work
compaction, and sparse-to-dense blocking. Therefore none of those alone is a
novel thesis. Relevant primary sources checked on 2026-09-04:

- Fast Parallel Evaluation of Exact Geometric Predicates on GPUs (2022):
  https://www.sciencedirect.com/science/article/pii/S0010448522000616
- GPredicates (2019): https://ieeexplore.ieee.org/document/8692354
- TC-GNN: https://arxiv.org/abs/2112.02052
- FP64 Tensor Core Euclidean distances: https://arxiv.org/abs/2209.11287

Novelty remains UNKNOWN. Only a numerical-uncertainty-specific, reproducible
granularity tradeoff could justify a further novelty audit. A generic batching
win or an offline layout win is insufficient. This is not E0 tree pruning, an
H2B multi-vector revival, or an explanation for G5 variance.

## Frozen input and quality

- Existing G2A CIFAR-GIST FP32 vectors, N=4096, D=512, C-contiguous.
- Input NPY SHA256: e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462.
- Actual G3B-R1 uncertainty set: 102079 uint64 IDs, `i*4096+j`.
- Raw sorted ID SHA256: 6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42.
- Squared threshold 0.5686872086178483, outward FP32 bounds.
- Same direct-difference FP32 interval expression as G3C-B-R1, two K=256
  reductions, fusion disabled consistently in all three diagnostic kernels.
- Output: one aligned accept/reject/uncertain byte per requested logical pair.
  Non-requested tile cells must not produce output. CPU FP64 oracle checks all
  definitive decisions. Uncertain decisions remain unresolved at this scope.
- Require identical three-way states across methods and permutations. If a
  reduction-layout change alters a state, retain evidence and stop this A0
  comparison rather than silently relaxing its contract.
- GPU FP64 repair, compaction, final pair export, first certificate pass, row
  permutation and planning costs are excluded from CUDA-event component time.
  Host planning time is separately reported, never hidden as free online work.

## Workloads, fixed before observing their occupancy

Actual set, same original logical IDs in every case:
1. identity row order;
2. seeded permutation (seed 20260904);
3. descending uncertainty-graph degree, stable original-ID ties (offline
   uncertainty-aware opportunity probe, not an online algorithm);
4. stable ascending first principal-component projection of centered vectors
   (offline data-only locality probe; no parameter search).

Positive control: the first 102079 upper-triangular pairs of a 453-vertex
clique, using the same original source vectors. Compare identity placement and
the same seeded row permutation. These pairs are SYNTHETIC requested work,
not real quantization uncertainty. Their logical set stays fixed across the
two placements. A positive control does not satisfy the real-data gate.

## Methods and shape

- P1: one requested pair per CTA, G3C arithmetic with aligned byte output.
- P16: sixteen requested pairs per CTA, same arithmetic, tile-sorted queue;
  this is the batching/locality control, not just a weak unsorted baseline.
- T4x4: one active physical 4x4 pair tile per CTA; load four left and four
  right vectors, broadcast differences, compute all 16 cells, store only the
  requested cells through an explicit slot map. Charge all padded work.
- P1 and P16 receive the same tile-sorted requested-pair queue as T4x4.
- Mapped endpoints are normalized to the physical upper triangle, preserving
  the undirected logical pair and its original-ID output mapping.
- Exactly one tile shape; no tuning menu after results. This initial test
  isolates direct-FP32 locality/reuse, NOT Tensor Core block performance.
  A negative here cannot exclude a future certified-TC mechanism.

## Design card

- Target SM120 RTX PRO 6000 Blackwell, physical GPU1 on gpu-host-8.
- Native geometry: ordinary FP32 arithmetic/reductions, no MMA/TMA/cluster.
- Four warps per CTA, no inter-CTA communication or atomics, one stage.
- P1: 256 temporary elements; P16/T4x4: 4096 pair-feature elements per K chunk.
- Persistent state: 1 or 16 FP32 distance sums, magnitude sums, difference
  counts, owned by the output CTA. Index and output slot maps are read-only.
- T4x4 operands: [4,1,256] and [1,4,256], C-contiguous feature dimension;
  intermediate [4,4,256]. Compiler-managed registers/shared reduction only.
- Live phases: load operands -> delta/magnitude -> K reduction -> release
  chunk tensors -> accumulate scalar state -> interval -> terminal stores.
  No operand reuse across chunks, custom barrier, swizzle, or async overwrite.
- Ready graph: slot/tile IDs -> loads -> elementwise arithmetic -> reductions
  -> next chunk accumulation -> interval -> disjoint stores. Terminal outputs
  overwrite every requested slot exactly once; padding slots have no writer.
- Expected tradeoff: T4x4 reduces distinct operand loads by at most 4x for full
  tiles, but arithmetic grows by `16*active_tiles/requested_pairs`; P16 avoids
  padding but cannot statically broadcast arbitrary indexed operands.
- Approximate peak chunk working set: several 4096-element temporaries, about
  100-200 registers/thread before compiler allocation. Resource/spill evidence
  must be recorded; no arbitrary register cap or shape retune is allowed.

## Execution and measurement

- Host: gpu-host-8, expected hostname gpu-host-8.
- Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
- Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python.
- Observed versions: torch 2.11.0+cu130, Triton 3.6.0, numpy 2.3.1,
  scipy 1.17.0; driver 590.48.01, toolkit 13.1.115.
- GPU UUID: GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3. 600W cap, clocks unchanged.
- Hold existing GPU1 campaign lock AND G6 GPU1 lock; 5-second quiescence,
  monitor compute processes every 0.25s; abort only own group on contamination.
- DO NOT TOUCH GPU0 / foreign PID919690 or any foreign process/service.
- First correctness/resource process, then compute-sanitizer memcheck and
  synccheck, then two fresh timing processes in forward/reverse method and
  workload orders. Do not overwrite any failed attempt.
- Timing: resident-GPU, default stream, no Graph, 20 warmups per method;
  40 retained rounds of all three methods, rotate method order each round;
  each sample is a CUDA-event block of 20 calls divided by 20. Synchronize
  around measurements; no profiler timing as estimator.
- Sustained diagnostic: 500 consecutive launches per method/workload, measured
  separately. Correctness before/after with independently allocated outputs.
- Primary estimator: per-process median of round-paired log(P16/T4x4), then
  geometric mean across the two processes. Report both processes, all samples,
  p10/median/p90, order split, and diagnostic 2-process interval limitations.
- Structural outputs: active tile count, mean/p50/p90/max occupancy, padded
  arithmetic amplification, and host layout/queue planning time.

## Stop rules and claim limits

Advance to an online routing experiment only if two actual layouts show an
opposite winner: P16/T4x4 >=1.15 in one and <=1/1.15 in another, in EACH fresh
process, without sustained sign reversal. Correctness and isolation must pass.
If only synthetic controls cross, record partial mechanism feasibility but do
not advance the real-data story. If all actual layouts favor one method, count
alone is not proven sufficient, but this A0 does not establish the proposed
layout-based routing thesis. No expensive formal campaign is authorized by an
A0 negative. No novelty, whole-join speedup, online routing, production safety,
or exact final-output claim can be made from this component test.

Artifacts: new `src/g8_*`, `src/run_g8_*`, `results/g8_*`, `raw/g8_*`,
`artifacts/g8_*`. Old kernels, papers, G5 negatives and keepers remain untouched.
Rollback: remove G8 from the current routing note; retain all new evidence.
