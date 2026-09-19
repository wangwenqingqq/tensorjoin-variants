# G9 A0: certified integer-TC refinement, optimistic staging

Frozen 2026-09-04 before implementation or G9 occupancy/timing observations.
Experiment: `tensorjoin_20260904_g9_int14_tc_refinement_optimistic`.

## Scope and Gate 0

Question: can a concrete certified TC refinement path beat the strong P16
batching control on the SAME actual G3B-R1 uncertain pairs, even if initial
encoding, row layout, tile metadata, and intermediate queue construction are
free? All actual numerical repair computations MUST be timed.

This is a bounded implementation feasibility test, not a novelty pass. Integer
limb decomposition and accurate GEMM emulation are prior art, including
https://arxiv.org/abs/2306.11975 and https://arxiv.org/abs/2504.08009.
TC distance computation/reuse is also prior art:
https://arxiv.org/abs/2508.21230. Do not claim the decomposition, use of TC,
or generic sparse-to-dense packing as a contribution. A future research thesis
requires a non-subsumed mechanism and independently justified end-to-end gains.

G8's FP32 4x4 negative stays intact. This tests a different native arithmetic
contract, not another FP32 tile size. G5's full-scale failure is not explained
or overwritten. No paper edits or full-scale campaign are authorized by A0.

## Frozen workload / semantic contract

- N=4096, D=512, original FP32 CIFAR-GIST vectors.
- Input NPY SHA256 e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462.
- Real logical pair count 102079; sorted uint64 raw SHA256
  6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42.
- Squared threshold T=0.5686872086178483, interpreted as that exact binary64
  value. Closed predicate: squared Euclidean distance <= T.
- Same six layouts/families as G8: real identity, random, degree-informed,
  PCA1; synthetic clustered/scattered clique control. Reuse G8's frozen layout
  generator without edits. The latter two are NOT actual uncertainty.
- Keep the G8 tile-sorted pair queue for P16 and diagonal packing; map physical
  TC tile results back to those same slots. Row permutations preserve logical
  undirected pairs. No same-count but different-pair substitution.
- Final output: one fully resolved predicate byte per requested pair on this
  frozen input. Every slot must match the CPU FP64 oracle; additionally, final
  FP64 repair is interval guarded and must leave zero unresolved states on all
  main workloads. Do not silently classify a remaining interval overlap.
- Intermediate three-way masks MAY differ across precision paths. Each repair
  queue must equal the actual preceding GPU uncertain mask, checked untimed.

## Encoding and proof boundary

Use per-row power-of-two scale s and q=round_even(x/s), |q|<=16319.
Represent q=128*h+l, h=floor((q+64)/128), with h in [-127,127],
l in [-64,63]. Store both limbs as signed INT8. Four ordinary INT8 MMAs give
exact INT32 hh, hl, lh, ll accumulators, then combine in INT64:

`dot_q = 16384*hh + 128*(hl+lh) + ll`.

At D=512, all INT32 products/sums fit; |dot_q|<=D*16319^2<2^38.
Conversion to FP64 and multiplication by row power-of-two scales are exact
within the frozen checked exponent range. q norm squares are exact INT64 sums
precomputed on the host. Residual norms are upper bounds proved by exact
integer comparisons on a checked 2^-40 coordinate grid. No tensor floating
accumulation precision assumption is needed. Full derivation and executable
preconditions: `NUMERICS_G9_INT14.md`.

The TC stage supplies exact quantized dot products only. A separate per-pair
certificate uses explicit directed FP64 arithmetic to enclose reconstructed
distance plus original-coordinate residuals. Only surviving intervals use
the G8 FP32 certificate, then interval-guarded FP64 direct differences.

## Fixed methods (no tuning menu)

- **P16**: G8 direct-difference FP32 arithmetic, 16 requested pairs per CTA;
  then FP64 repair of its actual uncertain subset. Output is aligned bytes,
  not the historical atomic-compacting G3C public pipeline.
- **TC16physical**: one physical 16x16 tile per CTA; four INT8 limb dot products
  at K-block64, 4 warps, 2 stages; all padded products are computed. Store only
  requested dot products through a slot map. Follow with per-pair directed
  certificate, actual FP32 repair subset, and actual FP64 repair subset.
- **TC16diagonal**: take each consecutive group of 16 requested pairs; left
  endpoints form 16 rows, right endpoints 16 columns. Same 16x16 native tile,
  same four limb products, but only 16 diagonal outputs are requested. This
  predeclared regular-pair packing control costs about 16x padded arithmetic,
  avoiding a verdict based only on unstructured physical tiles. Gather limbs
  directly from the row arrays; do not create giant duplicated vector buffers.
- Same TC arithmetic/kernel parameters across both packings. Only the row/
  column index arrays and output slot maps differ. No tile-size/precision
  sweeps, register caps, or favorable-input selection after results.

## Optimistic denominator

Included: every TC dot kernel, per-pair interval certificate, required FP32
refinement, required FP64 refinement, all corresponding input/output traffic
and aligned output overwrites. Certificate endpoints are also written for
numerical audit and these writes remain timed. Baseline includes its FP64 repair too.

Excluded for BOTH sides: initial INT8 filter, input encoding/preprocessing,
row permutation, tile/queue construction, dynamic count discovery, output ID
compaction/export, host ingestion. Intermediate queues are prebuilt from an
untimed actual run, never inferred from the oracle's answer. This is an
optimistic fixed-input staging experiment, NOT an implementable online router.
Queues must reproduce and partition the live stage outputs exactly.

Record separate filter/repair diagnostic stage timings, but decide using the
measured complete staged path, not a sum of selected component medians.

## Design / ownership / ready graph

Native geometry: documented `mma.sync` s8*s8->s32, normally m16n8k32 on SM120;
verify actual emitted PTX/SASS. No WGMMA, TCGEN, TMEM or undocumented opcode.
Same four warps own load, compute and store phases; Triton owns any shared
layout and synchronization. No custom async or cross-CTA handoff.

| Phase | Persistent state | Temporary state / last use | Expected pressure |
|---|---|---|---|
| Metadata | 16 row IDs, 16 column IDs | addresses | small |
| Limb MMA | four 16x16 INT32 accumulators | four 16x64 / 64x16 INT8 operand panels | about 8 accumulator registers/thread plus feed/control state |
| Combine/store | INT64 quantized dot | INT32 accumulators die after combination | two-register outputs, no precision loss |
| Certificate | per-pair FP64 interval endpoints | scales, exact norm2, upper norms | independent elementwise state |
| P16 repair | 16 distance/magnitude sums | 16x256 FP32 chunks | G8 control resource class |
| FP64 repair | per-pair FP64 distance sums | D=512 direct-difference chunks | separate kernel, no live TC state |

Ready graph: row/column indices -> read-only limb loads -> synchronous MMA
updates -> complete INT64 combine -> disjoint requested-slot stores -> stream
ordering -> certificate -> stream ordering -> FP32 subset -> FP64 subset.
No buffer is reused before the preceding same-stream kernel finishes.
Guarded allocations and complete overwrite tests cover tails and stale data.

Expected benefit: faster native multiply-accumulate and fewer FP32 full-vector
refinements. Added costs: four limb products, padded cells, INT64 combination,
FP64 certificate, extra launches and residual refinement. Record all of them.
Static admission: correct native IMMA, no spills/local stack, resources within
device limits. Failure is retained; no silent parameter retune.

## Host and safety card

Host gpu-host-8 / gpu-host-8; project
@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
Python @TENSORJOIN_ROOT@/isaacsim6/env/bin/python.
GPU1 UUID GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3.
Observed driver590.48.01, CUDA toolkit13.1.115, torch2.11.0+cu130,
Triton3.6.0, numpy2.3.1, scipy1.17.0. 600W cap, no clock/power changes.
Reuse both existing GPU1 lock files; 5-second quiescence and 0.25s monitoring.
Stop only our own process group on foreign-process contamination. Do not touch
GPU0/2/3/4 or foreign PIDs919690,1043857,1043860,1043861 seen in preflight.
Storage is tight; no dataset download or duplicate large vector corpus.

Gates: host exact encoding audit -> quantized-dot equality and interval
enclosure on all requested pairs -> final output/state/queue correctness ->
static native-code audit -> memcheck and synccheck -> two fresh ordered timing
processes. Include signed high-entropy data, zero/equal vectors, last partial
16-pair group, and threshold-neighbor checks as correctness-only fixtures.
Further race/init/Graph/stress gates are needed before production admission.

## Measurement and stopping

No Graph, default stream, resident arrays. Warmup20 complete paths; 40 rounds,
20 complete paths per CUDA-event sample, rotating method order. Second process
reverses method and workload order. Separate500-path sustained diagnostics.
Retain all raw samples and resource/binary hashes; no favorable-order retry.
Primary ratio: per-process median round-paired log(P16/candidate), exponentiate;
aggregate geometrically across two processes. Report p10/median/p90, marginal
ratio, both process results, order split, and the limits of n=2 inference.

Advance to a NEW fully-costed protocol only if a TC candidate beats P16 by
>=1.15x in BOTH processes on at least one ACTUAL layout and sustained sign
agrees. Wins confined to synthetic controls do not qualify. This is an
opportunity gate, not paper-grade acceptance. If no candidate qualifies, stop
these two TC packing implementations; do not tune or build an online router.
This does not prove that all conceivable TC refinement algorithms are slow.

Artifacts use G9 names only. Existing G8 kernels, results and manifests remain
untouched. Rollback removes G9 from current routing, retaining all evidence.
