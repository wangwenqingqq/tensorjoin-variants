# G19: reusable, diagnostic-free complete-cost admission

Designed 2026-09-05 before G19 source changes or GPU measurements.

## Decision and Gate 0

This is a bounded comparator-fairness experiment, not a new research direction.
G10 prior-art overlap and G16/G18 novelty restrictions remain. Reusable engine
wrapping, disabling diagnostics and conservative arithmetic are not claimed as
TensorJoin contributions. No paper drafting, broad data campaign or publication
is authorized by a positive G19 screen alone. A non-incremental thesis still
needs its independent novelty kill test.

Hypothesis: a separate RT-HiSS wrapper can reuse only data-independent OWL
context/module/program/pipeline state, rebuild every input-dependent structure,
materialize all canonical original IDs, and release per-call resources. Once
admitted it can be compared with the G16 GPU-prepared TC path and equally
GPU-prepared pedantic FP32-first control under one complete-cost denominator.
G18 remains immutable; diagnostic-on/off are separate G19 binaries.

## Run card

- Host: gpu-host-8, local -> tiaoban -> root@192.0.2.8:22223.
- Root: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
- Experiment: g19_reusable_complete_cost_20260905, no project Git root.
- Environment: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python; nvcc 13.1.115,
  driver 590.48.01, SM120. Record live versions and source/binary hashes.
- Target: physical GPU3, GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2.
- Isolation: existing G5 30-second quiescence/0.25-second monitor, one
  /tmp/tensorjoin_gpu3_campaign.lock. Abort on foreign occupancy; never signal
  foreign jobs. Retain blocked slots and require an explicit additive restart.
- Do not touch: all other users/processes, GPUs 0/1/4/5/6 currently occupied;
  clocks/power/MPS settings unchanged. OMP_NUM_THREADS=8, torch CPU threads=1,
  NVIDIA_TF32_OVERRIDE=0. CPU contention is a reported limitation.
- Logs/results/builds stay under G19; selected guard receipts are copied at
  closure. No port, service, model, checkpoint, training or robot action.
- Rollback: close only owned processes/engine; old dispatch and G18 unchanged.

## Frozen public operation

Prevalidated, immutable contiguous pageable-host FP32 N x 512 input and stored
FP64 squared threshold -> independently owned, sorted canonical host uint64
directed pair IDs including self. Initial matrix is exactly G17 manifest_r1's
nine inputs, N <= 4096, finite absolute coordinates <= 1. The timing anchor is
only CIFAR4096, T=0.5686872086178483, epsilon=0.7541135250198396, output 262,144
IDs with SHA-256 da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d.
No comparison with stale G16 60K times. No general exact-real/candidate proof.

The reference is the frozen complete-ID oracle, not a requirement that every
implementation use the same internal reduction order. Any scalar threshold
transport must retain the stored FP64 value where required, rather than silently
rounding it to FP32. Explicit new scalar-precision specialization, if needed,
is separately recorded and validated; no change to the G16 originals.

One data-independent engine is initialized/warmed outside the timer per method.
RT retains OWL context/module, geometry type/object, raygen, params and compiled
program/pipeline. It retains NO input, reordered vectors, candidate structures,
geometry/BVH, output or computed thresholds between operations. TensorJoin and
FP32 retain JIT/library handles and ordinary allocator caches, not input-derived
metadata/results. Engine startup/destruction are separate lifecycle evidence,
not subtracted from observed per-call times after measurement.

Inside timer: per-call cutoff construction, input clone/reorder/quantization or
norm construction, allocation, H2D, geometry/BVH/candidates, every refinement,
count readback, compaction, metadata/output D2H, original-ID recovery, canonical
sort, owned host output and per-call cleanup. RT shader binding records remain
input-dependent and are rebuilt inside. Exclude identically: file/oracle IO,
common input validation, process/CUDA initialization, library/JIT preparation,
post-timer hashes/oracle comparison and evidence serialization. No console/disk
export, diagnostic mask download or diagnostic per-candidate atomics in the
admitted RT timing binary. Required output counts/capacity checks remain.

RT default shared/shared refinement geometry, FP32/F64 predicate, RT traversal,
compression and ID mapping are retained. A minimal unsorted compressed-bit
decoder replaces evidence-only redundant sorting/duplicate scans; canonical
output sorting remains charged and every resulting ID is checked outside timing.
This adapter is not claimed to be the globally fastest conceivable ID exporter.

## Required gates before timing

1. Build explicit D512 SM120 diagnostic-on/off libraries. Freeze source diffs,
   compiler flags and binaries. Inspect selected code: no new MMA/FTZ/spills in
   repaired predicate; compression unchanged. No ownership/barrier kernel change.
2. Both RT binaries: nine fixtures in forward and reverse order on a reused
   engine, zero missing/extra IDs; diagnostic build exports actual maps/masks and
   work counters for G17 independent decoder checks. Compare off outputs exactly.
   TC and FP32: all nine fixtures, same complete oracle, fresh host allocations.
3. Memory and synchronization sanitizers on the exact new RT-off wrapper with
   changing real4096/zero32 inputs; full leak check requested. Memory sanitizer
   on both TC and FP32 full operators at these two inputs. Sanitizer errors stop
   admission; do not use an on-build pass to bless the off build.
4. Two complete RT engine creation/destruction cycles, 100 total alternating
   real4096/zero32 calls (50 per engine), immutable input hashes and exact outputs.
   After two warm calls per engine, device-used growth must stay <=16 MiB and
   host RSS growth <=64 MiB. TC/FP32: 32 complete alternating calls each, no
   live tensor allocation accumulation; reserved caches reported separately.
   No arbitrary streams/Graph/concurrent engine/sustained60K assertion.
5. Runtime-selected CUDA code on actual real4096, per-method library/cache hashes
   and unchanged code across timing; OptiX JIT identity is not inferred from PTX.

## Cheap direction-balanced screen (only after gates 1-5)

Two independent process blocks, method orders RT/TC/FP32 and FP32/TC/RT. Each
method process prepares its exact specializations, makes two complete unmeasured
warm operations, then retains five complete host-to-host observations. Synchronize
at each boundary. Store every observation, output hash, method/order, memory,
process and occupancy record. Do not select observations by latency.

Primary screen: median FP32 / median TC and median RT / median TC within EACH
block, and their minimum (strongest admitted external/control denominator).
Narrow screen passes only if that minimum >=1.10 in both blocks. Report all
method p10/median/p90, both order ratios, process wins, arithmetic/geometric means
and marginal ratio. With two blocks, no inferential CI, tail or formal speed claim.
Failing timing rejects the narrow performance screen, not numerical correctness
or all future mechanisms. No favorable rerun; contamination requires a documented
addendum, retains originals and is not silently mixed into clean measurements.

## Initial claim states

Reusable RT ownership: unknown. Diagnostic-free complete IDs: unknown.
Bounded memory/safety/stress: unknown. Same-output three-way complete-cost
advantage: unknown. Novelty and paper readiness: unknown, not promoted.
