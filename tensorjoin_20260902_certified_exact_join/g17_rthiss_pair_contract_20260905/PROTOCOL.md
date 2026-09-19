# G17: RT-HiSS full pair-contract admission, not a speedup campaign

Declared 2026-09-05 before new adapter implementation or GPU measurement.
G16's narrow positive is retained. Gate0 remains unknown: deterministic bounds,
precision escalation, GPU packing and TC self-join have antecedents. This gate
closes a necessary modern-comparator semantic gap, not an attempt to rescue a
subsumed thesis with packaging or data breadth. No60K/public timing admission.

## Primary-source refresh / novelty kill test

RT-HiSS (https://arxiv.org/abs/2609.01975, submitted2026-09-02, SC26 forthcoming)
is direct same-task prior art: RT index/filtering, CUDA refinement, two-pass
size estimation, batching, load balancing and compressed output are established.
Its pinned public code is https://github.com/revanthmunugala/rt-hiss at
`a42fc69cc4b602dc83071b029d185a41e69a04bd`, OWL gitlink
`c7c3a3ea35b17b5c096a3802ba74b9d8b4e2772a`. G7 built/smoked it, not pair-tested it.
FaSTED (https://arxiv.org/html/2508.21230v1) already covers TC self-join and
memory-path optimization. Merely moving packing/preparation to GPU is not new.
The current exact-reference/adaptive-arithmetic contract differs from approximate
acceptance, but a non-incremental systems mechanism remains unproved. Do not
claim generic tree/TC integration, filter-refine, dynamic counts or buffers new.

## Frozen execution card

Host: gpu-host-8, local -> tiaoban -> root@192.0.2.8:22223; gpu-host-8.
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
Local source: matching project under @LOCAL_WORKSPACE@/paper.
No Git root; preserve G5/G15/G16, external pinned source and G7 adapter.
GPU: physical2, UUID GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245, RTX PRO6000.
Driver590.48.01; nvcc13.1.115; OptiX9.1 installed read-only at
@TENSORJOIN_ROOT@/RT-TIDE/deps/optix-9.1. Build -j4, SM120, default upstream
shared/shared refinement, compressed masks, workload sorting, auto KD height.
Python: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python.
Lock: /tmp/tensorjoin_gpu2_campaign.lock plus existing G5 guard.
Do not touch: GPU1 PID1460887 and any foreign users/processes, including GPU6
PID1696161 observed at initial preflight. Reverify each GPU call; stop only our
child on foreign occupancy or timeout. No service, power or clock changes.
Artifacts: this directory's data/src/adapter_a0/raw/results/artifacts; append-only
attempt IDs, command/build/source/library/binary/output/guard receipts.
Rollback: retain separate adapter, disable admission; no original source edits.

## Adapter design / data ownership

- Read stored contiguous FP32 via a checked raw file for fidelity. Disk IO is
  outside any future public denominator. Capture the original values and all
  dimension/point permutations; do not change variance order, KD grouping,
  RT bounds, refinement arithmetic, compression kernels or tiling parameters.
- At the existing point reorder, save new-index -> original-row. Both query
  and candidate IDs use this map. Dimensions do not change point identity.
- Keep compressed bit positions. Recover primitive by upper_bound on cumulative
  word offsets; localBit = globalBit -32*prefix[primitive]. Divide/modulo by
  that primitive's candidate-query count to recover primitive point/query slots.
  Then map groupedPoints and candidatePoints through the point permutation.
- Copy the necessary prefix/count/ID metadata explicitly and charge it in any
  later performance variant. Correctness-only capture also exports the raw
  bitmask, compressed positions and dimension/permutation metadata. Its disk IO
  and debug captures make this **not a public timing implementation**.
- Existing GPU producer/refinement/compression kernels remain byte-for-byte
  source invariant. Existing synchronization remains: RT sync -> metadata/counts
  -> refinement sync -> compression sync -> host read/copy -> decode/sort.
  No new shared/async protocol, kernel, stream or Graph support.
- Host decoder owns the copied masks/metadata until all pairs are materialized;
  it emits every directed pair and self bit that actually exists. Never fill
  missing pairs, deduplicate away an error, or rerank only positives to hide FN.
- Fail closed before narrowing: N<=4096,D512, positive finite epsilon; full
  query/data permutation and 32-bit global mask positions must fit, counts and
  padding must be consistent. Do not silently fix overflow or reinterpret bits.
- Compression output count must equal mask popcount and refinement count.
  Decoder correctness is distinct from RT coverage and floating-point predicate
  equivalence. Sanitizer/compilation is not either numerical or latency proof.

## Frozen cheap matrix and admission

Small D512 fixtures: real31, deterministic signed31, shuffled onehot/duplicates31,
all-zero31 at positive epsilon, near-boundary31 at epsilon1, cancellation31,
subnormal31 at positive epsilon. Fixed random seed17052026. No fixture tuning.
Use inclusive threshold, all directed original IDs including self. For the
real4096 anchor, use existing G2A input/subset/oracle and original epsilon text;
record native rounded-FP32 epsilon/squared threshold separately. Do not silently
replace the frozen FP64 threshold with the native threshold.

1. CPU decoder unit tests before GPU: empty primitives/padding, unordered
   compressed positions, nonidentity point permutation, duplicate/invalid bits,
   multi-batch accumulation. Every invalid case fails closed.
2. Small GPU matrix: independently reconstruct the uncompressed bitmap and
   decoded IDs; check original-ID permutation/bijection, counts, duplicates,
   bounds and self inclusion. Compare frozen FP64 and native-threshold FP64.
   Replay disputed native decisions with scalar C fmaf in the actual reordered
   dimension order; classify decoder error, candidate miss, native arithmetic
   difference or threshold conversion. Preserve all cases, including negatives.
3. Real4096 characterization is admitted only if decoding and memory/count
   structure pass the small matrix. A small native arithmetic discrepancy does
   not forbid a bounded characterization of the real input; it does forbid
   exact-comparator/performance promotion until a separately designed repair.
   Any unsafe launch, permutation/capacity/decoder failure stops expansion.
4. Run memcheck and synccheck for the admitted small and4096 adapter. These are
   bounded safety gates only. Retain upstream and adapter selected function
   source/binary identities; build/runtime correctness is not generated-code
   precision proof or performance. No new throughput claim from this gate.
5. If native output fails reference equality, do not call RT-HiSS generally
   incorrect or drop it. Localize the discrepancy, record the precise domain,
   and predeclare a minimally invasive conservative repair before further work.
   If it matches, admit only same-output correctness on the tested matrix;
   complete timing, formal confidence and60K remain separate gates.

No paper drafting/Overleaf action is admitted by this gate alone. Record
implementation, mechanism and thesis decisions separately, and preserve G16's
positive and all earlier negative evidence regardless of comparator outcome.
