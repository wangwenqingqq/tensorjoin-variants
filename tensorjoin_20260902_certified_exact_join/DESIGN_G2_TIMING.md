# G2 public-denominator timing adapters and cheap screen

Date: 2026-09-03

Status: designed before implementation; no timing evidence in this document.

## Purpose

G2B full-scale correctness/resource smoke already established byte-identical
sorted canonical outputs for GDS-Join FP64, MiSTIC FP64, and TensorJoin.  The
smoke timings used different scopes and are therefore diagnostic only.  This
stage implements the frozen public end-to-end denominator from
`PROTOCOL_G2.md` without changing that protocol.

## Frozen input/output contract

- Host/GPU: `gpu-host-8`, physical GPU0 only.
- Input: CIFAR-10-GIST, 60,000 x 512, source values exactly as the frozen
  float32 array in `data/g2b_cifar60000/vectors_f32.npy`.
- Radius: epsilon 0.62890625; squared comparison threshold
  0.3955230712890625.
- Start boundary: the source float32 array is resident in ordinary pageable
  host memory. CUDA context initialization, dynamic-library loading, binary
  loading, and Triton compilation have completed, but no method-specific
  transformed input, index, schedule, certificate, or output allocation is
  retained.
- End boundary: the complete sorted canonical directed uint64 pair IDs are
  resident in host memory.
- Included: method-specific widening/quantization, CPU reorder/index/schedule,
  output allocation, H2D, search/refinement/compaction, D2H, ID remapping,
  symmetry normalization where required, and host sorting.
- Excluded: file I/O, process startup, CUDA context initialization, compilation,
  disk serialization, oracle construction, hashing, and correctness comparison.

All implementations use `time.perf_counter()` or `omp_get_wtime()` around the
same semantic boundary. The timer is host wall time because required CPU and
transfer work is part of the public denominator.

## Adapter designs

### GDS-Join FP64 keeper

Load the frozen float32 source before the timer. Load the pinned GDS shared
library and initialize a context before the timer. After the timer starts:

1. widen float32 to contiguous float64;
2. allocate the count vector;
3. call `GDSJoinPy`, including its reorder/index, transfers, and search;
4. retrieve neighbors with `copyResultIntoPythonArray`;
5. reconstruct original query IDs and canonical uint64 pair IDs;
6. host-sort the canonical IDs and stop the timer.

The library is called once in a fresh process. No GDS index or transformed data
is warmed or reused.

### MiSTIC FP64 keeper

Build a separate adapter copy; do not modify the frozen G2A adapter. The binary
loads the frozen float32 raw source into pageable host memory and initializes a
CUDA context before the timer. After the timer starts:

1. widen float32 to float64;
2. compute the dimension order and reordered data;
3. build MiSTIC's node index;
4. transfer and search;
5. reconstruct original IDs, normalize any absent self pairs, and host-sort;
6. stop and print `G2B_PUBLIC_SECONDS` before optional pair-file serialization.

The adapter changes only input widening, canonical-output materialization, and
timer placement. The upstream search/index code and compile flags stay fixed.

### TensorJoin candidate

Import and compile all three exact-path Triton kernels before the timer using
the same constexpr specialization as the full run. Destroy warmup tensors and
empty the allocator cache; retain no transformed source, schedule, certificate,
or result. Load the float32 source before the timer. After the timer starts:

1. construct per-vector INT8 codes, scales, rigorous residual bounds, norms,
   and the upper-triangular tile schedule;
2. transfer all method inputs and allocate fixed worst-case batch buffers;
3. run INT8 Tensor Core certification, guarded FP32 filtering, and FP64
   refinement for every batch;
4. copy admitted upper-triangle IDs to host;
5. concatenate, normalize directed symmetry, host-sort, and stop the timer.

Compilation warmup deliberately uses no production data and is not a reusable
method state. Candidate allocations and all production H2D transfers remain
inside the timed boundary.

## Cheap screen (diagnostic only)

Two fresh-process direction-balanced rounds over the three exact methods:

- round 0: GDS -> MiSTIC -> TensorJoin;
- round 1: TensorJoin -> MiSTIC -> GDS.

This reverses TensorJoin relative to each keeper. Before every process, hold
`@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`, record host/GPU state,
and require 30 continuous seconds with no compute process on physical GPU0.
During a process, abort only the launched process group if a foreign GPU0 PID
appears; never signal the foreign PID. Preserve every attempt and its order.

The screen advances to the frozen eight-round campaign only if all conditions
hold:

1. all six processes produce the frozen canonical count and hash with no
   structural or capacity failure;
2. TensorJoin is faster than both keepers in both matched rounds;
3. define the faster keeper by the lower two-observation marginal median;
4. TensorJoin's per-round speedup over that keeper is at least 1.50x in both
   rounds and its paired geometric mean is at least 1.60x;
5. no foreign-process contamination or output-capacity failure occurs.

Passing is only a resource-allocation decision, not publishable performance
evidence. Failure rejects the current public-path implementation at this
workload; reopening requires a named mechanism that changes the measured
denominator rather than post-hoc timing exclusions.

## Verification ladder

1. adapter build and exact source/binary hashes;
2. one unmeasured/diagnostic adapter validation per changed implementation;
3. byte/hash equality to the frozen full-scale three-way output;
4. two-round cheap screen under the shared lock;
5. only after a screen pass: candidate memcheck/stress, then the frozen eight
   fresh-process formal campaign and predeclared bootstrap estimator.

