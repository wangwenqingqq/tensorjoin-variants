# Design G2B: Triangular, Chunk-Bounded Exact Cascade

Date frozen: 2026-09-03

Experiment family: `tensorjoin_20260903_external_exact_selfjoin_g2`

## Decision before implementation

G2B will not estimate one monolithic output capacity. It will scan only the
upper triangle of the self-join in bounded batches of output tiles. For a batch
containing `T` tiles of shape 64 x 64, every compaction buffer has capacity
`T * 4096`. This is a deterministic worst-case bound on the number of unique
pairs that can enter any mutually exclusive stage in that batch. Therefore a
successful implementation needs neither an output-count oracle, a sampling
margin, nor a retrying full scan.

This replaces the weaker capacity-sampler idea named in `DECISION_G2A.md` with
a proof-based capacity rule. It does not rewrite the G2A result: the new route
must first reproduce the frozen G2A oracle before it is admitted to G2B.

## Target and roofline hypothesis

- Target: RTX PRO 6000 Blackwell, physical GPU0, live `sm_120` capability
  recorded at execution; CUDA/Triton versions and clocks recorded separately.
- Shape: Cifar60K, N=60,000, D=512, source float32 coordinates.
- Threshold: epsilon=0.62890625 and the FP64 direct-difference predicate frozen
  in `PROTOCOL_G2.md`.
- Hypothesis: high-dimensional GDS/MiSTIC pruning leaves enough candidate work
  that a dense INT8 Tensor-Core certificate followed by sparse FP32/FP64
  refinement can win despite scanning the upper triangle.
- Material risk: host canonicalization, atomic compaction, or ambiguity may
  dominate; the candidate is rejected rather than rescued if the frozen
  end-to-end gate does not pass.

## Numeric and output contract

- Source layout: row-major contiguous float32 `[N,512]` in pageable host memory.
- Quantization: symmetric per-vector signed INT8 codes and one float32 scale;
  INT32 dot accumulation.
- Certificate: reconstructed distance plus stored per-vector reconstruction
  error and `BOUND_PAD=1e-4`.
- Guard: ambiguous pairs use direct float32 differences; pairs within
  `FP32_DISTANCE_GUARD=1e-3` of the squared threshold use direct float64
  differences over exact float32 widening.
- Internal pair ID: int64 `i*N+j`, with only `i<=j` admitted.
- Public output: sorted little-endian uint64 directed IDs including self. Every
  non-self upper pair is expanded to both directions before host sorting.
- Exactness is empirical against independently validated FP64 implementations;
  the bound pad is not promoted as a universal floating-point proof.

## Dispatch boundary and tile schedule

- `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=64`; refinement block K=256.
- The host constructs `np.triu_indices(ceil(N/64))` in row-major order and
  transfers the two int32 tile-coordinate arrays to the GPU.
- One Triton program consumes one scheduled upper tile. Diagonal tiles also
  apply the element mask `i<=j`; padded rows/columns are masked.
- Frozen production tile batch: 4,096 tiles. The last batch may be smaller.
- The candidate is fail-closed outside `D=512`, square self-join, float32 source,
  or int64-addressable pair IDs.

## CTA/warp role table

| Unit | Role |
|---|---|
| Host | Build upper-tile schedule, quantize, transfer state, launch bounded batches, expand/sort canonical output |
| Certificate program | One 64x64 upper tile; INT8 dot, distance interval, mutually exclusive accept/reject/ambiguous classification |
| FP32 refinement program | One ambiguous upper pair; direct-difference accumulation and accept/reject/FP64 routing |
| FP64 refinement program | One guarded upper pair; exact-widened direct-difference predicate |
| Four warps per program | Cooperative Triton-generated INT8 dot or scalar reduction; no cross-CTA persistent state |

## Data layout and persistent ownership

- Host owns source float32 rows, tile schedule, per-batch accepted chunks, and
  final canonical directed IDs.
- Device persistent state owns float32 rows, INT8 codes, transposed codes,
  scales, reconstructed norms, upward-rounded reconstruction errors, and tile
  schedule.
- Device batch-local state owns accepted IDs, ambiguous IDs, FP64 IDs, and five
  int32 counters. No all-pairs status matrix or global monolithic result buffer
  exists.
- Each upper pair is owned by exactly one scheduled tile and exactly one final
  decision stage. Atomic positions are the only cross-program shared state.

## Capacity proof

For a batch with `T` scheduled 64x64 tiles, let `C=T*4096`. Masking can only
reduce the number of valid upper pairs. The certificate partitions every valid
pair into accept, reject, or ambiguous, so both accept and ambiguous counts are
at most `C`. FP32 partitions the ambiguous set, and FP64 consumes a subset; the
total final accepted count is also at most `C`. Allocating each int64 buffer at
`C` therefore precludes overflow independent of selectivity. Runtime counters
still check the proof; any overflow is a correctness failure.

## Phase live set and ready graph

| Phase | Large live objects | Last use / overwrite rule |
|---|---|---|
| Quantize | host source, codes, scales, errors, norms | host temporaries may be released only after all device copies finish |
| Certificate | persistent quantized state, schedule slice, accepted/ambiguous buffers | ambiguous buffer remains live through FP32; accepted buffer accumulates all stages |
| FP32 | source float32, ambiguous IDs, accepted IDs, FP64 IDs | ambiguous IDs may be overwritten only after FP32 completes |
| FP64 | source float32, FP64 IDs, accepted IDs | FP64 IDs may be overwritten only after refinement completes |
| Drain | accepted IDs, counters | next batch may reuse buffers only after accepted slice and counters reach host |
| Canonicalize | host upper chunks, expanded reverse IDs | chunks released after the final sorted array and hash are materialized |

Ready graph per batch:

```text
schedule slice ready -> certificate -> counters ready
-> FP32 on exactly ambiguity_count -> counters ready
-> FP64 on exactly fp64_count -> counters ready
-> accepted D2H -> buffer reuse by next batch
```

There is no asynchronous buffer overlap in the correctness smoke. A later
double-buffered optimization would be a separate attributable candidate.

## Work and movement ledger

Relative to the G2A full directed scan:

- useful dense dot products: from N^2 to N(N+1)/2 (about 50%);
- refinement evaluations: once per unordered pair rather than twice;
- final public IDs: unchanged after symmetric expansion;
- added movement/control: two int32 tile schedule arrays, per-batch counters,
  multiple launches and drains, host symmetric expansion;
- invariant: numeric predicates, source values, threshold, self-pair policy,
  and canonical output.

No claim is made yet that the saved arithmetic exceeds the new batching and
canonicalization costs.

## Verification ladder and stop rules

1. **G2A2 exact admission:** new triangular runner must reproduce the existing
   262,144 directed-ID oracle with zero missing, extra, duplicate, invalid, or
   overflow IDs in two isolated executions.
2. **Stage safety on G2A2:** no direct INT8 false acceptance/rejection and no
   guarded FP32 false acceptance/rejection relative to the upper-triangle
   oracle.
3. **G2B smoke:** one isolated full execution per exact method; GDS, MiSTIC, and
   TensorJoin must agree on count and canonical hash. The pinned FaSTED accuracy
   record's FP64-GDS count 3,926,074 is a diagnostic expectation, not the sole
   oracle.
4. **Resources:** record peak host/device memory, tile count, batch count, all
   stage counts, and every overflow counter.
5. **Formal timing:** forbidden until the smoke passes and the public adapters
   implement the same host-float32-array to sorted-host-ID denominator.
6. **Promotion:** apply the exact eight-round criteria in `PROTOCOL_G2.md` with
   no post-hoc estimator or threshold change.

Immediate rejection conditions are any G2A2 mismatch, unsafe stage decision,
overflow, lower-triangle internal ID, duplicate scheduled pair, or need to retry
a tile batch. Retain the failed artifact and diagnose before any full run.

## Comparator fairness boundary

- GDS-Join is exact per its upstream README. For D=512, its high-dimensional
  optimizations are enabled; `STAMP=0` is retained because upstream labels
  `STAMP=1` a low-dimensional optimization not extensively tested at high D,
  and the canonical Python output interface is the audited path.
- MiSTIC is FP64 as in its original paper configuration, not the FP32 variant
  used in FaSTED's performance figure.
- FaSTED's Cifar accuracy record reports 3,926,078 mixed-precision pairs versus
  3,926,074 FP64-GDS truth pairs (838 test-only and 834 truth-only). It remains
  lower-quality context and has no exact-gate vote.
- G2B public timing must include method-specific float32-to-float64 widening for
  GDS/MiSTIC and quantization for TensorJoin, but exclude file I/O and process
  startup for all methods.

