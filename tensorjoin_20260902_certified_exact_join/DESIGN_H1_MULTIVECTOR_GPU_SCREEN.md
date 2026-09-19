# H1 Design Card: Exact GPU Multi-Vector Join Screen

## Decision question

Can H0's object-level certificate algebra become an exact resident-GPU
operator that beats exhaustive direct-FP64 evaluation under the identical
variable-cardinality symmetric-Chamfer threshold-join contract?

This is a screen, not a paper claim.  Correctness and safety gates precede
timing.  A speed result is usable only inside the frozen H1 denominator.

## Target and roofline hypothesis

- Host: `gpu-host-8` (`gpu-host-8`).
- Device: physical GPU 1, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  compute capability 12.0, 188 SMs, 96 GiB.
- Runtime observed before implementation: driver 590.48.01, CUDA 13.0,
  PyTorch 2.11.0+cu130, Triton 3.6.0.
- Hypothesis: exhaustive direct-FP64 token distances are compute-bound.  The
  candidate replaces the full FP64 scan with INT8 Tensor Core dot products,
  small object reductions, and FP64 work only for compacted ambiguous object
  panels.  The fixed host-visible ambiguity count is a charged serial cost.

## Semantic and numerical contract

- Input tokens are normalized in FP64 and rounded once to FP32, exactly as H0.
- Query/base objects and their deterministic split are identical to H0:
  128 query objects, 512 disjoint base objects, seed 20260903.
- Score is symmetric Chamfer over **squared** direct-FP64 token distances:
  one half of the mean row minimum plus one half of the mean column minimum.
- Thresholds are H0's immutable strict mid-gap thresholds for target results
  per query 1 and 8.
- Canonical output is the sorted uint64 object-pair ID `q * 512 + b`.
- Quantization is per token, symmetric INT8 in `[-127,127]`, with INT32 dot
  accumulation and analytic residual upper bounds.
- Token certificate output stores nonnegative L2 lower/upper bounds in FP32.
  Object aggregation converts them to FP64, squares monotonically, and applies
  an absolute outward square guard of `1e-10` before `min` and `mean`.
- Ambiguous object panels are recomputed from original FP32 tokens by direct
  FP64 differences.  No reconstructed-norm identity is used in the exact path.

## Shapes, layouts, and dispatch boundary

| Dataset | Query tokens | Base tokens | D | Tokens/object |
|---|---:|---:|---:|---:|
| ESC-50/PANNs | 640 | 2560 | 2048 | exactly 5 |
| UCF101/R3D-18 | 4--7/object | 4--7/object | 512 | variable 4--7 |

The exact query/base token counts are recovered from the immutable H0 split.
Flattened token matrices are row-major.  Base INT8 codes are additionally
stored transposed.  Object starts/counts are contiguous int32 arrays.  H1
supports at most eight tokens per object; any larger object is a hard refusal,
not truncation.

## Kernel and role table

| Phase | Grid/role | Work | Output |
|---|---|---|---|
| C0 token certificate | 2-D 64x64 token tiles, 4 warps | INT8 Tensor Core dot, analytic L2 interval | full token lower/upper matrices |
| C1 object decision | one program/object pair | padded 8x8 FP64 square, row/column `min`, directed means | direct IDs, ambiguous object IDs |
| C2 exact panel | one program/ambiguous valid token cell | direct FP64 difference/reduction over D | padded 8x8 exact panel |
| C3 exact object | one program/ambiguous object pair | row/column `min`, directed means | appended final IDs |
| K0 keeper token | one program/4x4 token tile | direct FP64 difference/reduction over D | full exact token matrix |
| K1 keeper object | one program/object pair | exact padded 8x8 reduction | canonical IDs |

No cluster or cross-CTA handoff is used.  Each output scalar has a single CTA
owner.  Atomics own only compacted output positions and counters.

## Storage and live-set ledger

| Phase | Persistent input | Large live object | Last use | Persistent output |
|---|---|---|---|---|
| C0 | INT8 codes, scales, reconstructed norms, residual bounds | 64x64 INT32 accumulator | token epilogue | two FP32 token matrices |
| C1 | interval matrices, object starts/counts | two padded 8x8 FP64 panels | object classification | IDs/counters |
| C2 | original FP32 tokens, ambiguous IDs | one BLOCK_K FP64 difference vector | per-cell reduction | ambiguous_count x 64 FP64 panel |
| C3 | exact panels, ambiguous IDs | one 8x8 FP64 panel | object classification | appended IDs/counter |
| K0 | original FP32 tokens | one 4x4xBLOCK_K FP64 difference tile | per-tile reduction | full FP64 token matrix |
| K1 | full exact matrix | one 8x8 FP64 panel | object classification | IDs/counter |

The candidate's padded exact-panel workspace is allocated for all 65,536
object pairs, so allocation is outside timing but the launched exact work uses
only the observed compacted count.  Invalid padding cells are explicitly set
to `+inf` and are charged when a panel is refined.

## Ready graph

1. Resident quantization metadata is ready before timing.
2. C0 completes before C1 reads interval matrices.
3. C1 completes before the host reads the ambiguity count; this synchronization
   is charged.
4. The count determines C2/C3 grids.  C2 completes before C3 reads exact panels.
5. C3 completes before final IDs are copied to the host; this synchronization
   and copy are charged.
6. K0 completes before K1; K1 completes before keeper IDs are copied.

No output buffer is overwritten before its final consumer.  Counters and ID
buffers are reset on-device at the start of every timed invocation.

## Useful-work, traffic, and control hypotheses

- Candidate invariant work: all token INT8 dots, all object reductions, two
  counter/ID copies per invocation.
- Candidate variable work: 64 padded FP64 token cells per ambiguous object.
- Keeper invariant work: every query-token/base-token direct-FP64 distance and
  every object reduction.
- Candidate added traffic: two full FP32 interval matrices plus padded exact
  panel output.  Keeper writes one full FP64 token matrix.
- Candidate added control: object compaction atomics and a host-visible dynamic
  count.  This is the principal launch/serialization risk.

## Keeper, comparator, and reject rule

- Keeper: H1 K0+K1 exhaustive direct-FP64 GPU implementation under the same
  score, threshold, final-ID, and resident-data scope.
- Independent oracle: H0 CPU direct-FP64 differences and object aggregation.
- Correctness rejection: any non-finite valid score, token/object containment
  violation, unsafe direct decision, counter overflow, duplicate ID, candidate
  mismatch, keeper mismatch, or candidate/keeper disagreement.
- Safety rejection: any Compute Sanitizer memcheck error on both keeper and
  candidate correctness modes.
- Screen performance pass: in every one of four dataset/threshold cells,
  candidate median latency must be lower than keeper median latency and the
  median speedup must be at least 1.25x.  This screen uses 10 warmups and 50
  timed observations per variant in alternating KC/CK order.  It does not
  establish a confidence interval or sustained production result.

## Verification ladder

1. CPU reference and H0 threshold/hash validation.
2. Kernel compile and adversarial ragged smoke on counts 4, 5, 6, and 7.
3. Full four-cell output equality and interval/direct-decision audit.
4. Compute Sanitizer memcheck for keeper and candidate correctness modes.
5. Only after 1--4 pass, alternating CUDA-event timing under the GPU1 lock.
6. Preserve raw samples, environment, source hashes, SASS/resource metadata,
   decision, and reopen conditions.  Formal timing is a separate H2 gate.
