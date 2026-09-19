# Design G3B: Separate GPU Analytic-Certificate Candidate

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_gpu_analytic_certificate_g3b`

## Target and hypothesis

- Hardware: RTX PRO 6000 Blackwell, physical GPU0, `sm_120`.
- Shape: frozen G2A self-join, `4096x4096x512`, float32 source.
- Hypothesis: the G3A analytic reconstruction bound can be realized around the
  existing exact INT8-to-INT32 MMA without increasing upper-triangle ambiguity
  above 127,846 pairs.
- Status boundary: correctness/soundness candidate only. No G3B timing is a
  performance claim.

The accepted G2B/G2A2 sources are immutable keepers. G3B is a separate file and
does not enter active dispatch.

## Numeric contract

- Codes: signed INT8 in `[-127,127]`.
- Dense accumulation: INT32. The dispatcher rejects any shape with
  `D*127*127 > min(2^24, INT32_MAX)`.
- Scale: per-vector float32, interpreted as the exact stored value.
- Code norm: exact integer sum, converted through float64 on the host and stored
  as float32 for the GPU expression.
- Residual norm: computed from exact-widened float32 source, codes, and scales;
  the float64 squared-norm reduction is inflated by
  `1/(1-gamma_(2D+2))`, square-rooted, then rounded upward to float32.
- GPU reconstructed-distance expression radius:
  `2^-16 * term_abs_sum + 16*float32_min_normal`. This is twice G3A's
  already-conservative relative radius to cover evaluation of the radius itself
  from stored float32 terms.
- GPU square-root/residual/final-add margin:
  `2^-20 * max(root_hi + residual_i + residual_j, 1) +
  16*float32_min_normal`. This exceeds the PTX-documented `2^-23` maximum
  relative error for `sqrt.approx.f32` and covers the remaining float32
  additions under the frozen finite-normal domain.
- Threshold: two outward float32 encodings of the exact float64 epsilon. Direct
  accept compares with the lower encoding; direct reject compares with the
  upper encoding.
- Every unresolved pair is refined directly in float64 in G3B. The existing
  FP32 refinement stage is intentionally excluded so its proof obligation
  cannot be hidden inside the INT8-certificate result.

## Layout, roles, and schedule

| Unit | Ownership and role |
|---|---|
| Host | Quantize, derive upward residuals, build upper-tile schedule, validate IDs |
| One Triton program | Own one scheduled 64x64 upper tile and its INT8 MMA/certificate decisions |
| Four warps | Cooperatively execute compiler-lowered INT8 dot and scalar interval arithmetic |
| FP64 refinement program | Own one ambiguous pair and append it only if the frozen predicate passes |

- Source/codes are row-major; the second code operand is materialized
  K-major/transposed exactly as in G2A2.
- Tile geometry is 64x64x64. The 2,080 upper tiles form one proof-bounded batch
  with capacity `2080*4096`; diagonal element masking enforces `i<=j`.
- No manual shared-memory swizzle or barrier is introduced; those details remain
  compiler-owned and must be inspected in generated code rather than assumed.

## Phase live set

| Phase | Large live objects | Last use |
|---|---|---|
| Preprocess | source, codes, scales, code norms, residual bounds | host temporaries die after device copies |
| Certificate | code matrices, scales, norms, residuals, tile coordinates, direct/ambiguous buffers | ambiguous IDs survive to FP64 |
| FP64 refine | source, ambiguous IDs, accepted buffer, counters | all device buffers survive through validation copies |
| Validate | host direct/ambiguous/final IDs plus frozen oracle | released after JSON result |

Expected register peak is compiler-dependent because the 64x64 MMA accumulator
and interval terms overlap. Compilation/SASS must report registers, stack/local
memory, and the exact integer-MMA and square-root lowering before promotion.

## Ready graph

```text
host upward bounds ready
-> device copies + upper schedule ready
-> analytic certificate complete
-> direct/ambiguous counters and IDs ready
-> FP64 refinement of exactly ambiguity_count
-> final accepted IDs ready
-> host canonical expansion and oracle validation
```

No buffer is reused before its consumer completes. G3B is synchronous by design
to isolate numeric correctness from overlap.

## Work and movement delta versus G2A2

- Invariant: one INT8 MMA per upper pair, same source/codes/tile schedule,
  identical final predicate and canonical output.
- Removed: empirical constant pad and the FP32 refinement stage.
- Added per dense pair: absolute-term sum, analytic expression radius, two
  square roots, outward residual/final margin, and lower/upper epsilon compares.
- Added refinement: all ambiguity goes to direct FP64, so this is intentionally
  not a latency candidate yet.
- Movement is unchanged except that residual bounds are recomputed by the new
  host routine.

The expected benefit is a checkable numerical contract, not speed. A later
G3C may prove and restore FP32 routing only after G3B passes.

