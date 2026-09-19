# Protocol C0B: Compact-and-Refine End-to-End Candidate

Experiment ID: `tensorjoin_20260902_triton_compact_refine_c0b`

## Frozen scope

- One shape and radius only: ESC-50 `M=512, N=4096, K=1024`, midpoint radius
  producing 1 exact result/query.
- Inputs and preallocated output buffers are resident. Feature extraction,
  quantization, allocation, and host-to-device transfer are excluded.
- Keeper: unchanged guarded-FP32 exact-output pipeline from B0Y.
- Candidate output: a GPU-resident compact INT32 pair-ID array plus a result
  count. Pair order is unspecified; the set must equal the FP64 oracle.

## Candidate stages

1. Zero three GPU-resident INT32 counters.
2. A C0A-derived Triton INT8 MMA kernel computes the padded certificate and
   atomically appends direct-accept IDs to the result buffer and ambiguous IDs
   to a separate buffer. It writes no dense status or score matrix.
3. Read the ambiguous count to launch an exact-size Triton refinement grid.
4. Each refinement program recomputes one ambiguous pair directly from the
   original FP32 vectors promoted to FP64, then atomically appends an accepted
   pair ID to the result buffer.

Buffers hold 8,192 IDs. Any overflow rejects the run. C0A's `1e-4` bound pad,
`64x64x64` scan tile, four warps, and three stages remain unchanged.

## Correctness and safety

- Exact equality of sorted result IDs to the NumPy FP64 oracle.
- Zero direct false accepts, zero missing oracle IDs, zero duplicates, and zero
  buffer overflow.
- Extracted SASS must show native INT8 MMA in the scan and FP64 arithmetic in
  refinement; neither kernel may store a dense score/status matrix.
- `compute-sanitizer --tool memcheck` must exit normally with zero errors.
- A 1,000-launch stress run must preserve counts and output hashes.

## Paired performance contract

- GPU 0 lock and per-process idle check as in B0Y.
- CUDA-event resident-input end-to-end timing, including counter reset, scan,
  ambiguous-count synchronization, exact refinement, and final compact output.
- 20 warmups, 100 retained observations, eight fresh processes with
  `A/B, B/A` repeated four times. A is guarded FP32 and B is C0B.
- Primary estimator: geometric mean of per-process median keeper/candidate
  ratios; deterministic 20,000-process-bootstrap 95% interval; report all raw
  samples, marginal p10/median/p90, process wins, and order split.
- Pass only if correctness/safety gates pass, the 95% lower speedup bound is
  >=1.5x, and at least 7/8 processes win.

Failure rejects the current performance thesis. A pass admits broader shapes,
modalities, and external baselines; it is not by itself a paper-level result.

