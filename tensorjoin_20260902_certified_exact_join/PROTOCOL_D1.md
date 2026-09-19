# Protocol D1: Learned-Audio End-to-End Breadth Gate

Experiment ID: `tensorjoin_20260902_panns_triton_d1`

## Frozen scope

- Feature cache: D0's deterministic ESC-50 PANNs Cnn14 2048-D embeddings,
  SHA-256 recorded in each process result.
- Query/base split: D0's seed-`20260902` clip-disjoint `M=512, N=4096` split.
- Two radii: for nominal targets of 1 and 64 results/query, find the distance
  at the requested global rank and the next strictly greater distance, then use
  their FP64 midpoint. This tie-aware mid-gap rule retains every tie while
  avoiding cross-device ambiguity from comparing at an observed distance.
  Every run must report the two endpoints, gap, and actual output count; no tie
  is dropped to force a requested cardinality.
- Inputs and preallocated outputs are GPU-resident. Feature extraction,
  quantization, allocation, host-to-device transfer, and the FP64 oracle are
  excluded from timing.

## Variants

- Keeper: exhaustive non-TF32 FP32 GEMM, a fixed `1e-3` distance-squared guard,
  direct FP64 refinement of the boundary band, and compact exact pair IDs.
- Candidate: fused per-vector INT8 Tensor-Core scan, padded three-way
  certificate, in-kernel atomic compaction, and direct FP64 refinement of only
  ambiguous pairs.
- Candidate buffers hold 65,536 pair IDs. The scan tile remains `64x64x64`;
  D=2048 is consumed by the K loop. Refinement consumes K in 256-element
  blocks and accumulates in FP64.
- Candidate timing includes counter reset, scan, ambiguous-count host
  synchronization, exact refinement, and the final resident compact output.

## Correctness and safety

- Oracle distances use blocked direct FP64 differences, not the cancellation-
  prone norm expansion.
- Sorted candidate and keeper pair IDs must exactly equal the oracle.
- Zero unsafe direct accepts, duplicates, and output/ambiguity overflows.
- `compute-sanitizer --tool memcheck` must exit normally with zero errors for
  both radii.
- A 1,000-launch test at each radius must preserve counts and output hashes.
- Extracted SASS must retain native INT8 MMA in scan and FP64 arithmetic in
  refinement, with compact outputs rather than a dense score/status matrix.

## Paired timing

- Physical GPU0 only, with the campaign lock and a fresh idle check before
  every process. No clock or power changes.
- For each radius: 20 warmups, 100 retained CUDA-event observations, eight
  fresh processes with `AB, BA` repeated four times.
- Primary estimator: geometric mean of per-process keeper/candidate median
  ratios. Report deterministic 20,000-process-bootstrap 95% intervals,
  marginal p10/median/p90, process wins, and AB/BA split geomeans.

## Pass/stop rule

Both radii must pass all correctness and safety gates, have a bootstrap 95%
lower speedup bound of at least 1.5x, and win at least 7/8 processes. A failure
is negative breadth evidence and stops promotion of the learned-audio
performance claim. A pass admits video embeddings and an indexed/external-
baseline stage; it is not by itself paper-level evidence.
