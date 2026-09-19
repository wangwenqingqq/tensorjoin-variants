# Protocol F1: High-Output GPU Precision Cascade

Experiment ID: `tensorjoin_20260903_gpu_cascade_f1`

## Purpose

F1 is a shape-local implementation test admitted by the favorable target-64
rows in F0. It asks whether the fixed FP32 ambiguity filter can rescue the
measured high-output regressions without reopening F0's rejected universal
cascade claim.

## Frozen workload

- Reuse the exact D1, D2, and D3 feature caches, deterministic splits, and
  tie-aware nominal-64 midpoint radii.
- Shapes are `512x4096x2048` audio, `512x4096x512` video, and
  `512x4096x1984` HSI.
- Inputs and prebuilt quantization metadata are resident. Timing begins before
  output-counter reset and ends after the final refinement launch completes.
- GPU: physical GPU 0 on `gpu-host-8`, isolated by
  `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`; abort if occupied.

## Variants

1. `guarded_fp32`: unchanged FP32 matrix scan, `1e-3` guard, FP64 boundary
   refinement.
2. `two_stage`: unchanged fused INT8 certificate followed by FP64 verification
   of every INT8-ambiguous pair.
3. `cascade`: same fused INT8 scan, direct FP32 filter of all ambiguous pairs,
   then FP64 verification of only the fixed guard band.

All variants return the same complete pair-ID set. The F1 code may add only the
FP32 filter, its compact FP64-ID buffer/counter, and required orchestration; it
must not change Stage 1, radii, guards, tiles, or capacity.

## Cheap admission test

- One fresh process per modality, five warmups and twenty retained CUDA-event
  observations per variant.
- Record raw samples and medians; this is diagnostic smoke, not a public timing
  result.
- Correctness gate: zero false FP32 direct accepts/rejects, final mismatches,
  duplicates, or overflows for both `two_stage` and `cascade`.
- Work gate: measured FP64-ID counts must equal F0's 366/1,061/928 for
  audio/video/HSI.
- Performance gate, all required:
  - `two_stage_median / cascade_median >= 1.15x` in every modality;
  - `guarded_fp32_median / cascade_median >= 1.50x` for audio;
  - `guarded_fp32_median / cascade_median >= 1.35x` for video and HSI.

If any cell fails, stop F1 before sanitizer, stress, SASS, or formal timing and
retain the negative result. Do not tune the guard or specialize the refinement
tile after seeing smoke. A pass admits, but does not replace, the full safety
and eight-process campaign.

## Safety and formal promotion after smoke

The smoke passed before this section was activated. The exact smoke source is
retained in `artifacts/f1_smoke_source_058cc566.py`; subsequent source changes
may add validation/stress orchestration only and must not alter a kernel,
numerical threshold, tile, variant, or timing function.

For every modality:

1. run `compute-sanitizer --tool memcheck` in validation-only mode and require
   normal exit plus `ERROR SUMMARY: 0 errors`;
2. run 1,000 cascade launches and require a single final-ID hash, exact oracle
   equality, and zero overflow;
3. retain generated SASS proving the inherited INT8 scan contains IMMA and the
   residual refinement contains FP64 arithmetic; record source/cubin hashes;
4. run eight fresh processes with 20 warmups and 100 retained CUDA-event
   observations per variant. Process orders are frozen as
   `GTC, CTG, TCG, GCT, CGT, TGC, GTC, CTG`.

For each process, compute the ratio of variant medians. The decision estimator
is the geometric mean across eight process ratios with a deterministic
20,000-sample process bootstrap (`seed=20260903`). Every process must favor the
cascade. The 95% lower bound must be at least 1.15x for `two_stage/cascade`, at
least 1.50x for audio `guarded_fp32/cascade`, and at least 1.35x for video and
HSI `guarded_fp32/cascade`. Report marginal medians and order-position splits
as diagnostics without substituting them for the predeclared estimator.

A formal pass remains a resident-input, single-GPU, three-shape result. It does
not establish ingest-inclusive, multi-GPU, sustained-service, or external-SOTA
performance.

## Contamination handling activated during formal run

Video process 3 (`GCT`) began in the same second as an unrelated user process
on physical GPU0. The post-process occupancy check detected PID 2,237,400 and
aborted the campaign. Preserve `results/f1_video_process_3.json` and its raw
log as contaminated negative-control evidence, exclude it from every estimator,
and run exactly one same-order replacement as
`results/f1_video_process_3_replacement.json` after GPU0 is empty. Continue
video processes 4--7 and HSI 0--7 without rerunning the already clean audio
0--7 or video 0--2 samples. No other replacement is allowed unless the same
explicit occupancy/failure condition is recorded.
