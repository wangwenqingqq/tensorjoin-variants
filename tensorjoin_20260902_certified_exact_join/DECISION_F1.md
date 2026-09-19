# F1 Final Decision

Date: 2026-09-03

## Conclusion

F1 is **accepted in its frozen scope**. The fixed INT8-to-FP32-to-FP64 GPU
cascade preserves exact radius-join output and beats both the unchanged
INT8-to-FP64 two-stage path and guarded FP32 on all three target-64 workloads.
All correctness, safety, stability, SASS, per-process-win, and bootstrap
lower-bound gates pass.

## Formal result

| Modality | Shape | Two-stage / cascade | 95% interval | Guarded FP32 / cascade | 95% interval | Wins |
|---|---:|---:|---:|---:|---:|---:|
| PANNs audio | `512x4096x2048` | 1.588x | [1.580x, 1.596x] | 2.439x | [2.433x, 2.445x] | 8/8, 8/8 |
| R3D-18 video | `512x4096x512` | 1.690x | [1.677x, 1.701x] | 1.959x | [1.948x, 1.971x] | 8/8, 8/8 |
| HSI patches | `512x4096x1984` | 1.909x | [1.901x, 1.917x] | 2.210x | [2.203x, 2.216x] | 8/8, 8/8 |

The corresponding cascade marginal medians are 163.296, 150.976, and 191.264
microseconds. FP64 refinement counts are exactly 366, 1,061, and 928, matching
F0. All admitted processes match the direct-difference FP64 oracle.

## Safety and mechanism evidence

- Every modality reports `ERROR SUMMARY: 0 errors` under memcheck.
- Every 1,000-launch stress run retains one exact output hash.
- Formal SASS contains 48 IMMA instructions and FP64 refinement arithmetic.
- Promotion source SHA-256 is `19fd8341...32a8e6`; SASS SHA-256 is
  `b95b2729...b8cfd`; final-summary SHA-256 is `c3aa7f42...0507b`.
- The original contaminated video process 3 is retained but excluded. The sole
  protocol-allowed replacement ran after a 30-second empty-GPU quiescence
  check and completed without overlap.

## Claim boundary

This result establishes a resident-input, exhaustive-scan, single-SM120-GPU,
three-shape target-64 mechanism result. It does not establish ingest-inclusive,
sustained-service, multi-GPU, larger-scale, adaptive-router, or external-SOTA
performance. It strengthens the middle extension strategy; it does not repair
the rejected broad “first exact low-precision search” novelty claim.

## Next Gate-0 test

Do not add another modality or tune a tree. Freeze a larger-scale/full-
denominator contract and compare against the strongest buildable exact GPU
brute-force and specialized similarity-join baselines. The direction remains
paper-admissible only if the same mechanism gives a decisive reproducible
advantage and a router/cost model selects precision regimes without oracle
knowledge.

## Primary evidence

- `results/f1_summary.json`
- `results/f1_{audio,video,hsi}_process_*.json`
- `results/f1_video_process_3_replacement.json`
- `results/f1_{audio,video,hsi}_{memcheck,stress1000,sass_validation}.json`
- `raw/f1_formal_campaign.log`
- `raw/f1_formal_campaign_resume2.log`
- `artifacts/f1_formal.sass`
- `receipts/f1_formal_*`
