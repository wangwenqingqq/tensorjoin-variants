# F1 Partial Decision

Date: 2026-09-03

Historical checkpoint only. Superseded by `DECISION_F1.md` after the remaining
clean formal processes completed and the global gate passed.

## Conclusion

F1 has passed the three-modality smoke gate, all declared safety gates, and the
complete eight-process audio formal gate. This is strong evidence that the
added FP32 routing stage is a real high-output execution mechanism rather than
modality packaging. The global F1 decision is nevertheless **open**, because
clean formal timing is still missing for video processes 3--7 and HSI
processes 0--7.

## Strongest evidence

| Gate | Audio | Video | HSI | Decision |
|---|---:|---:|---:|---|
| Smoke `two_stage/cascade` | 1.559x | 1.674x | 1.898x | all pass >=1.15x |
| Smoke `guarded_fp32/cascade` | 2.428x | 1.952x | 2.184x | all pass modality gates |
| FP64 pairs after FP32 routing | 366 | 1,061 | 928 | exact F0 counts |
| Memcheck | 0 errors | 0 errors | 0 errors | all pass |
| Stress | 1 hash / 1,000 | 1 hash / 1,000 | 1 hash / 1,000 | all pass |
| Formal process coverage | 8/8 | 3/8 clean | 0/8 | incomplete |

For audio, the eight-process geometric-mean speedups are 1.588x over the
unchanged two-stage candidate (95% interval [1.580x, 1.596x]) and 2.439x over
guarded FP32 ([2.433x, 2.445x]), with exact output and 8/8 wins for both
comparisons.

## Material caveat

Video process 3 was contaminated by an unrelated GPU0 process and is excluded
by the frozen protocol. The resume attempt stopped before launch because GPU0
was occupied again. This is an infrastructure interruption, not evidence for
or against the mechanism, and it cannot be repaired by using the contaminated
sample, moving to another GPU, or silently reducing the eight-process gate.

## Research interpretation

The result strengthens only the middle strategy: an exact, output-sensitive
mixed-precision vector-join operator. It does not pass the broad novelty gate,
establish an external-system advantage, or justify a paper claim across all
modalities. Modality breadth remains evaluation evidence rather than the
contribution.

## Next admissible action

When physical GPU0 is continuously idle, run exactly one video process-3
replacement in order `GCT`, then video processes 4--7 and HSI processes 0--7
under the existing lock and occupancy checks. Do not rerun audio or clean video
processes 0--2. If the complete F1 gate passes, the next Gate-0 test is a
same-contract external comparison and larger-scale/full-denominator campaign,
not another modality or tree sweep.

## Evidence

- `results/f1_audio_formal_partial_summary.json`
- `results/f1_{audio,video,hsi}_smoke.json`
- `results/f1_{audio,video,hsi}_{memcheck,stress1000,sass_validation}.json`
- `artifacts/f1_formal.sass`
- `receipts/f1_formal_*`
- `raw/f1_video_process_3_contamination.log`
- `raw/f1_formal_resume1_blocked.log`
