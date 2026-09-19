# G2 precision/fusion design card

SM120 warp-synchronous IMMA/HMMA; no clusters, TMEM, tcgen05 or async-TC claims.
INT8 native m16n8k32/s32; FP16 m16n8k16/f32. Both compute64x64x512 tiles in
128threads, BK64,3stages. Only upper-triangle valid IDs are classified; dot
work includes bounded tile padding. Same938x938 triangular tile inventory,
108host batches. Same original FP32/FP64 per-pair kernels and canonical output.

| Phase | Owner/live state | Layout / ready and release |
|---|---|---|
| Preparation | one original metadata CTA/vector for INT8; one admitted FP16 metadata CTA/vector | Original input H2D; q+qt/s/h/e or half+4metadata; immutable during call |
| Dot |4warps,32accumulator values/thread before epilogue | Shared operand buffers and sync-MMA fragments; same loop within each dtype; kill operand references after last MMA |
| Split edge | only S8/S16 | Each CTA owns4096scores at pid*4096; same stream kernel completion before predicate consumer; reusable64MiB buffer |
| Predicate | fused accumulator owner or matched split consumer CTA | Original INT8 expression or unchanged directed FP16 interval; metadata loaded after dot, interval dies before compaction |
| Compaction | original-style per-pair atomic counters, same for both dtypes | Shared six counters, output/ambiguous global arrays; capacity fail-closed; no cross-block handoff |
| Refinement | original128-thread per-pair CTA | Blocking host count read then stage2, count read then terminal; identical for all4methods |
| Output | host | Batch IDs D2H, mirror off-diagonal, canonical sort; within full timer |

Prediction: S8/S16 add8bytes per computed dot cell and one kernel/batch;
F8/F16 remove that global score traffic at the cost of a larger live epilogue
and possible lowered occupancy. Useful dot counts and final output stay fixed.
The local live-set hypothesis is ~32accumulator GPRs plus address/operand and
interval temporaries; no-spill admission is empirical, not guessed occupancy.
FP16 may need more MMA instructions but fewer numerical refinement CTAs; this
tradeoff is explicitly measured rather than inferred from peak throughput.

No source-level register cap or dtype-specific autotuning. Resource/ABI/SASS
must be read from actual frozen artifacts. Full/tail ID addressing retains
60000-row range masks and64bit global IDs. Grid is one CTA per supplied tile,
not arbitrary N/D dispatch. Score storage is4bytes for both precisions so the
split-memory perturbation is comparable. Shader/host scheduling differences
between Gate1 C and this matrix are not mislabeled as an isolated fusion effect.
