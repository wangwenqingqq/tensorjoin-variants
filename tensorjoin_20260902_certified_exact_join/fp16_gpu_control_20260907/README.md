# FP16 dense GPU comparator: functional qualification checkpoint

Last checked: 2026-09-08 CST. **The FP16 dense control is implemented and passes
full-output, numerical-sample, compiled-code, ABI, stress, Graph-consumer, and
bounded sanitizer checks on same-host GPU0. Original GPU2 library traces pass,
but original-card composite confirmation is not complete. No A/C performance
comparison or new paper contribution is established.**

## Evidence that changed

| Gate | Observed result | Exact scope |
|---|---|---|
| Actual library execution | Four shapes, three selected kernels, HMMA.16816.F32 | Owned FP16 GemmEx, original8p GPU2; full containers and normalized selected SASS retained |
| Public-panel outputs | 22,798,128 upper pairs;29,122 accepted; exact terminal agreement | Three frozen panels on GPU0; rejected candidates included in the oracle check |
| Full output | 3,926,078 directed pairs; canonical hash matches | N60000,D512,T25921/65536 on GPU0; diagnostic single call, not timing admission |
| GPU FP16 routing | 1,042 stage1-ambiguous,21 FP64 inputs | Same three panels; matches previous CPU FP16 routing counts |
| Original INT8 routing | 21,810 stage1-ambiguous,21 FP64 inputs | Retained matched-panel census, not re-timed here |
| Static compiled audit | Nine programs; no local loads/stores, no spills, no FTZ | Eight classifier variants plus metadata; classifier has no FP64 arithmetic |
| Runtime ABI | Nine captured cubins pass offsets/types/resource checks | GPU0 CUDA Driver queries, exact captured binaries |
| Stress |32 alternating runs, two distinct vector allocations | Original fixture and fixed-seed wide-exponent/cancellation input |
| Graph |16 stress replays pass | Immutable precomputed scores/metadata; classifier/counters/stage2/terminal only |
| Sanitizers | Memcheck, synccheck, initcheck, racecheck all clean | Bounded two-input512-row composite; not full60K sanitizer coverage or leak-check-full |
| Offline independent replay |2,560 exact dot samples,161 exact metadata rows,1,920 exact interval samples,5 panel-output audits pass | Saved artifacts, local CPU, no GPU timing |

Full output SHA256:
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
Full GPU FP16 stage counts are86,029 ambiguous and1,734 FP64 inputs, with
1,993,039 upper-triangle output pairs. These are counts, not speed evidence.

## Contribution/evidence decision

**The small FP64 queue is not an INT8-specific differentiator.** The matched
FP16 control reproduces the same21 FP64 inputs while forwarding far fewer
stage1 pairs. This strengthens the prior counterevidence; do not rescue novelty
through the comparator's implementation work or a small future speedup.

What remains testable is the complete cost of reaching the same fixed-reference
answer: preprocessing, matrix scores, interval classification, compaction,
refinement, host dispatch/transfers, and canonical output. The FP16 comparator
now uses fused GPU metadata and directed FP32 pair arithmetic, not a deliberately
slow per-pair FP64 classifier. The previous1.518533x result remains scoped only
to its pedantic FP32 comparator; it says nothing quantitative about this FP16
control. The old5.083x claim is not promoted.

## Material limitations and next gate

1. The selected-cuBLAS dot envelope is an explicit **conditional assumption**.
   SASS and finite tests do not prove its universal validity on SM120. See
   `NUMERICAL_SCOPE.md`; reference equivalence is not exact-real semantics.
2. GPU2 repeatedly received another user's work. The original-card confirmation
   attempt stopped in quiescence before starting its child. GPU0 evidence is
   not silently substituted for the frozen original-card gate.
3. No fair A/C timing campaign has been frozen or run. Diagnostic seconds in raw
   JSON and NCU durations must not be used to form a speed ratio.
4. Novelty Gate0 remains unresolved/overlapped by prior work. This checkpoint
   qualifies a control, not a new thesis or paper-ready numerical theorem.

Next: obtain a genuinely available original GPU2 window, rerun the frozen
composite confirmation without changing binaries, then freeze direction-balanced
complete host-to-host A/C measurements. Keep the conditional model and exact
workload scope in any resulting claim. Do not change the manuscript first.

## Artifacts

- `PROTOCOL.md`, `DESIGN.md`: preimplementation contract and design.
- `NUMERICAL_SCOPE.md`: explicit formula, assumption, sources, and limitations.
- `CLAIM_EVIDENCE.md`, `FAILURE_LEDGER.md`: allowed claims and retained failures.
- `results/selected_trace_audit_r1.json`, `compiled_static_audit.json`,
  `gpu0_abi_a0.json`, `offline_review.json`: instruction/ABI/independent evidence.
- `results/gpu0_*`: complete validation and isolation records.
- `artifacts/*freeze*.json`: source/build identity checkpoints.
- `delivery_manifest.json`: closed local/remote delivery inventory, excluding
  itself and post-seal verification sidecars.

No prior sealed evidence, original four G16 cubins, manuscript, or Overleaf
project was modified.119 previously sealed files were rehashed unchanged.
