# Original GPU2 confirmation: passed, no new speed claim

Completed2026-09-08 CST. **All nine original-card continuation steps and the
whole-suite isolation guard pass with unchanged numeric source and captured
cubin identities.** This discharges the pending original-card qualification
from `../fp16_gpu_control_20260907`; it does not amend that earlier checkpoint
or turn its excluded attempts into passes.

## Result

| Step | Result and scope |
|---|---|
| Runtime ABI | Nine actual captured cubins; parameter offsets/types and resource checks pass |
| Fixture | Two distinct input allocations;131,328 upper pairs each;796 outputs,43 stage1-ambiguous and40 FP64 inputs |
| Public panels |22,798,128 upper pairs checked against the unchanged FP64 terminal;29,122 accepted,1,042 ambiguous,21 FP64 inputs |
| Full60K |3,926,078 directed output pairs; exact canonical output hash matches |
| Stress |32 alternating complete bounded composites; original and wide-exponent/cancellation inputs |
| Graph |16 immutable-input consumer replays; classifier/counters/stage2/terminal, not a general GEMM/dynamic-input Graph |
| Four sanitizers | Memcheck/synccheck/initcheck/racecheck all clean on the bounded two-input512-row composite |
| Frozen identities | Numeric source, nine captured cubins, four original G16 cubins and library identities verified; no changed predicate coefficients |
| Isolation | Original GPU2 UUID16f27f5a-dfcd-48e0-bb39-bebbe4009245; suite guard passes; peak device-used2256MiB |

Full output SHA256:
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
Full stage counts:86,029 stage1-ambiguous,1,734 FP64 inputs,
1,993,039 upper output pairs. They match the secondary-card qualification.
Raw per-call seconds are diagnostic only and must not form an A/C speed ratio.

Two bounded NVML exit tombstones were logged for previously ancestry-verified
child PIDs578833 and579157; neither was an unknown or live foreign process.
The R1/R2 guard rule, process start ticks, logs, quiescence, locks and memory
limit are preserved. No foreign process was terminated or reconfigured.

## Contribution and next decision

A conventional FP16 control reproduces the same tiny terminal queue on the
matched panels, with substantially fewer first-stage ambiguities than INT8.
**The queue itself is therefore not an INT8-specific contribution.** This is
counterevidence about exclusivity, not a performance or novelty verdict.

The control is now eligible for a separately frozen, direction-balanced,
complete host-to-host A/C experiment under the tested fixed-reference workload
and explicit conditional dot model. The comparison must count preprocessing,
score generation/materialization, interval/queue work, both refinements, host
control/transfers and canonical output. Remeasure both methods; keep ordinary
and repeated-call order splits. Do not divide by old runs or NCU durations.
No such timing campaign was run in this checkpoint.

Material limits remain unchanged: one60K dataset/threshold; bounded sanitizer
coverage; immutable consumer Graph; fixed-FP64 reference rather than exact-real
membership; no universal proof of the cuBLAS SM120 dot envelope. The derived
interval is conditional on that envelope. This is control qualification, not
novelty Gate0 admission, a new numerical theorem, or a paper-ready speed claim.

## Evidence links

- `results/original_confirmation_a0.json`: nine child outcomes.
- `results/original_confirmation_a0_guard.json`: whole-suite isolation and
  executed source hashes; do not require nonexistent per-child guard files.
- `results/gpu2_*`, `raw/gpu2_*`: actual original-card checks.
- `artifacts/original_confirmation_freeze.json`: pre-run exact code/binary freeze.
- `ORIGIN.md`: predecessor manifest hash and copy/cache provenance.
- `../fp16_gpu_control_20260907/NUMERICAL_SCOPE.md`: equations, sources and limits.
- `../fp16_gpu_control_20260907/results/selected_trace_audit_r1.json`: actual
  original-GPU2 selected cuBLAS code for the four shapes.

Implementation: qualified on original GPU2 at the stated test scope.
Mechanism: retained conventional FP16 control, conditional numerical model.
Thesis: exclusivity claim narrowed; superiority and novelty remain unestablished.
Manuscript/Overleaf unchanged; no GPU jobs intentionally left running.
