# Claim-evidence ledger

Experiment: tensorjoin_20260907_fp16_gpu_control_d512.
Last live validation:2026-09-08 CST, gpu-host-8/gpu-host-8.
Evidence stages and target boundaries below are separate from universal proof.

| ID | Claim / denominator | State | Primary evidence | Counterevidence / restriction | Allowed wording |
|---|---|---|---|---|---|
| C1 | Owned FP16 GemmEx uses FP32-accumulating HMMA for the four frozen shapes | measured | selected_trace_audit_r1.json; four admitted GPU2 reports; container and normalized function hashes | Failed shape3_a1 retained; repeat3_a2 passes; mnemonics do not specify a numerical theorem | Runtime trace observes HMMA.16816.F32 in the selected kernels |
| C2 | Same complete60K output | measured | gpu0_full_a2.json and guard; source/runtime binary hashes | GPU0 only; one dataset/threshold; fixed FP64 reference, not exact-real | The GPU0 control returns the same3,926,078 canonical directed IDs |
| C3 | Safe direct decisions on all selected public pairs | measured | gpu0_public_a2.json and saved IDs |22,798,128 selected upper pairs, not all possible inputs | No wrong direct accept/reject on the three tested panels |
| C4 | Directed FP32 pair classifier rather than FP64-per-pair handicap | measured | nine actual cubins/PTX/SASS; compiled_static_audit; gpu0_abi_a0 | Preprocessing still performs FP64 per vector; no speed inference | Actual classifier variants have directed FP32 arithmetic, no FP64 arithmetic and no local spills |
| C5 | Tiny FP64 queue is not specific to INT8 | inferred | matched-panel prior census and offline_review count comparison | Same21 FP64 input count does not alone prove identical work order or latency | A conventional FP16 filter reproduces the small terminal queue in the tested panels |
| C6 | Boundary, wide-exponent, stress and consumer Graph checks pass | measured | gpu0_stress_a0; saved samples; offline_review |512-row bounded fixtures; Graph uses immutable precomputed scores and calibrated counts |32 alternating runs and16 consumer Graph replays pass on GPU0 |
| C7 | Bounded sanitizer gates clean | measured | gpu0_{memcheck,synccheck,initcheck,racecheck}_a0 logs, results, guards | No full60K sanitizer or leak-check-full claim | Four sanitizer tools report zero errors/hazards on the bounded composite |
| C8 | Universal cuBLAS FP16 dot error envelope on SM120 | unknown | Explicit premise and derivation in NUMERICAL_SCOPE; finite checks | Not guaranteed merely by cuBLAS mode or HMMA; architecture-specific prior analysis does not cover this selected implementation universally | Conditional interval assuming the stated library dot envelope |
| C9 | Original GPU2 composite requalification | unknown | original_confirmation_a0_guard.json, child PIDnull | Foreign PID571421 prevented launch; GPU0 does not discharge this frozen gate | Original-card confirmation remains pending |
| C10 | A is faster than this C in full host-to-host execution | unknown | No evidence; no timing protocol or paired runs | Old pedantic-FP32 ratio and new diagnostic seconds are incompatible denominators | No A/C speed claim |
| C11 | This implementation supplies a new paper contribution | unknown | No novelty evidence added here | Low-precision filtering/refinement overlaps prior work; engineering a control is not novelty | Comparator qualification and counterevidence, not a new thesis |

Implementation decision: GPU0 correctness/static/ABI/safety-qualified, original-
card and timing promotion pending. Mechanism decision: preserve conditional
FP16 comparator; do not reject it because final timing is missing. Thesis impact:
remove exclusivity wording around the small FP64 queue; no new novelty admission.
