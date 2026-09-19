# H1 Exact GPU Multi-Vector Screen: Execution Status

## Current conclusion

H1-R1 **passes** its frozen correctness, safety, and performance screen on
physical GPU3 of `gpu-host-8`.  The original GPU1 target remained occupied, so a
pre-run-only revision moved to an idle card of the identical model without
changing data, semantics, kernels, denominator, or gates. No foreign process
was stopped or modified.

## Completed work

- Frozen design: `DESIGN_H1_MULTIVECTOR_GPU_SCREEN.md`.
- Frozen protocol: `PROTOCOL_H1_MULTIVECTOR_GPU_SCREEN.md`.
- Execution card: `RUN_CARD_H1.md`.
- Candidate/keeper Triton kernels: `src/h1_multivector_kernels.py`.
- Correctness, bounded memcheck, source-drift, and timing runner:
  `src/run_h1_multivector_gpu_screen.py`.
- Immutable memcheck summarizer: `src/summarize_h1_memcheck.py`.
- A candidate compile smoke completed before the foreign-PID guard was added,
  but the same preflight already showed a foreign allocation.  It is explicitly
  excluded from evidence.
- CPU interpreter validation of the tiled direct-FP64 keeper kernel found a
  maximum absolute reduction-order difference of `5.684341886080802e-14` on a
  deterministic 5x7x64 smoke.  This is implementation diagnostics only, not
  target-GPU correctness evidence.

## Resolved resource blocker

At 2026-09-03 08:10--08:18 GMT, `gpu-host-8` had one external eight-GPU process
on every device.  Physical GPU1 held about 52.7 GiB under external PID 4046278.
`gpu-host-shared` simultaneously held about 90.6 GiB for vLLM/Python processes.

The strengthened original runner resolved physical GPU1's UUID and refused
before any experimental kernel launch if another compute PID exists. The refusal was
verified under the H1 advisory lock; evidence:

- `raw/h1_foreign_process_refusal.log`
- SHA-256 `bdb0839eb516a8820418c8acf1d32332c64b2501dc60f28932a085ff07b96245`

GPU3 and GPU4 later became idle; H1-R1 selected GPU3 under a separate lock and
repeated the foreign-PID check before every phase.  GPU3 was empty again after
the campaign.

## Result and next action

All four cells reproduce CPU/keeper IDs, four bounded memchecks report zero
errors, and median outer-wall speedups over exhaustive direct FP64 are
65.063x/37.477x for audio target 1/8 and 21.570x/13.707x for video target 1/8.
See `DECISION_H1_R1.md`.

Next run H2 against certified FP32/TF32 exact filter/refine baselines and larger
streamed object matrices.  Do not add a tree/index contribution until that
strong-baseline kill test passes.
