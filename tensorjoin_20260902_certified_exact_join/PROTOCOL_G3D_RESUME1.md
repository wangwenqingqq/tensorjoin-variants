# Protocol G3D-Resume1: slotwise recovery from prelaunch GPU0 blockage

Date frozen: 2026-09-03

Parent experiment: `tensorjoin_20260903_g3d_complete_pipeline_timing`

Status: frozen after the operational blockage below and before any resumed
runner execution.

## Trigger and immutable evidence

The original G3D driver waited until physical GPU0 was empty, then launched the
frozen screen campaign.  Screen process 0 (`KC`, attempt 0) completed cleanly,
with exact pre/post outputs and no foreign-process overlap.  Before screen
process 1 launched, an unrelated `sglang::scheduler` process appeared during
the required 30-second quiescence check.  The orchestrator stopped before
starting the timing runner.  Preserve without modification:

- `results/g3d_screen_process_0_kc_a0.json` and its raw/preflight/occupancy
  evidence and Triton cache;
- `raw/g3d_screen_process_1_ck_a0_preflight.log`;
- `raw/g3d_screen_process_1_ck_a0_occupancy.jsonl`;
- the empty attempt-0 cache directory for screen process 1; and
- `raw/g3d_campaign_driver.log`, including the traceback.

The blocked screen attempt contains no timing sample and is excluded.  The
clean process-0 sample is retained and must not be rerun.

## Only admitted operational change

Run the remaining work one frozen process slot at a time after GPU0 again has
30 continuous empty seconds.  This changes only outer scheduling and manifest
recovery.  It does not change the runner, variants, kernels, data, launch
extents, timing denominator, order, warmups, retained observations, sustained
launches, estimator, or gates in `PROTOCOL_G3D.md`.

Frozen recovery sources are:

- unchanged measurement runner `src/run_g3d_complete_pipeline_timing.py`,
  SHA-256
  `e6518efc901d6c5ab316fe8f8f4d95e37ce4de535e08ac9f353acd8e46e1fd3a`;
- `src/run_g3d_resumed_slot.py`, SHA-256
  `3fabc11881c13501c5b61d9414014f9e4b56d872ce25e79a4931da367d797398`;
- `src/finalize_g3d_resumed_campaign.py`, SHA-256
  `4d7ade07e600ab19c44399f39723ed8edbdf64b847aec111b98138fc27a22684`.

Screen process 1 resumes as `CK`, attempt 1.  After the two admitted screen
slots exist, the recovery finalizer creates the parent screen manifest, while
explicitly retaining the prelaunch-blocked attempt.  The runtime-cubin audit
and frozen screen summarizer then run unchanged.

If the screen passes, each of the eight formal slots may likewise run
individually in the original order schedule.  The formal finalizer is invoked
only after exactly one clean admitted record exists for every slot.  Each
launched contaminated attempt is retained; no slot may exceed the parent's
three-attempt limit.  A foreign process discovered before runner launch is
recorded as a prelaunch blockage and contains no admissible measurement.

## Safety and claim boundary

Every resumed slot still holds the project GPU0 lock, repeats the original
30-second quiescence check, monitors ownership every 0.25 seconds, kills only
its own process group on overlap, and requires empty postflight compute state.
No foreign process is terminated.  All acceptance thresholds and the claim
boundary remain exactly those in `PROTOCOL_G3D.md`.
