# G15 decision: the complete FP32-first control wins

Last verified on gpu-host-8 GPU 2, 2026-09-05. The fixed six-process public
screen is complete with all full-output and occupancy gates passing. This is
an early same-contract screen, not formal confidence or sustained evidence.

## Decisive same-contract evidence

CIFAR-GIST stored FP32 60000x512, D2 threshold 0.3955230712890625. Pageable
host source to all sorted directed host uint64 IDs, including self. All
methods return 3,926,078 IDs with SHA-256
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
Every preparation, transfer, allocation, count read, compaction, terminal,
mirror and host sort is charged; startup/JIT and validation are not.

| Fixed block | Original G5 (s) | CPU-chunked G5 (s) | Pedantic FP32-first (s) | Original/chunked | FP32/chunked |
|---|---:|---:|---:|---:|---:|
| 0: original, FP32, chunked | 0.918468967 | 0.745645422 | 0.308740553 | 1.231777 | 0.414058 |
| 1: chunked, FP32, original | 0.959831809 | 0.755950435 | 0.307452931 | 1.269702 | 0.406710 |

The CPU integration gate (>=1.15x in both blocks) passes. The TC-specific
screen (FP32/chunked >=1.25x in both) **fails decisively**. The stronger control
is about 2.42–2.46x faster than even the CPU-improved TC path. Prior MiSTIC
advantages do not establish TC necessity. Do not use additional datasets or
renaming to rescue this failed implementation-level performance claim.

## What is actually validated

- CPU blockwise construction: four fresh CPU processes, all five arrays
  bitwise identical; prep 0.757/0.766 s versus 0.596/0.578 s. Full integration
  preserves G5 kernels and work counts.
- Complete FP32-first: seven independent small real/adversarial fixtures,
  rectangular FFI mapping and conservative interval checks; full public hash.
- Pedantic mode queried as 2, compute type 69, TF32 override 0. Actual NSYS
  full-operator dispatch has 120 public SGEMMs plus three warmups. NCU exports
  all three full/tail shapes; selected normalized SASS hash is
  `fbc00ffde2c9eda01bdbbc0b3800a71f5a636128e3e00cb641efc7f8a33df7e3`,
  with 2,424 static instructions, 768 FFMA and zero MMA/TF32.
- Full memcheck device-access and fixture synccheck logs report zero errors;
  1,000 alternating-input/pointer-churn complete N129 calls preserve outputs.
  This is not leak-free shutdown, arbitrary streams/Graphs or sustained60K.
- Reference equivalence is tested against the frozen FP64 terminal. G10's
  exact-real counterexample remains valid; no universal exact-real claim.

## Decisions, separated

1. **Implementation:** retain CPU blocking as an attributed local improvement;
   reject the CPU-prepared G5 path as superior to the complete strong control.
2. **Mechanism:** the measured CPU-preparation cost dominates this implementation.
   This does not prove a lower bound against TC computation with a different
   data path; missing GPU-preparation evidence is uncertainty, not rejection.
3. **Thesis:** generic certification/quantize-refine novelty is unresolved and
   nearest prior-art overlap remains. Moving preparation to GPU is engineering,
   not by itself a non-incremental mechanism or a novelty pass.

## Bounded next step and reopen condition

G16 tests GPU construction of the existing metadata, keeping G5 stages fixed,
and grants the FP32 control the same GPU-preparation opportunity. Require the
same host-output predicate/hash, correctness/safety/precision gates and reversed
process ordering. Only a decisive same-contract win admits a narrow candidate;
it still does not admit formal multi-dataset work or paper promotion until Gate0
and modern external comparator gaps are resolved. Do not drop the G15 control.

## Durable evidence

`results/{cpu_campaign,correctness_gates,precision_collection,precision_audit,public_screen}.json`,
`results/public_b*_p*.json`, source/design freezes in `artifacts/`, and referenced
root `raw/g5_g15*.log`, occupancy receipts and per-process fresh compiler caches.
Raw failed transport and first SASS-normalizer attempts are retained in
`ATTEMPTS.md`; corrected branch-address normalization changes no kernel or time.
