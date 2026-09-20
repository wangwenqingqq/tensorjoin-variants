# Lite selector: G0 stopped on a CPU-reference contract mismatch

**Status: G0 FAILED / INCOMPLETE; G1, G2 and G3 NOT RUN.** The prescribed stop
rule was honored. There are no selector timings, trained models, held-out
performance results, speedup claims or mechanism-rejection claims from this run.
The first mismatch is in the newly added CPU check, not evidence that E3M4 or
INT8 disagrees with the frozen GPU FP64 predicate.

## What was executed

The supplied v1.0 plan is frozen in [CONTRACT.yaml](CONTRACT.yaml), including its
SHA256. The keeper is `5386c33e2e34b5d4e4245fbb3f5099f76cb7df79` on the
`precision-format-routing` branch. Existing kernels, precision envelopes,
classifiers, refinement and the binary64 terminal remain byte-for-byte unchanged.
Only the new experiment adapter adds common timed host input validation,
capacity checking, after-timing memory/identity records and a completed-call
kernel identity registry. No block32 path or selector is introduced.

- Hardware: physical GPU 7 of the eight-GPU RTX PRO 6000 Blackwell server,
  SM120, driver 590.48.01, CUDA/ptxas 13.1.115/13.1. CPU threads: four.
- Python 3.12.3, NumPy 2.3.1, PyTorch 2.11.0+cu130, Triton 3.6.0. Scikit-learn
  was absent and was not installed; training was gated on G1.
- One cooperative guard covered the complete attempted G0/G1 pipeline. It
  stopped the pipeline on the G0 nonzero exit. A second guarded process ran
  only bounded mismatch diagnosis. No foreign process was stopped, no GPU reset
  occurred and clocks/power limits were unchanged. Shared-host CPU activity is
  recorded, not isolated.
- The CPU generator created all **96 manifest configurations / 24 parents**.
  Split sizes are 48 training, 16 validation and 32 test. Test inputs have
  hashes and thresholds, but **zero test GPU executions/timings** occurred.
- Native identity, anti-fallback values, producer rounding and N=97 metadata/
  interval containment passed. This remains conditional numerical evidence,
  not a universal error proof.
- The 14 inherited fixtures plus three N=33 zero/duplicate/signed/abs-one/
  underflow fixtures passed all three fixed paths: **51 candidate calls plus
  17 direct references**. The binary64-neighbor one-coordinate checks passed.
- Six new configurations completed direct-reference, independent CPU32-subset
  and all three fixed-path checks: the four `clustered_101` configurations and
  the two N=1024 configurations of `clustered_102`.
- The seventh configuration stopped in the CPU32-versus-direct-FP64 check,
  before its three complete fixed-format calls. The validate file contains
  49 completed records, including that configuration's full direct reference.
- The scheduled sanitizer/stress processes were not reached. G0 is not promoted
  merely because earlier probes and inherited fixtures passed.

## First failing boundary

Configuration: **clustered / seed 102 / N=4096 / target k=8**. The 32-row subset
uses the frozen PCG64(31337) indices. The mismatch concerns one unordered pair,
parent rows **3279 and 3727** (zero-based), or subset rows 25 and 28.

| Quantity | Value |
|---|---|
| Input SHA256 | `8c521f67c444a4f8d58d83d20c769032290962b1b7333135d9463b0d44801736` |
| Threshold decimal | `0.09083749390179904` |
| Threshold binary64 | `0x1.74120418f0f3ep-4` |
| Newly added NumPy CPU reference distance | `0x1.74120418f0f3ep-4` (accepts) |
| CPU emulation of the measured GPU reduction | `0x1.74120418f0f3fp-4` (rejects) |
| Difference | One binary64 ULP |
| CPU-reference directed IDs | 36 |
| GPU-reference directed IDs | 34 |
| Differing directed IDs | 828 and 921 |

The new check used `np.sum(delta*delta, axis=2)` as if every FP64 reduction order
implemented the same finite predicate. That assumption is false at this sampled
quantile boundary. Even splitting into two NumPy sums of 256 values does not
reproduce this compiled GPU reduction.

The recorded terminal PTX adds adjacent squares per thread, reduces 32 lanes by
XOR offsets 16/8/4/2/1, reduces four warp sums by XOR 2/1, and finally adds two
256-element chunks. [replay_failure.py](replay_failure.py) independently follows
that exact addition graph on the CPU; its complete 32-row ID array matches the
recorded GPU array. This is a candidate corrected CPU reference, **not an adopted
change to the stopped campaign and not a fresh GPU validation**. Re-derive it
if the compiler, layout or terminal implementation changes.

The exact rational squared distance is also above the threshold: their
positive difference is `4277 / 295147905179352825856`. Correct rounding once and
`math.fsum` of rounded squares both give the upper hexadecimal value above.
These diagnostics explain this pair; they do not replace the declared finite-GPU
predicate with an exact-real oracle for the experiment.

## Same-process differential diagnosis

The unchanged keeper and the new adapter each ran direct FP64, INT8, E3M4 and
FP16 three times on the identical failing subset: **24 calls**. All 24 return
34 identical directed IDs. Thus this counterexample does not distinguish the
new adapter from its keeper or E3M4 from either conventional fixed format.

This localizes the first failure to the CPU-reference boundary. It does not
establish that every remaining large configuration, sanitizer, stress condition
or model wrapper is correct. No favorable validation/compiler-initialization
latency is treated as a benchmark result.

The first failing call raised before writing its completed-call JSON record;
its failure is retained in the process summary and traceback. The explicit
differing arrays come from the subsequent independent diagnosis, not a dump
silently attributed to the first call. The diagnosis records 24 output hashes
but its compiled manifest contains only the final FP16 call's five kernels:
it is not a complete per-call binary map of all 24 trials. The main G0 runner
does record each completed call's observed kernel identities. Root-level
diagnosis/replay helper hashes are recorded separately from the ten-file runtime
source manifest.

## Gate ledger and missing deliverables

| Gate / deliverable | State | Reason |
|---|---|---|
| G0 identity + inherited regression | Measured pass, bounded scope | Raw probe and 17-fixture records |
| G0 all new configurations | Failed / incomplete | CPU-reference predicate mismatch at configuration 7 |
| G0 sanitizer / alternating stress | Not run | Pipeline already stopped |
| G1 free-oracle benefit | Not run | G0 did not pass |
| G2 DT1 / DT3 / RF32 | Not run | No valid G1 admission; no models trained |
| G3 held-out and timed CIFAR anchor | Not run | Model was never selected or frozen |
| Format crossover / selection-loss figures | Not produced | No eligible timing dataset; diagnostic times are not substitutes |

[model.json](model.json) is an explicit not-trained status record, not a usable
model. `summarize.py` implements the registered G1 estimator but intentionally
refuses to run without all four complete G1 process files. None exists here.

**Recommended next action:** correct only the independent CPU reference to the
frozen GPU operation order, verify that reference independently, and register a
new G0 attempt before proceeding. Keep the v1 failure intact. Do not change
input generation, thresholds, gamma, classifier, refinement, output semantics or
test split to bypass this failure. The current run does not answer whether the
lightweight selector has enough performance headroom.

## Evidence and reproduction

- [workloads.jsonl](workloads.jsonl): all 96 input/parent/generator hashes and
  exact decimal/hex thresholds. This is input construction, not held-out timing.
- `results/g0_probe_v1.*`, `g0_regression_v1.*`, `g0_validate_v1.*`: immutable
  completed records and the failed process summary, including actual cubin
  hashes, per-call order, CPU activity, initialization/diagnostic times and flags.
- [g0_failure_diagnosis_v1.json](results/g0_failure_diagnosis_v1.json): complete
  CPU/GPU subset ID lists, three-repeat A/B results and exact-rational diagnosis.
- [g0_failure_subset_v1.npy](results/g0_failure_subset_v1.npy): generated synthetic
  32 x 512 reproducer; no restricted real dataset is distributed.
- [failure_terminal.code.ptx](results/failure_terminal.code.ptx): measured terminal
  code with private file paths/debug sections removed; original PTX hash retained.
- [GPU_SUPERVISION.json](results/GPU_SUPERVISION.json) and
  [EVIDENCE_MANIFEST.json](results/EVIDENCE_MANIFEST.json): guard outcomes,
  numeric telemetry, source/evidence hashes and curation boundaries.

CPU-only checks from the repository root:

```sh
python lite_selector_20260920/replay_failure.py
python lite_selector_20260920/verify_evidence.py
python -m unittest discover -s lite_selector_20260920 -p test_cpu.py
```

The unchanged failing G0 runner and GPU diagnosis are retained for reproduction
on the stated environment with authorized source fixtures and an idle GPU under
the existing guard. Never run `bench` after this failed G0. The recorded
compiler-preparation/validation wall times include compilation and are not
eligible performance measurements.
