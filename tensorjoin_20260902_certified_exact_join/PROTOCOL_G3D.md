# Protocol G3D: complete resident-GPU cascade attribution

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g3d_complete_pipeline_timing`

Status: frozen before any G3D GPU execution.

## Question and prior gates

G3D asks whether the accepted G3C-B-R1 certified FP32 middle stage improves the
complete resident-GPU exact pipeline over the accepted G3B-R1 two-stage
pipeline on the frozen G2A workload.  G3B-R1 and G3C-B-R1 correctness,
generated-code, memcheck, adversarial, and 1,000-launch stress gates have
already passed.  G3D changes orchestration only and does not tune a numerical
constant, kernel, tile, threshold, or data contract.

Frozen orchestration sources are:

- `src/run_g3d_complete_pipeline_timing.py`, SHA-256
  `e6518efc901d6c5ab316fe8f8f4d95e37ce4de535e08ac9f353acd8e46e1fd3a`;
- `src/run_g3d_campaign.py`, SHA-256
  `8a856bff2ef31990826d32f9bfa1836f152e395a258dd8394a74a4c7438f5a63`;
- `src/audit_g3d_runtime_artifacts.py`, SHA-256
  `7447a889aef5dad8b28f632262e340acfde9eb788aeff329bc31853244821816`;
- `src/summarize_g3d.py`, SHA-256
  `0eec0e19ebd6f0bcdb248c1d1b9fdb6911afdea58e350e607bf0b9bfd99c5a2d`.

## Frozen comparison contract

- Host: live `gpu-host-8` container hostname `gpu-host-8`.
- GPU: physical GPU 0, NVIDIA RTX PRO 6000 Blackwell Server Edition, natural
  dynamic clocks; no clock or power setting is changed.
- Input: frozen contiguous CIFAR-10-GIST float32 vectors, shape `4096x512`,
  vector SHA-256
  `e6a720399067e9753bc6f45ff6f67b73a3861964255e85e1ed9a1ce7906df462`.
- Quality target: exact sorted upper-triangle uint64 pair IDs, 133,120 pairs,
  SHA-256
  `036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959`.
- Keeper: unchanged G3B-R1 analytic certificate followed by FP64 refinement of
  all 102,079 accepted G3B-R1 ambiguous pairs.
- Candidate: the identical G3B-R1 certificate, followed by the accepted
  G3C-B-R1 certified FP32 filter, followed by FP64 refinement of its 97
  accepted residual pairs.
- Both variants have separate output, ambiguity, and counter buffers.  Inputs
  and quantization metadata are shared read-only.

The fixed second- and third-stage launch extents come from two accepted,
hash-identical G3C-B-R1 validation processes.  They avoid host synchronization
inside the measured range and are legal only for this shape-local attribution
experiment.  A later deployable operator must discover these counts without
assuming them.

## Timing denominator

The primary denominator is one complete resident-GPU pipeline invocation.
CUDA events begin before the device counter reset and end after every GPU
kernel in the variant.  Thus the keeper includes counter reset, G3B-R1, and
FP64 refinement; the candidate includes counter reset, identical G3B-R1,
certified FP32 filtering, and residual FP64 refinement.

Excluded are host input ingest, quantization preprocessing, host discovery of
dynamic stage counts, device-to-host result copy, and host sorting.  Therefore
G3D is not ingest-inclusive or output-materializing end-to-end latency and must
not be reported as such.

Each process validates exact counts and the final upper-pair hash immediately
before and after timing.  Fresh Triton caches must contain exactly the accepted
runtime cubins:

- G3B-R1 certificate:
  `dd32e97936d8e0c6abd3924d8fdeeb17088e7a6e1e4d8ac2f1fc492bbc2114ac`;
- G3C-B-R1 FP32 filter:
  `db750295875f8363679cfe9f9762cef884e1ecd371e3e419fd500d91c3a20670`;
- FP64 refinement:
  `ff0572a7eefcb6ccba91c7da0020d370e45d7f8f7c03d3a55e8f5ca53c477c50`.

## Cheap performance kill test

Run two fresh processes with block orders `KC` and `CK`, where `K` is the
keeper and `C` is the candidate.  Each variant receives five warmups, 20
retained per-invocation CUDA-event observations, and a separate batch of 100
back-to-back complete pipeline invocations.

The screen passes only if both processes remain exact, both process-median
keeper/candidate ratios are at least 1.10x, both sustained-batch ratios are at
least 1.05x, and every timed cubin matches the accepted hashes.  Failure stops
G3D before formal timing and is preserved as negative evidence.  No constant,
launch extent, tile, warmup, sample count, or gate may be changed after seeing
the screen.

## Formal paired campaign

Only a passing screen admits eight fresh processes.  Frozen block orders are
`KC, CK, KC, CK, CK, KC, CK, KC`.  Each variant receives 20 warmups, 100
retained per-invocation CUDA-event observations, and a separate sustained batch
of 1,000 back-to-back complete pipeline invocations.

For each process, compute the ratio of the two variant medians.  The primary
estimator is the geometric mean of the eight process ratios with a
20,000-sample process bootstrap, seed `20260903`.  Formal acceptance requires:

1. exact pre/post output and accepted runtime cubins in all eight processes;
2. the candidate wins all eight process-median comparisons;
3. the 95% bootstrap lower bound is at least 1.15x;
4. the candidate wins all eight sustained comparisons; and
5. the geometric mean sustained speedup is at least 1.10x.

Raw observations, p10/median/p90, process ratios, direction, order-position
splits, and sustained batch totals are retained.  Marginal pooling and
order-position statistics are diagnostics and do not replace the primary
paired estimator.

## Isolation, contamination, and claim boundary

Every process uses physical GPU 0, the lock
`@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`, 30 seconds of empty
quiescence, 0.25-second occupancy monitoring, and an empty postflight compute
state.  A foreign process terminates only the owned experiment process group;
the foreign process is never terminated.  At most three contamination attempts
are allowed per slot, and every contaminated attempt is retained and excluded.

A formal pass supports only a shape-local mechanism-attribution claim on the
frozen GPU/compiler artifacts: the certified FP32 stage reduces complete
resident-GPU pipeline time relative to refining all G3B-R1 ambiguity in FP64.
It does not establish public end-to-end speedup, arbitrary shapes or inputs,
another GPU/toolchain, a dynamically scheduled deployable operator, or
submission readiness.
