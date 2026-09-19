# G3D complete resident-GPU pipeline timing decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_g3d_complete_pipeline_timing`

## Decision

**PASS for shape-local mechanism attribution.  On the frozen `4096x512` G2A
contract, the exact G3C-B-R1 three-stage pipeline is 2.670x faster than the
unchanged exact G3B-R1 two-stage pipeline by the predeclared paired estimator
(95% process-bootstrap interval [2.658x, 2.678x], 8/8 wins).  This is a
complete resident-GPU pipeline result, not ingest-inclusive or
output-materializing end-to-end latency.**

The separate 1,000-invocation continuous batches also favor the candidate in
8/8 processes, with a 2.868x geometric-mean speedup.  Exact outputs and the
accepted runtime cubins are preserved in every process.

## Frozen comparison

Both variants use the same frozen CIFAR-10-GIST `4096x512` float32 input, same
radius, same upper-triangle schedule, same G3B-R1 certificate kernel, same
FP64 refinement kernel, same output capacity, and the same exact 133,120-pair
upper-triangle output.

- Keeper: G3B-R1 followed by FP64 refinement of all 102,079 G3B ambiguity
  pairs.
- Candidate: identical G3B-R1 followed by the certified G3C-B-R1 FP32 stage,
  then FP64 refinement of its 97 residual pairs.

The second- and third-stage launch extents are the exact counts established by
two earlier G3C-B-R1 validation processes.  They are fixed to avoid a host
synchronization inside this attribution denominator.  This is appropriate for
isolating the value of the middle stage, but it is not a deployable dynamic
router.

## Timing result

Eight fresh processes use block orders `KC, CK, KC, CK, CK, KC, CK, KC`, 20
warmups and 100 retained CUDA-event observations per variant.  An event begins
before the device counter reset and ends after all variant kernels complete.

| Metric | G3B-R1 keeper | G3C-B-R1 candidate |
|---|---:|---:|
| Marginal p10 | 753.917 us | 282.944 us |
| Marginal median | 763.088 us | 285.408 us |
| Marginal p90 | 767.942 us | 290.947 us |

The decision statistic is not the marginal ratio.  Per-process median ratios
range from 2.633x to 2.683x.  Their geometric mean is 2.669585x; a deterministic
20,000-sample process bootstrap with seed `20260903` gives
[2.658105x, 2.677861x].  All eight processes favor the candidate, clearing the
frozen 1.15x lower-bound gate.

The order split does not reverse the result:

- `KC`: four-process geometric mean 2.6785x;
- `CK`: four-process geometric mean 2.6607x.

## Sustained result

Every formal process separately enqueues 1,000 back-to-back complete pipeline
invocations per variant on one CUDA stream.  The per-process speedups range
from 2.854x to 2.871x; the geometric mean is 2.867997x and the candidate wins
8/8.  This passes the frozen 1.10x sustained geometric-mean gate.

The sustained result is GPU execution throughput for the declared resident
contract.  It is not a sustained multi-client service measurement.

## Correctness and generated-code identity

Immediately before and after timing, all eight processes reproduce:

- keeper counts `[133120, 102079, 0, 0, 0, 0]`;
- candidate counts `[133120, 102079, 0, 97, 59859, 0]`;
- 133,120 exact upper-triangle pair IDs;
- upper-pair SHA-256
  `036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959`.

The formal runtime audit checks every fresh process cache and finds exactly one
copy of each accepted cubin, with no extra Triton cubin:

- G3B-R1 certificate `dd32e979...114ac`;
- G3C-B-R1 FP32 filter `db750295...20670`;
- FP64 refinement `ff0572a7...477c50`.

Compilation identity, correctness, prior memcheck/stress, per-invocation
timing, and sustained timing remain separate evidence gates; none is
substituted for another.

## Isolation and preserved operational evidence

All admitted processes ran on physical GPU0, UUID
`GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1`, under the project lock after 30
seconds of empty quiescence.  The external 0.25-second monitor observed no
foreign process during any admitted run, and every admitted process left an
empty postflight compute state.  GPU clocks and power policy were not changed.

The original screen campaign completed clean process 0, then stopped before
screen process 1 launched when an unrelated `sglang::scheduler` appeared
during quiescence.  That blockage, empty attempt cache, logs, and traceback are
retained.  `PROTOCOL_G3D_RESUME1.md` changes only outer scheduling: it keeps the
clean process, excludes the prelaunch blockage, and runs the missing slot after
GPU0 is empty.  No measurement setting or gate changed after the first result.

## Claim boundary and next gate

Allowed wording:

**On the frozen `4096x512` G2A resident-GPU contract, adding the certified FP32
middle stage reduces FP64 refinement from 102,079 to 97 pairs and makes the
complete exact GPU pipeline 2.670x faster than refining all first-stage
ambiguity in FP64 (95% process-bootstrap interval [2.658x, 2.678x], 8/8
wins).**

Not allowed:

- ingest-inclusive, host-output-materializing, or public-system end-to-end
  wording for G3D;
- a claim that the current hard-coded launch extents implement a dynamic
  runtime router;
- generalization to full Cifar60K, other dimensions, output densities,
  datasets, GPUs, or compiler artifacts;
- claiming that G3D retroactively proves the separately timed G2B kernel; or
- submission readiness.

The next paper-critical gate is not another modality port or another
micro-kernel tweak.  It is a compact public breadth matrix with a deployable
dynamic-count path across dimensions, output densities, and scales, retaining
the same complete-output contract.  This must show where the router selects
INT8/FP32/FP64 work and whether the shape-local G3D benefit survives its real
orchestration costs.

## Evidence

- Formal summary: `results/g3d_formal_summary.json`, SHA-256
  `2778c76ead1350fb9aa87c27754ed74a115cfa922ed3de4e7e50db8783dbf6de`.
- Formal manifest: `results/g3d_formal_campaign_manifest.json`, SHA-256
  `adfa81b3ebac9743e293747385693fa9ef5814ab4ba277b042533ad1ce1d2c1f`.
- Formal runtime audit: `results/g3d_formal_generated_code_audit.json`,
  SHA-256
  `84f6d9aaa3791f8c5abf700cbd75f2e87edbc50024d98654ec36f85913eec147`.
- Screen summary: `results/g3d_screen_summary.json`, SHA-256
  `0c5bee5127da981de9247f09a90986bfed6047a4c18622276bd73d7b47e49d76`.
- Protocols: `PROTOCOL_G3D.md`, SHA-256
  `6839b2fb656815c623cd46db0d3a435791c90c128a1b88b5c4ef926bc014cd1b`,
  and `PROTOCOL_G3D_RESUME1.md`, SHA-256
  `c0ecae17ee10293980cb43c6a905129f74074e1b94c0bcced01fce5165e468ae`.
- Measurement runner: `src/run_g3d_complete_pipeline_timing.py`, SHA-256
  `e6518efc901d6c5ab316fe8f8f4d95e37ce4de535e08ac9f353acd8e46e1fd3a`.
- Formal driver log: `raw/g3d_formal_resumed_driver.log`, SHA-256
  `2d9e165ae333a0cb03debf0d57ea80bc7a9b05e67d8d93e0f189d56d69b6e59a`.
