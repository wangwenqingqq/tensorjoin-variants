# Protocol G4A: deployable dynamic-count router attribution

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g4a_dynamic_count_router`

Status: frozen before any G4A GPU execution.

## Purpose and falsifiable hypothesis

G3D established a 2.670x complete resident-GPU advantage for the certified
three-stage cascade, but used stage launch extents learned from earlier
validation.  G4A removes that deployment shortcut.  Each invocation reads the
actual device counters, synchronizes as required, and launches the next stage
with the observed count.

Hypothesis: after charging this host/device orchestration, G3C-B-R1 remains at
least 1.25x faster than G3B-R1 under the formal paired estimator.  Failure stops
the breadth implementation: fixed-count G3D remains valid attribution, but the
current host-dispatched router is not promoted.

## Frozen sources

- `src/run_g4a_dynamic_router_timing.py`, SHA-256
  `fe7aaa92c7556e11b2a08e62815ce5189ff0ab677a16706f272ec097b20ba843`;
- `src/run_g4a_isolated.py`, SHA-256
  `170392c1bf03d94927b6354553991636fa8c67a12a572cba27c7d75c2de579ad`;
- `src/audit_g4a_runtime_artifacts.py`, SHA-256
  `1bbc1d33cdf8cc32e4005997b3c5b964f33262e5e899883a196e9805091289be`;
- `src/summarize_g4a.py`, SHA-256
  `cda82673385d9a17b3a3439c585c095b521a55c616574f34c85582b4897a8ca7`.

The imported G3B-R1, G3C-B-R1, and FP64-refinement sources and accepted cubin
hashes remain those frozen in G3D.  G4A adds no kernel or numerical change.

## Workload and variants

- Live host: `gpu-host-8`, hostname `gpu-host-8`.
- Hardware: physical GPU0, RTX PRO 6000 Blackwell Server Edition, natural
  dynamic clocks and unchanged power policy.
- Workload: frozen public-source G2A CIFAR-10-GIST self-join, `4096x512`
  contiguous float32, identical radius and upper-triangle schedule.
- Output: exact 133,120 sorted upper-triangle uint64 IDs, SHA-256
  `036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959`.
- Keeper: G3B-R1; copy its actual ambiguity count to the host; launch FP64 for
  that count; copy the final count to the host.
- Candidate: identical G3B-R1; copy actual ambiguity count; launch certified
  G3C-B-R1 FP32; copy actual residual-FP64 count; launch FP64 for that count;
  copy final count.

Expected counts are checked, not supplied to the launch path: keeper
`102079 -> 102079 -> 133120`; candidate `102079 -> 97 -> 133120` for
G3B ambiguity, FP64 work, and final upper output.

## Timing denominator

Use host `perf_counter_ns`.  Timing begins before the device counter reset and
ends after the final count has been copied to the host.  It includes:

- counter reset;
- every G3B/G3C/FP64 GPU kernel in the variant;
- actual scalar D2H count copies;
- synchronization caused by dynamic count discovery;
- Python/Triton launch overhead; and
- final GPU completion and final-count availability.

Inputs, quantization metadata, output buffers, and pair-ID buffers are already
resident.  Input ingest/preprocessing and final pair-ID D2H copy/sort remain
excluded.  G4A is therefore a host-dispatched dynamic-count resident operator,
not the public end-to-end denominator.

Every process performs full exact-output validation before and after timing.
Every retained observation records the observed dynamic stage counts.  The
sustained path must expose exactly one count tuple per variant.

## Cheap screen

Run two fresh processes in orders `KC` and `CK`.  Each variant receives ten
warmups, 50 retained host-wall observations, and a separate 200-invocation
sequential host-dispatched run.

The screen passes only if:

1. all pre/post outputs and every recorded dynamic count are exact;
2. all fresh runtime cubins match the accepted G3B/G3C/FP64 hashes;
3. each process-median keeper/candidate ratio is at least 1.20x; and
4. each sustained keeper/candidate ratio is at least 1.20x.

Failure is retained and stops G4A before formal timing.  No timing setting,
threshold, or implementation may change after seeing the screen.

## Formal campaign

Only a passing screen admits eight fresh processes in orders
`KC, CK, KC, CK, CK, KC, CK, KC`.  Each variant receives 20 warmups, 200
retained host-wall observations, and a separate 1,000-invocation sequential
dynamic run.

The primary statistic is the geometric mean of eight process-median ratios.
Use a deterministic 20,000-sample process bootstrap with seed `20260903`.
Formal acceptance requires exact output/counts and accepted cubins in all
processes, 8/8 process-median wins, a 95% bootstrap lower bound of at least
1.25x, 8/8 sustained wins, and sustained geometric-mean speedup of at least
1.20x.  Retain p10/median/p90, raw order, process ratios, and order-position
diagnostics without substituting them for the paired estimator.

## Safety and claim boundary

Each process holds `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`,
requires 30 continuous empty seconds on physical GPU0, monitors ownership every
0.25 seconds, terminates only its own process group on overlap, and requires an
empty postflight compute state.  Up to three attempts per slot are retained;
prelaunch blockage is not a timing observation.

The G3B/G3C kernels already passed generated-code, full memcheck, adversarial,
and 1,000-cascade safety gates.  G4A's new risk is host orchestration; its
pre/post exact checks and 1,000 sequential dynamic invocations test that path.
Passing G4A admits only a shape-local dynamic-count attribution claim.  It does
not establish input/output-inclusive end-to-end performance, full Cifar60K,
other datasets/shapes, or a device-resident asynchronous scheduler.
