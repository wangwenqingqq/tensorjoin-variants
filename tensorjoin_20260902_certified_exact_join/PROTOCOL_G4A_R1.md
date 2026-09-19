# Protocol G4A-R1: dynamic-count router attribution on physical GPU1

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g4a_r1_dynamic_count_router_gpu1`

Status: frozen before any G4A-R1 GPU execution.

## Revision boundary

The original G4A screen passed on physical GPU0, but three subsequent formal
campaign attempts were interrupted by unrelated GPU0 workloads.  The first two
partial campaigns were excluded before summarization and preserved under
`rejected/`.  In the third attempt, the isolation monitor terminated only the
TensorJoin process when a foreign SGLang scheduler acquired GPU0.  No
contaminated observation is admissible.

G4A-R1 changes only the physical device and isolation namespace:

- from physical GPU0 UUID
  `GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1`;
- to physical GPU1 UUID
  `GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3`;
- from lock `.tensorjoin_gpu0_campaign.lock` to
  `.tensorjoin_gpu1_campaign.lock`; and
- from artifact prefix `g4a_` to `g4a_r1_`.

GPU1 is the same RTX PRO 6000 Blackwell Server Edition model, with the same
driver, toolkit, natural dynamic-clock policy, code, workload, and measurement
contract.  The original GPU0 screen is not reused: G4A-R1 must pass a new GPU1
screen and a complete eight-process GPU1 formal campaign.

## Purpose and falsifiable hypothesis

G3D established a 2.670x complete resident-GPU advantage for the certified
three-stage cascade, but used stage launch extents learned from earlier
validation.  G4A-R1 removes that deployment shortcut.  Each invocation reads
the actual device counters, synchronizes as required, and launches the next
stage with the observed count.

Hypothesis: after charging this host/device orchestration, G3C-B-R1 remains at
least 1.25x faster than G3B-R1 under the formal paired estimator.  Failure stops
the breadth implementation: fixed-count G3D remains valid attribution, but the
current host-dispatched router is not promoted.

## Frozen sources

- `src/run_g4a_r1_dynamic_router_timing_gpu1.py`, SHA-256
  `f06064fb891d310e4119aa125fa808319cf50159c6364cbc6d4fe51013b26db6`;
- `src/run_g4a_r1_isolated_gpu1.py`, SHA-256
  `83928a70866a34d7657649bb106cd9f5b3a4eba8a15ec65a6ea6cdd870785540`;
- `src/audit_g4a_r1_runtime_artifacts_gpu1.py`, SHA-256
  `9ba08eb65961ca6334119df24e3447f2b3e9913056714dbc7460a1e169fee0a5`;
- `src/summarize_g4a_r1_gpu1.py`, SHA-256
  `acdf8fdff8ee6e62ba1ff56e4579b3ba748f68465bc74b6c73326dceafee9b82`.

The imported G3B-R1, G3C-B-R1, and FP64-refinement sources and accepted cubin
hashes remain those frozen in G3D.  G4A-R1 adds no kernel or numerical change.

## Workload and variants

- Live host: `gpu-host-8`, hostname `gpu-host-8`.
- Hardware: physical GPU1, UUID
  `GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3`, RTX PRO 6000 Blackwell Server
  Edition, natural dynamic clocks and unchanged power policy.
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
`102079 -> 102079 -> 133120`; candidate `102079 -> 97 -> 133120` for G3B
ambiguity, FP64 work, and final upper output.

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
excluded.  G4A-R1 is therefore a host-dispatched dynamic-count resident
operator, not the public end-to-end denominator.

Every process performs full exact-output validation before and after timing.
Every retained observation records the observed dynamic stage counts.  The
sustained path must expose exactly one count tuple per variant.

## Cheap screen

Run two fresh GPU1 processes in orders `KC` and `CK`.  Each variant receives ten
warmups, 50 retained host-wall observations, and a separate 200-invocation
sequential host-dispatched run.

The screen passes only if:

1. all pre/post outputs and every recorded dynamic count are exact;
2. all fresh runtime cubins match the accepted G3B/G3C/FP64 hashes;
3. each process-median keeper/candidate ratio is at least 1.20x; and
4. each sustained keeper/candidate ratio is at least 1.20x.

Failure is retained and stops G4A-R1 before formal timing.  No timing setting,
threshold, or implementation may change after seeing the screen.

## Formal campaign

Only a passing G4A-R1 screen admits eight fresh GPU1 processes in orders
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

Each process holds `@TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock`,
requires 30 continuous empty seconds on physical GPU1, monitors ownership every
0.25 seconds, terminates only its own process group on overlap, and requires an
empty postflight compute state.  Up to three attempts per slot are retained;
prelaunch blockage is not a timing observation.

The G3B/G3C kernels already passed generated-code, full memcheck, adversarial,
and 1,000-cascade safety gates.  G4A-R1's new device-migration risk is covered by
fresh exact pre/post validation, runtime cubin identity, and 1,000 sequential
dynamic invocations in every admitted process.

Passing G4A-R1 admits only a shape-local dynamic-count attribution claim on a
same-model GPU.  It does not establish input/output-inclusive end-to-end
performance, full CIFAR60K, other datasets/shapes, or a device-resident
asynchronous scheduler.
