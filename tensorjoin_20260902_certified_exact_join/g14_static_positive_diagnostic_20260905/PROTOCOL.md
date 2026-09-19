# G14: Attribute the existing static-join positive evidence

Frozen before measurement: 2026-09-05. Diagnostic only; no novelty pass,
formal promotion, new kernel, paper edit, or replacement of G5 P4 evidence.

## Question and symmetric hypotheses

G2B previously completed eight clean rounds near 0.926 s. The audited G5
artifact completed two exact rounds in 1.179 s and 5.285 s with identical
generated kernels and work counts. The missing evidence is phase attribution.
Both a repeatable cost in G5 and a non-reproducing historical slowdown remain
possible. Positive evidence is retained; missing attribution is not a mechanism
failure. Generic mixed-precision certification/TC joins have prior art: these
measurements do not resolve Gate 0 or justify a new paper thesis.

## Frozen contract

- Host: `gpu-host-8`, SSH through `tiaoban`, expected hostname `gpu-host-8`.
- Root: `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`.
- GPU: physical 2, RTX PRO 6000 Blackwell Server Edition,
  UUID `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245`; driver 590.48.01.
  No clock/power changes. Record live clocks, power, temperature and occupancy.
- Runtime: `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`, Python 3.12.3,
  PyTorch 2.11.0+cu130, CUDA runtime 13.0, Triton 3.6.0, NumPy 2.3.1;
  installed nvcc 13.1.115. No rebuild of the external comparator.
- Source: contiguous pageable FP32 CIFAR-GIST-512, shape 60000 x 512.
  NPY SHA-256 `95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c`.
- Epsilon 0.62890625; squared threshold 0.3955230712890625.
- Output: all 3,926,078 directed IDs including self, sorted host uint64,
  `row * 60000 + column`; SHA-256
  `13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
  The admission is agreement with this frozen oracle, not a new all-input proof.
- Scope: source host array to sorted canonical host IDs. Include preprocessing,
  schedule, allocations, H2D, all routing/count waits, D2H, reconstruction and
  sorting. Exclude file IO, process/context/JIT startup, validation and receipts.
- G2B and G5 kernels, shapes (64 x 64 x 64 stage 1), tile order, 4096-tile
  batches, capacities, precision contracts and original synchronization/logging
  behavior remain unchanged. Diagnostic adapters are additive generated copies.
- MiSTIC is freshly measured via the unchanged G5 adapter and existing FP64
  executable. Record its binary hash. It is a scope control, not claimed to be
  the strongest current literature baseline.

## Bounded schedule and controls

Twelve independent processes, no replacements or optional favorable reruns:

| Block | CPU/memory placement | Order |
|---|---|---|
| 0 | inherited/default | G2B, MiSTIC, G5 |
| 1 | node 0 CPUs and strict node 0 memory | G2B, MiSTIC, G5 |
| 2 | node 0 CPUs and strict node 0 memory | G5, MiSTIC, G2B |
| 3 | inherited/default | G5, MiSTIC, G2B |

This ABBA placement schedule reduces, but does not eliminate, time confounding.
Binding means `numactl --cpunodebind=0 --membind=0` (CPUs 0-15,64-79).
Node 0 has GPU 2 affinity. Binding is not exclusive CPU or memory-bandwidth
isolation; a foreign GPU 1 job remains active and untouched. No causal claim
about NUMA can be made from binding alone. No thread-count overrides are added.

One nonblocking campaign lock plus the unchanged G5 GPU-2 process lock. Before
each child: 30 seconds of GPU quiescence. During execution: unchanged 250 ms
occupancy guard. Foreign GPU-2 occupancy or any exactness/source/identity failure
stops the campaign and retains the failed slot. Never kill a foreign process.
Record host load, affinity, NUMA placement/policy and resource usage. The existing
G5 guard uses legacy file labels and legacy protocol fields; this protocol and
the G14 manifest define the diagnostic contract, not the guard's old labels.

## Attribution and gates

Instrument host wall and process-CPU boundaries for preprocessing, schedule,
transfer/allocation, each route and blocking count read, accepted-ID readback,
bookkeeping/logging, and final canonicalization. Save every per-batch segment.
Pre-create CUDA event pairs outside the timer. Launch proxies leave kernel
arguments unchanged and measure stream spans surrounding kernel dispatch.
These spans can contain host issue gaps and are NOT profiler-isolated kernel
durations. Instrumentation is timed and may perturb execution; do not promote
instrumented public speedups as an uninstrumented production result.

Pass requires exact IDs, no overflow, clean occupancy, source identity and G5
selected cubin/PTX identity matching the retained P4 artifact, plus phase sums
closing to the measured denominator (absolute discrepancy <= 1 ms). Preserve
all observations including mismatches and slow results. The previous G5 P1-P3
safety evidence applies to unchanged kernels, not newly claimed safety of an
optimized implementation. This is attribution, not a new optimization gate.

Report all 12 public times, per-placement p10/median/p90 (only two observations
per method/placement), order splits, paired ratios and wins, no confidence or
population-stability claim from this small diagnostic. Stage deltas determine
whether a follow-up has a specific target. If the historical 5.285 s slow mode
does not recur, explicitly leave its root cause unresolved. No formal eight-round
promotion follows automatically, even if every new observation is positive.

## Stop and next action

Do not run other datasets, threshold sweeps, tree packaging or new stories.
The next implementation experiment needs a measured dominant stage and a
separate frozen correctness/performance gate. A paper still needs a genuinely
different contribution and the strongest same-contract control (including
adaptive FP32-first); broader packaging alone cannot rescue subsumed novelty.

Rollback: no originals are edited. Stop only the identified campaign/own child
processes; preserve the G14 directory, guard logs and all caches as evidence.
