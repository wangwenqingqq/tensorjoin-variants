# G8 A0 decision: synthetic crossover, no real-data crossover

Verified 2026-09-04 on gpu-host-8 physical GPU1. **Do not advance this A0
layout-based routing story to a paper claim or an online router.** The frozen
real-data gate failed. The synthetic positive control passed, so the experiment
does detect a layout-dependent crossover when the requested work is clustered.

## Strongest evidence

All cases use N=4096, D=512 and exactly 102079 requested pairs. Each family
keeps the same logical pairs and vectors under every physical permutation.
The real family is the frozen G3B-R1 CIFAR-GIST uncertainty set. The synthetic
family is a clique-shaped requested-work control, NOT real uncertainty.

Ratio below is `T4x4 time / P16 time`, the reciprocal of the predeclared
process-paired estimator, aggregated geometrically across two fresh processes.
Above one means tiling is slower. This is a refinement COMPONENT denominator.

| Family / layout | Active 4x4 tiles | Padded work / requested work | T4x4 / P16 | Winner |
|---|---:|---:|---:|---|
| Real / original | 87479 | 13.71 | 8.62 | P16 |
| Real / seeded random | 87890 | 13.78 | 8.66 | P16 |
| Real / uncertainty-degree ordering | 60135 | 9.43 | 5.93 | P16 |
| Real / PCA1 ordering | 72029 | 11.29 | 7.11 | P16 |
| Synthetic / clustered | 6500 | 1.02 | 0.776 | T4x4 (1.289x faster) |
| Synthetic / scattered | 75477 | 11.83 | 7.63 | P16 |

Both independent processes agree on every winner; the separate 500-call
sustained diagnostics retain every sign. There were 40 rounds per workload
per process, three method observations per round, and 20 calls per observation.
All observations, reverse order, and failed setup attempts are retained.

The actual uncertainty set remains scattered: even the degree-informed
reordering computes 9.43 pairs per requested pair when using full 4x4 tiles.
This reordering uses the completed uncertainty graph, but it is only an
oracle-informed HEURISTIC, not a globally optimal layout or an upper bound on
every possible packing algorithm. No layout search was performed after timing.

## Correctness and artifact identity

- All four real layouts and all three methods return exactly 59859 reject,
  42123 accept, and 97 uncertain states; logical state hashes match.
- Synthetic layouts/methods return 99941 reject, 2138 accept, zero uncertain.
- Every definitive decision agrees with CPU FP64 direct-difference evaluation
  of the original FP32 vectors. All output slots were written and guards held.
- The 97 real residuals were NOT resolved in the timed component. This is not
  a new exact final-pair export or an end-to-end join result.
- Memcheck and synccheck report zero errors. Outputs were independently
  reallocated and checked after timing. Graph, initcheck, racecheck, broad
  adversarial-input, and full production stress campaigns were not run.
- Compiled cubin hashes match across correctness, both sanitizer processes,
  and both timing processes. P1/P16/T4x4 use 34/137/96 registers respectively,
  zero reported spills, zero stack/local bytes, and no Tensor Core instructions.
- Triton's shared-memory metadata is 16/128/2048 bytes; cuobjdump separately
  reports SHARED:1024 for each compiled function. Both raw fields are retained;
  they are not treated as interchangeable resource measurements.
- No foreign GPU1 process was observed. Existing GPU1 locks were held; final
  GPU1 state was 14 MiB, 0% utilization, no compute process. GPU0's foreign
  training process remained active and was not touched.

## Interpretation, limits, and thesis impact

**Measured:** a layout-dependent crossover exists for synthetic requested work,
but none of the four actual-data layouts meets the opposite-winner gate.
Changing physical distribution while preserving logical work can matter; the
number of requested pairs alone does not describe every synthetic case.

**Inferred:** real-data padding dominates this direct-FP32 tiled implementation's
opportunity. T4x4 time closely tracks active tile count across the real layouts.
This is consistent with added padded work, not a profiler-confirmed statement
about a particular memory or arithmetic bottleneck.

**Scope limit:** T4x4 deliberately uses ordinary FP32 direct differences. It
does NOT test certified Tensor Core block refinement. It therefore cannot kill
all TC-based refiners, all tile sizes, all datasets, or adaptive precision as a
whole. Novelty remains UNKNOWN; adaptive precision, compaction, and generic
sparse-to-dense blocking already have close prior art (see the protocol).

**Useful engineering signal:** P16 is about 2x faster than the A0 aligned-output
P1 control on the actual workload. This is generic batching, not a research
contribution, and is not yet a speedup over the original compacting G3C keeper.
Any adoption must reintroduce real counters/compaction/FP64 repair and compare
the original keeper under the full stage denominator.

**Excluded costs:** row layout and queue construction were done on the CPU,
reported separately, and not included in the GPU ratio. They cost milliseconds
versus microseconds for these kernels; this implementation is not an online
router. These host costs neither rescue the tiled negative nor establish a
lower bound on a future GPU organization algorithm. CUDA-event blocks can
include Python submission gaps; sustained diagnostics use the same launch API.
Two processes provide a directional screen, not a publication-grade confidence
claim. The summary retains a clearly labeled small-sample diagnostic interval.

## Decisions and reopen conditions

| Level | State | Decision |
|---|---|---|
| Implementation | partial | Correct fixed-shape component with memcheck/synccheck; not production admission |
| Real-data A0 crossover | rejected | No winner switch; do not implement the proposed online router from this result |
| Synthetic locality mechanism | partial | Opposite winners reproduced on the same synthetic logical set |
| Tensor Core refinement | unknown | Not implemented or measured by A0 |
| Novel thesis / paper story | unknown | A0 supplies no real-data support sufficient to justify it |

Next defensible research action, if continuing: first define a distinct
certified-TC refinement mechanism and its numerical contract, compare with the
strong P16 control, and check an optimistic work/precision bound on real
uncertainty sets before implementing a router. Do not simply tune tile shapes,
add a cost model, or present synthetic clustering as the paper's main evidence.
An alternative engineering action is a separately frozen P16 integration test;
it must remain an implementation optimization, not a novelty claim.

G5's rejected full-scale result is unchanged and unexplained by G8. G7's
RT-HiSS exact high-dimensional admission remains pending. No manuscript or
Overleaf project was edited, compiled, or synchronized in this experiment.

## Evidence and reproduction

- Contract: `PROTOCOL_G8_REFINEMENT_LAYOUT_A0.md`.
- Full summary: `results/g8_layout_summary_a0.json`.
- Raw timing: `results/g8_timing_p0_a0.json`, `results/g8_timing_p1_a0.json`.
- Safety/isolation: matching `results/g8_*_manifest.json`, `raw/g8_*`.
- Compiled code and static audit: `artifacts/g8_check_a2/`; per-run caches retained.
- Failed pre-measurement setup attempts: `artifacts/g8_launcher_a0/`,
  `artifacts/g8_probe_a1/`, `raw/g8_check_a1.log` (no timing from these attempts).
- Evidence hashes: `artifacts/g8_evidence/evidence_manifest.sha256`.

From the remote project root, inspect results with:

```sh
@TENSORJOIN_ROOT@/isaacsim6/env/bin/python -m json.tool results/g8_layout_summary_a0.json
```

The frozen runner refuses to overwrite existing labels. Reproduction requires
a fresh label, fresh GPU1 quiescence, and the same existing lock domain; do not
delete evidence to rerun a favorable sample.
