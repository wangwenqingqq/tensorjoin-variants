# Protocol: H2B-P4 Pedantic-cuBLAS Outer-Wall Kill Test

Date: 2026-09-03

## Decision question

Does the unchanged H1 exact INT8 Tensor-Core candidate retain a material
advantage over the P1--P3 admitted direct pedantic-cuBLAS FP32 exact keeper in
every frozen audio/video target cell?

This is the Phase-P performance kill test.  Failure closes the upper
performance thesis before any streaming or tree/index work.

## Immutable dependencies

- H1 correctness/runner/kernel SHA-256:
  `06461064d1b237aaa29f4face0eb3c223f5487abf478ddcf5fccb353ed28dca4` /
  `11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e` /
  `c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db`.
- H2B P1/P2/P3 SHA-256:
  `68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99` /
  `c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77` /
  `3a02d1a55016c70989d59ace2b1003d1ccbde9ba2bb7720bee45354ae9efd245`.
- Timing runner: `src/run_h2b_p4_timing.py`, SHA-256
  `b6ea11bc5948bf26cb29d49ae415cd0cb570bb8f98cf74123a5c33f40c6574e0`.

## Frozen operator denominator

Host monotonic outer wall starts immediately before and stops immediately
after one complete variant call.  Both variants include:

- all GPU scans/dot products and object aggregation;
- device-side status compaction;
- actual dynamic count transfer and synchronization;
- exact FP64 repair at the dynamically observed extent;
- final count and ID transfer;
- host canonical sort.

One-time data loading, normalization, H1 quantization/scales, H2B exact norm
metadata, device allocation, cuBLAS handle setup, compilation, and warmup are
excluded symmetrically.  ID hashing/validation after the returned sorted IDs
is outside the timed interval.  No profiler or sanitizer duration is used.

## Frozen workloads and target

- PANNs D2048 fixed-five-token objects and R3D-18 D512 ragged 4--7-token
  objects;
- 128 disjoint query objects x 512 disjoint base objects;
- exact target 1 and 8 thresholds/canonical IDs from P1/H1;
- physical GPU4 on `gpu-host-8`, identical RTX PRO 6000 Blackwell Server
  Edition / SM120, natural clocks;
- GPU4 campaign lock and foreign-PID refusal before every process;
- `CUDA_VISIBLE_DEVICES=4`, `NVIDIA_TF32_OVERRIDE=0`, IEEE PyTorch FP32
  flags, queried pedantic math mode, explicit pedantic compute type.

## Fresh-process campaign

Run slots 0--7 as eight separate Python processes.  Each process runs all four
cells with 10 excluded warmups per variant and 100 retained outer-wall
observations per variant/cell.  Variant order alternates per observation and
is shifted by process/cell; dataset and target order are also shifted across
slots.  Thus each cell retains 800 observations per variant across eight
process units without treating observations as independent processes.

For each cell calculate the process-paired ratio

```text
speedup_slot = median(baseline outer wall) / median(candidate outer wall)
```

Report the geometric mean of the eight paired ratios and a two-sided 95%
percentile interval from 20,000 seeded bootstrap resamples of the eight process
units.  The lower endpoint, not individual observations, is the promotion
statistic.

## Sustained campaign

Run one additional isolated process.  Per cell, after 20 warmups/variant,
retain two direction-balanced blocks: 500 baseline then 500 candidate calls,
and 500 candidate then 500 baseline calls.  Every block and their combined
1,000 observations/variant must exceed 1.25x by median outer wall.

## Pass/stop rule

P4 passes only if:

1. every timed call reproduces the frozen variant-specific direct/ambiguous
   counts, final count, exact ID hash, zero duplicate, and zero overflow;
2. every one of the 32 process/cell medians favors the candidate;
3. every cell's paired-bootstrap 95% lower bound is strictly greater than
   1.25x;
4. both sustained direction blocks and the combined sustained result exceed
   1.25x in every cell;
5. no target-GPU foreign process or execution failure occurs.

Any exactness failure rejects the implementation.  Any performance failure
closes H2B Phase P; extra packaging, a tree, or a larger benchmark may not
rescue it.  Passing P4 admits only a separately frozen larger streaming ladder,
not a paper/system claim by itself.

## Evidence paths

- `results/h2b_p4_process_0.json` through `_7.json`;
- `results/h2b_p4_sustained.json`;
- append-only `raw/h2b_p4_*` logs and pre/postflight snapshots;
- a deterministic post-run summary with a separate source hash.

## Current state

Rejected by the predeclared early-stop rule in fresh process slot 0.  All 800
calls were exact, but the four median speedups were
1.169x/0.666x/1.016x/0.750x; no cell cleared 1.25x and two cells favored the
keeper.  Slots 1--7, bootstrap, sustained, streaming, and tree/index work were
not executed.  See `DECISION_H2B_P4.md`.
