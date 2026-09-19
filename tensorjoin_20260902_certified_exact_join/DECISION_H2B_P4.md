# Decision: H2B-P4 Pedantic-cuBLAS Performance Kill Test

Date: 2026-09-03

## Decision

**REJECT H2B Phase P and stop the upper multi-vector performance thesis.**
The first predeclared fresh process already violates the frozen stop rule in
all four cells: both target-1 speedups are below 1.25x, and both target-8 cells
favor the pedantic-cuBLAS exact keeper.  Slots 1--7, bootstrap, sustained, the
larger streaming phase, and tree/index work were therefore not executed.

The exact multi-vector certificate remains a correct measured mechanism; the
rejected claim is that this H1 execution has a decisive same-contract
performance advantage over the strongest admitted direct-FP32 exact keeper.

## Exact outer-wall result

Each cell retains 10 excluded warmups and 100 direction-balanced observations
per variant.  The denominator includes all scan/dot work, object aggregation,
dynamic count synchronization, exact FP64 repair, final ID transfer, and host
canonical sort.

| Dataset | Target | Pedantic keeper median | H1 candidate median | Keeper/candidate | Paired candidate wins | Gate |
|---|---:|---:|---:|---:|---:|---|
| ESC-50/PANNs D2048 | 1 | 672.221 us | 575.269 us | 1.169x | 100/100 | fail `<1.25x` |
| ESC-50/PANNs D2048 | 8 | 674.442 us | 1,012.429 us | 0.666x | 0/100 | fail, candidate slower |
| UCF101/R3D-18 D512 | 1 | 505.183 us | 497.300 us | 1.016x | 76/100 | fail `<1.25x` |
| UCF101/R3D-18 D512 | 8 | 587.617 us | 783.151 us | 0.750x | 0/100 | fail, candidate slower |

All 800 retained calls reproduce the frozen direct/ambiguous/final counts,
exact ID hashes, zero duplicates, and zero overflow.  Direction-stratified
medians lead to the same conclusion; the loss is not caused by one A/B order.
GPU4 is idle before and after the process and no target-UUID compute process is
present at the shell postflight.

## Mechanism diagnosis

The actual pedantic keeper is a highly optimized FP32 SIMT SGEMM followed by a
very selective certificate.  It sends only 2/9 audio and 0/4 video object pairs
to FP64 at targets 1/8.  H1 sends 70/358 audio and 94/822 video objects to its
exact panel path.  At target 8, that additional repair work overwhelms the
candidate's faster INT8 Tensor-Core scan, while the direct cuBLAS keeper stays
near 0.59--0.67 ms.

This also explains why H2A was positive: its custom direct-difference FP32
kernel was a valid exact comparator but not the fastest admitted FP32
implementation.  H2A evidence remains true in its scope and may not be used
as the headline comparator after H2B.

## Preserved negative evidence

- **Target/workload:** resident 128x512 exact variable-cardinality Chamfer
  threshold join, PANNs D2048 and R3D-18 D512, target 1/8.
- **Keeper:** runtime-resolved pedantic FP32 `cublasGemmEx` plus certified
  object aggregation and exact FP64 repair.
- **Rejected mechanism:** current H1 object-first INT8 Tensor-Core certificate
  plus exact ambiguous-panel repair.
- **Added cost responsible:** 35x--206x more ambiguous object panels than the
  pedantic keeper across the four cells, with the decisive target-8 loss.
- **Measured effect:** 1.169x/0.666x/1.016x/0.750x; two cells slower and no
  cell clears 1.25x.
- **Reopen condition:** only a distinct mechanism that materially reduces H1
  ambiguity/repair cost and first clears the same four-cell 1.25x cheap kill
  may reopen this route.  More datasets, tree packaging, streaming scale, or a
  small kernel-only gain cannot rescue it.

## Evidence

- Rejection summary: `results/h2b_p4_rejection.json`, SHA-256
  `a61fbf9873f25c6a100e13754330d4dc233f89c085ed7f168a5a2794bc352fe3`.
- Slot-0 result/log: SHA-256 `c8ac7d4e...c9c` / `68db0aa9...ba96`.
- Timing runner/summarizer: SHA-256 `b6ea11bc...74e0` /
  `8f12c9c3...f260`.
- Pre/postflight: SHA-256 `4d30afee...9684` / `1c57e7fe...4506`.

## Strategic consequence

Do not run H2B streaming or add a conventional tree/index layer to this upper
route.  Preserve H0--H2A as scoped mechanism/ablation evidence, but return the
paper effort to the already supported single-vector exact precision-router
direction (G2B external exact comparison plus G4 breadth/dynamic routing).
