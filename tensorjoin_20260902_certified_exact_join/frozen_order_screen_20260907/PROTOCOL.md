# Frozen-code direction-balanced complete-operator screen

Experiment: tensorjoin_20260907_frozen_order_screen_d512.
Predeclared before implementation and observations. This is a narrow
same-contract screen, not a novelty or final paper-performance admission.

## Hypothesis and keeper/control

The retained G16 TensorJoin pipeline preserves the fixed terminal predicate
and has a lower complete warmed host-to-host cost than the equally GPU-prepared
pedantic FP32 dense control, without depending on process order. Negative results
remain evidence; no clock changes, replacement processes or estimator selection.

A: original G16 INT8 certificate + FP32 filter + fixed FP64 terminal; four
retained cubins, including GPU preparation, directly loaded without compilation.
B: original GPU-norm FP32 dense control: cuBLAS GemmEx, compute32F_PEDANTIC69,
math mode2, FP32 inputs/output, algorithmDEFAULT(-1), NVIDIA_TF32_OVERRIDE=0;
retained GPU norm and three full/tail classifier cubins, same FP64 terminal.
B is a tested dense control, not a claim of the strongest possible/latest
algorithm. No CPU preprocessing handicap is imposed on B. The old MiSTIC
headline does not enter this comparison.

## Frozen execution and denominator

Host gpu-host-8 / original GPU2 / driver590.48.01 / Python3.12.3 /
torch2.11.0+cu130 / NumPy2.3.1; inherited configured SSH topology. Inspect live
state and require both per-GPU locks, 30s quiescence and no foreign occupancy.
No power/clock changes. Source/binary hashes, CUDA ABI and cuBLAS library hashes
must match the recorded manifests. Library algorithm/API metadata are recorded;
new timing is not a new runtime SASS attribution audit.

N60000,D512, finite stored FP32 |x|<=1, EPS161/256,T25921/65536. Exact retained
NPY and directed canonical output count/hash. Eager/default stream, pageable
host input -> sorted canonical uint64 host output. Timer includes new device
buffers, source H2D, all GPU metadata, scan/refinement/compaction, counters and
result D2H, scheduling and canonical sorting. It excludes file IO, context/module/
library initialization, correctness hashes and JSON. No prepared vector/codes/
norm state is retained between calls. Torch allocator reuse is permitted for
both methods: this is explicitly a warmed repeated-call denominator, not the
old single-shot fresh-allocation timer. No comparison to old absolute times.
No timed progress printing and no sampled-pair diagnostic transfer for either.

## Required new-harness gates before timing

- Original-host TensorJoin qualification complete under its separate protocol.
- Both new complete operator paths reproduce full output.
- New control launch wrapper passes runtime ABI/full-tail-prefix direct-terminal
  checks, high-entropy/cancellation/boundary fixture prefix checks and full output.
- Same-contract wrapper memcheck and synccheck on bounded prefix diagnostics;
  prior frozen kernel resource/sanitizer evidence is retained, not misdescribed
  as new whole-workload racecheck. Wrapper allocations/counters are re-created.
- Identity unchanged before/after each process. New candidate host path does not
  become the historical host executable simply because its cubins are unchanged.

## Sampling and decision (no replacements)

8 independent processes; order AB,BA repeated4 times. Within each process,
2 full warm calls per method followed by9 recorded complete calls per method,
then16 consecutive complete calls per method for repeated-call stress in the
opposite method-block order. Validate every returned count/hash outside each
timer. Keep every sample and failure. GPU memory target below4GiB.

Primary: geometric mean of8 process-paired ratios of method medians (B/A),
with a predeclared 10,000-resample order-stratified percentile95% bootstrap CI
(seed20260907). Report p10/median/p90, marginal ratio, arithmetic/geometric paired
means, process wins and AB/BA split. Repeated-call stress uses each process's
sum of16 latencies, then the same paired analysis; report first/last4-call ratio.
This is repeated complete-call stress, not continuous GPU saturation or
multi-hour sustained proof; correctness hashing inserts host work between calls.

A narrow screen win requires all8 processes valid, primary CI lower>1,
>=7/8 process wins, both order-group geometric ratios>1, and repeated-call
stress CI lower>1 with both order groups>1. Otherwise report inconclusive or
rejected at the measured scope; do not drop slow but admitted observations.
No public novelty/breadth/general-hardware claim is promoted by this screen.
