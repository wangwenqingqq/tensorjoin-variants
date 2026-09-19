# G16 decision: a narrow positive survives the equally optimized strong control

Verified 2026-09-05 on gpu-host-8 physical GPU2. All six predeclared processes,
all preceding safety/precision gates, and final code/output identity checks are
complete. **The narrow >=1.25x screen passes in both directions.** This is a
bounded engineering candidate, not a novelty, formal/sustained or paper-ready
promotion. No replacement process or favorable remeasurement was used.

## Strongest measured evidence

Same stored FP32 CIFAR-GIST60000x512 and D2=0.3955230712890625. Complete
pageable-host source to sorted canonical-host uint64 output, including all
preparation, allocation, transfers, count reads, refinement, mirror and sorting.
All six processes emit exactly3,926,078 IDs including self, raw hash
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
Startup/context/library/JIT, disk IO and post-timer checking are excluded.

| Fixed process block/order | GPU-prepared G5 (s) | CPU-norm FP32 (s) | GPU-norm FP32 (s) | Faster control / G5 |
|---|---:|---:|---:|---:|
| 0: G5, FP32-CPU, FP32-GPU | 0.162313295 | 0.308708357 | 0.243425080 | 1.499724 |
| 1: FP32-GPU, FP32-CPU, G5 | 0.191934949 | 0.306258334 | 0.243218102 | 1.267190 |

The paired geometric ratio is1.378563; the marginal median ratio is1.373735.
Both blocks win, but the second ratio only narrowly clears the1.25 gate.
G5's reverse observation is18.25% slower than its forward observation; its
cause is not localized. **Do not advertise stable1.5x or replace the reverse
sample.** Two blocks cannot support confidence, tail or sustained claims.
Descriptive n=2 p10/median/p90 values and every raw order are retained in
`results/closure_checks.json` and `results/public_screen.json`.

## What changed, and the attribution limit

The original G5 CPU quantization and host-created transposed operand were
replaced by one charged FP32 source transfer, GPU code/scale/norm/error
construction, a charged GPU transpose-contiguous copy, and normal downstream
execution. All three G5 routing/refinement kernels, capacities, launch arguments,
work counts and host canonicalization remain unchanged. Real60K metadata is
byte-identical to the original CPU construction. This supports a data-path
attribution, not a new pruning algorithm or a claim of reduced dense-pair work.

Before any G16 public timing, an addendum granted the FP32 control the same
GPU-preparation opportunity. Its norms now cost about0.00074–0.00075s instead
of0.0651–0.0654s CPU preparation in these public observations. The direct
pedantic cuBLAS calls, classifier, terminal and full output loop were unchanged.
The faster GPU-norm control, not the slower CPU arm, sets the acceptance ratio.
The CPU-prepared G15 loss remains real and is not reclassified.

PyTorch peak allocated bytes are593,544,704 for G5 versus469,237,760 for either
FP32 control in both public blocks. The candidate uses124,306,944 more measured
allocator bytes (about26.5%). This is not total device memory/driver-library
workspace accounting, and no memory-efficiency superiority claim is admitted.
Original FP32 input remains available for refinement; INT8 is not a replacement
for all retained input storage.

## Evidence gates

- GPU quantized metadata: full60K and six deterministic adversarial fixtures,
  zero independent residual/norm/transpose failures. The tiny subnormal fixture
  changes code/scale bytes under GPU execution, while its actual reconstruction
  norm and residual bound remain valid. Do not claim all-input byte identity.
- GPU FP32 norms: full60K plus six fixtures checked against independent extended-
  precision CPU sums; zero norm-enclosure/L2-upper violation. Seven complete
  small join fixtures also pass interval/direct-decision, rectangular cuBLAS
  mapping and reference-output checks, including ties and subnormals.
- Full60K compatibility and public output hashes match. Separate1000-call
  alternating-input/pointer-churn tests pass for metadata and complete N129
  FP32 joins, with two distinct input pointers. This is not1000 full60K joins.
- Full60K memcheck access tests for both complete paths, metadata synccheck
  across all fixtures including60K, and FP32 fixture synccheck report zero
  errors. Leak-free shutdown is not established; no new Graph/arbitrary-stream
  support or unbounded stress claim.
- GPU quantization: selected cubin SHA-256
  `028e1a231cb553df10b2754e7ff935d3d2fa534812608f4a9b7170f4502da21a`,
  848 static instructions,38 registers/thread,32 shared bytes, zero local bytes
  or static LDL/STL. GPU norm metadata: cubin
  `4b9b0373b3511123d13cf707bc656012839624f2165f36512911e533223734be`,
  376 instructions,30 registers/thread,32 shared bytes and zero local traffic.
  Both use four warps and have no MMA instructions.
- Actual NSYS dispatch plus three NCU full/tail exports verify the same G15
  pedantic SIMT SGEMM function, normalized SASS
  `fbc00ffde2c9eda01bdbbc0b3800a71f5a636128e3e00cb641efc7f8a33df7e3`,
  768 FFMA and zero MMA/TF32. Queried mode2/compute69, explicit TF32 override0,
  unchanged cuBLAS/cuBLASLt container hashes. Parent code-object linkage remains
  unresolved; actual selected-function export is verified.
- All public original G5 and FP32 classifier/terminal cubin/PTX sets match their
  corresponding compatibility evidence. G5 still has1,827,007 first-stage
  uncertain pairs and1,734 terminal pairs. No profiler time enters the ratios.

## Implementation, mechanism and thesis decisions

1. **Implementation:** keep this separately versioned GPU-prepared prototype as
   an admitted narrow CIFAR60K early-screen candidate. Do not delete old G5/G15
   sources or make the new variant a general default.
2. **Mechanism:** a material preparation-path bottleneck was removed and a
   same-output advantage survives a control given equal optimization. This is
   positive evidence against declaring TC joins futile from G15 alone. It does
   not establish a universal TC lower bound or every-order stable speedup.
3. **Thesis:** no standalone novelty pass. Quantization, deterministic bounds,
   adaptive precision and GPU preparation have antecedents. A non-incremental
   mechanism/cost interaction beyond the nearest work remains to be established.
   Modern same-contract RT-HiSS/repaired-FaSTED/COSS comparisons remain pending.

Next follow `PAPER_READINESS.md`: resolve the differentiated mechanism and admit
modern output adapters with safe tiny/4096 gates before expensive expansion.
If the core thesis is subsumed, switch it rather than using data breadth or a
conventional tree to disguise overlap. If a modern strongest control removes
this narrow advantage, preserve the result and reassess the mechanism. Do not
infer an untested baseline is slow or incorrect.

The current historical manuscript remains unchanged and contains identified
stale statements. No local paper compile/render or Overleaf synchronization is
claimed. G10's exact-real counterexample remains valid: these results establish
frozen FP64-reference output on tested inputs, not arbitrary-input exact-real
predicates. G5's historical failed screen and G15's adverse comparison remain
intact. Two launch-environment failures are retained across G15/G16, not hidden
as numerical success or timing observations; see each `ATTEMPTS.md`.
