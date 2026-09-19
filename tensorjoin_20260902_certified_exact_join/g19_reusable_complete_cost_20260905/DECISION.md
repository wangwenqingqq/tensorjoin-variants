# G19 decision: reusable controls admitted; performance screen not passed

Verified 2026-09-05 on gpu-host-8, RTX PRO 6000 physical GPU3. The frozen
CIFAR4096 full host-to-host comparison completed without favorable reruns.

## Conclusion, strongest evidence, caveat and next action

The current TensorJoin path has lower medians than both admitted controls in
both process blocks, but it **fails the predeclared every-block 1.10x screen**
against the faster control. FP32-first / TensorJoin is **1.287224x / 1.059163x**;
repaired RT-HiSS / TensorJoin is **11.217547x / 10.243591x**. The second block's
small margin cannot be rescued by selecting the RT comparator, the first block,
or a more favorable aggregate. This rejects the narrow performance gate, not
all numerical work or all possible future mechanisms.

The strongest new positive asset is a complete reusable comparison stack:
all nine fixture outputs agree, all three full shutdown memchecks are clean,
and 100 RT plus 32 TC plus 32 FP32 full alternating-input stress calls pass.
The actual pedantic cuBLAS function still matches G16 after switching to owned
handle lifetime; the control was not replaced by TF32 or a weaker math mode.

The main caveat is scope: one 4096x512 input/threshold, two independent blocks,
five retained samples per method/process, natural clocks and shared-host CPU
contention. There is no CI, formal/sustained promotion, general RT traversal
certificate, exact-real theorem or novelty pass. G16's separate 60K positive
screen remains intact; its ratios are not pooled with this result.

**Next:** retain the corrected comparator and all raw evidence. Stop performance
promotion and broad paper expansion. Follow NEXT_RESEARCH_GATE.md to identify
one non-incremental work-elimination mechanism and run a fresh cheap Gate 0
before adding data, conventional tree/shape routing, or paper prose. The present
certificate/lifecycle engineering does not supply that contribution by itself.

## Complete-cost result

Input: prevalidated immutable contiguous pageable-host FP32 4096x512 and stored
FP64 T=0.5686872086178483. Output: independently owned sorted canonical host
uint64 directed/self IDs, 262,144 entries, SHA-256
`da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`.

Each fresh method process excludes data-independent engine/JIT initialization,
executes two full warm calls, then retains five full operations. All per-input
preparation, H2D, BVH/SBT/candidates where applicable, numerical work, counts,
D2H, mapping/sort and per-call cleanup are charged. Final engine shutdown is
separately checked for every method, not subtracted after measuring. RT's
raw-mask/counter/disk diagnostics are compiled out. Ordinary allocator caches
and explicit library workspace are retained without retaining input metadata.

| Block | Process position | Method | p10 (ms) | Median (ms) | p90 (ms) |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | Repaired RT-HiSS (off/R1) | 72.772967 | 74.063632 | 74.605818 |
| 1 | 2 | TensorJoin (G19 TC) | 6.573103 | 6.602480 | 7.944336 |
| 1 | 3 | Pedantic FP32-first (owned/R3) | 8.447884 | 8.498871 | 8.741510 |
| 2 | 1 | Pedantic FP32-first (owned/R3) | 8.540313 | 8.586318 | 8.945724 |
| 2 | 2 | TensorJoin (G19 TC) | 8.022550 | 8.106700 | 8.223603 |
| 2 | 3 | Repaired RT-HiSS (off/R1) | 74.304682 | 83.041719 | 87.979351 |

The primary strongest-control ratio is min(median RT, median FP32)/median TC
within each block. Both blocks must be >=1.10; only block 1 passes. Descriptive
process wins are 2/2, arithmetic ratio mean 1.173194, geometric process-paired
ratio 1.167639, and pooled marginal ratio 1.074153. These are different
estimators, not interchangeable evidence of passing the primary rule.

All 30 timed observations and 12 warm calls have exact outputs. All are kept in
`results/screen.json`, per-process `results/inner_screen_*.json`, and raw
`calls.jsonl`. GPU guards are clean in all six slots. TC's block-1 observations
span roughly 6.56--7.96 ms and block-2 observations 7.99--8.24 ms; this is an
observed distribution difference, **not a proven CPU/NUMA/clock diagnosis**.
No sample or block was removed, replaced or rerun.

## Admission and mechanism evidence

- Fourteen composite admission slots: four complete matrices, four full safety
  slots (RT memcheck/synccheck, TC memcheck, FP32 memcheck), three stress slots,
  three diagnostic NSYS profiles. Exact predecessor lineage is in
  `results/admission_r3.json`; all A0/R1/R2 failures stay failed.
- RT executes 100 full calls across two engine lifetimes; TC/FP32 execute 32
  each. Post-warm device-used range is zero in all four engine cycles. RSS
  ranges are 8,437,760 / 53,248 bytes for RT, 36,495,360 for TC and 4,255,744
  for FP32, within the frozen 64 MiB cap. This is bounded stability, not an
  arbitrary-duration service or 60K sustained test. RT's residual driver/context
  footprint does not return to the initial pre-JIT level between engines; it
  is stable across the two closes. Process-shutdown full memcheck is separate.
- RT off refinement is 968 static instructions, 40 runtime registers, 47,104
  shared bytes and zero observed local traffic. Its diagnostic-on counterpart
  is 992 instructions. The private OWL host-release fix leaves each mode's
  CUDA instruction stream identical to A0. Compression retains the exact
  408-instruction G17/G18 stream.
- Actual Triton captures bind four TC and three FP32 compiled objects to real
  launches. Stored FP64 threshold literal is present in classification/terminal
  PTX. The selected TC stage has IMMA; the other owned functions have no MMA.
  Audited cubin/PTX hashes recur in every corresponding timing-process cache.
- Fresh NCU selected-function export confirms pedantic cuBLAS's 2,424-instruction
  SIMT kernel, no MMA/TF32 instruction, same normalized hash as G16:
  `fbc00ffde2c9eda01bdbbc0b3800a71f5a636128e3e00cb641efc7f8a33df7e3`.
  Handle ownership is new; math mode 2, compute 69, algorithm -1, stream and
  8,519,680-byte workspace are explicit. The parent code-object image and
  OptiX driver-JIT identity remain unresolved, not guessed.
- Real-input work: TC schedules 2,080 upper block tiles; 102,079 pairs enter
  FP32 and 97 reach FP64. FP32-first computes a full 4096-square GEMM within
  its frozen panel and refines 589 upper pairs. RT's diagnostic build retains
  all 16,777,216 candidates, with 184 FP64 terminals, equal to G18's counters.
  Small fallback does not imply an output-sensitive whole-join algorithm.

**Attribution limitation:** the admitted FP32 control computes both triangles
inside its 4096-square panel, while TC uses upper block tiles. It is the faster
admitted control, not a proof of the globally fastest FP32/SYRK/blocking path.
The measured ratio therefore cannot be attributed solely to precision routing.
Likewise RT's adapter is not asserted to be the fastest possible ID exporter
or allocation-pool implementation. Preserve these limits in any later claim.

## Ownership failures repaired without suppression

A0 exposed an OWL cudaMallocHost/cudaFree mismatch (72 bytes). R1 fixes only the
two private PinnedHostMem releases and passes full RT safety. TC then exposed
425,721,860 bytes of retained allocator caches; R2 adds explicit engine shutdown
cleanup and passes full TC memcheck. FP32 still exposed 8,520,704 bytes in a
borrowed cuBLAS handle's internal allocations. R3 owns a new handle, preserves
the chosen workspace capacity, and destroys it explicitly; full FP32 memcheck
becomes zero-error/zero-leak. Installed libraries and borrowed handles are not
modified or destroyed. See ATTEMPTS_FINAL.md and the three design addenda.

No old experiment, shared upstream source, manuscript, Overleaf project, foreign
process, GPU clock/power setting or service was modified. Hash-preservation and
postflight evidence are in `results/closure_checks.json`; the final remote
inventory is `artifacts/raw_evidence_manifest.json`.

## Three separate decisions

| Level | State | Reopen condition |
|---|---|---|
| Reusable same-output comparator implementation | measured, bounded admitted | New shapes, streams, Graph, concurrency, allocator/toolkit changes require separate admission. |
| Every-block >=1.10x complete-cost claim | rejected for this frozen screen | A separately designed attributable mechanism and new predeclared campaign; never a favorable replacement of these blocks. |
| Non-incremental database/systems paper thesis | unknown; no novelty pass | Concrete work term beyond existing filters, closest-prior-art distinction, decisive same-contract mechanism evidence. More engineering or a renamed conventional planner is insufficient. |
