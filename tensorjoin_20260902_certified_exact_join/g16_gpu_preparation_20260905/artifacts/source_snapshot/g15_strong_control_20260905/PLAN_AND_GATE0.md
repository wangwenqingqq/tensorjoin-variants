# G15: make the original static-join evidence defensible

Declared 2026-09-05, before new implementation or measurement. The user
authorizes autonomous continuation without per-step confirmation. Shared-host,
publication, credentials and exact-evidence boundaries remain in force.

## Research decision before engineering

The original static-join asset is retained; no replacement topic is introduced.
G14 reproduced a useful implementation-level positive but did not resolve
novelty or the strongest adaptive-FP32-first comparison. G10's generic
certificate/precision-escalation overlap remains. The working research question
is whether a fully accounted TC-first exact-oracle join removes a material cost
that a well-engineered adaptive FP32-first join still pays. This is a necessary
cheap attribution test, not a new contribution or a novelty pass.

Current primary-source refresh:

- [RT-HiSS](https://arxiv.org/abs/2609.01975), posted 2026-09-02, explicitly
  includes candidate refinement, two-pass bounded result sizing, batching,
  shared-memory tiling and compressed result masks. Do not claim those generic
  ingredients as new. Its project code was located in G7; full same-output
  adapter admission remains separate.
- [FaSTED](https://arxiv.org/abs/2508.21230) is the nearest TC self-join route;
  approximation errors in the frozen earlier comparison do not justify
  omitting a repaired/adaptive control.
- [NVIDIA cuBLAS compute types](https://docs.nvidia.com/cuda/cublas/index.html#cublascompute-type-t)
  distinguish pedantic FP32 from reduced-precision modes. Use direct queried
  pedantic calls and resolve actual selected code, not only a framework flag.
- Existing VA-file, adaptive predicates, QDOT, TRIM, and certificate semantic
  distinctions remain recorded in `../DECISION_G10_CERTIFICATE_NOVELTY.md`.

No expensive dataset/hardware expansion or paper-facing novelty claim is
admitted by an engineering win alone. A subsumed thesis is not rescued by
more packaging. If an independently optimized control removes the advantage,
retain the result and separate implementation failure from a mechanism limit.

## Work packages and predeclared stops

### A. CPU data-path attribution, not a new numerical algorithm

Compare original analytic quantization plus full `codes.T.copy()` against
4096-row blocking that calls the *same* quantizer on each row block and writes
the identical row-major and transposed INT8 operands. Charge all allocations,
four metadata/code outputs, transpose and assembly. Do not reuse prepared
state across measurements. Frozen input: original contiguous FP32 60000 x 512
CIFAR-GIST, NPY hash `95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c`.

Four independent CPU processes in original/chunked/chunked/original order;
node-0 CPU/memory placement, one full preparation each. Reference computation
and all five bitwise-output checks follow the timer. Retain every observation.
Any output-byte difference stops candidate integration. A minimum paired
preparation ratio of 1.20 admits only a full-operator integration screen;
it is neither a GPU speedup nor a paper contribution. No block-size sweep.

### B. Complete pedantic FP32-first control

Extend the previously admitted H2B direct cuBLAS FFI binding to the original
single-vector self-join. Use `CUBLAS_PEDANTIC_MATH`,
`CUBLAS_COMPUTE_32F_PEDANTIC`, `NVIDIA_TF32_OVERRIDE=0`, queried API modes,
and runtime-selected code evidence. No input quantization is required by this
control. Charge FP64 norm metadata construction, input transfers, allocations,
all GEMMs, interval classification, compaction, dynamic counts, the unchanged
G5 FP64 terminal, accepted-ID transfer, symmetry expansion and sorting.

Stream 4096-row panels over upper-triangle block pairs; diagonal GEMMs also
compute their discarded lower half. Record this extra work. Three full-input
GEMM shapes are 4096x4096x512, 4096x2656x512 and 2656x2656x512. Worst-case
per-panel output/ambiguity capacities are 4096^2 IDs, not oracle-sized buffers.
All source FP32 coordinates must be finite with absolute value <= 1. Thresholds
used here are nonnegative exactly-FP32-representable values. Uncovered domains
fail closed rather than inheriting a universal certification claim.

Reuse the H2B norm/dot-error envelope, explicitly adding an FP64-terminal
rounding allowance. Keep the H2B envelope's coefficients fixed before data
measurement. No post-failure radius widening or selected-threshold tuning.
Exactness means agreement with the frozen original terminal/oracle, not an
arbitrary-input exact-real predicate. Preserve G10's counterexample distinction.

Validation ladder: deterministic rectangular layout check; exhaustive small
real and signed/cancellation/zero/tie/subnormal/ragged fixtures; full 60K output
hash; selected function/precision audit; memcheck and synchronization checks;
1000 alternating-buffer subset invocations; then bounded same-contract timing.
Missing later gates permit diagnostic attribution only, never promotion.

### C. Public operator screen, only after relevant correctness gates

Two reversed-order independent-process blocks: G5-original, FP32-first,
G5-chunked; then G5-chunked, FP32-first, G5-original. The chunked variant is
admitted only if A passes. Freeze all source hashes before launch; no replacements
or favorable reruns. Use original N=60000, D=512, epsilon=0.62890625,
threshold_d2=0.3955230712890625 and sorted directed uint64 `row*N+column`
output, including self: 3,926,078 IDs, hash
`13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.

Public scope remains pageable-host source to sorted-canonical-host output,
excluding disk IO, process/context/library/JIT setup and post-timer validation.
Warm exact specializations outside timing and retain no method-prepared input.
Use default CPU placement for all three public variants, rather than selecting
the placement that gave a favorable external ratio in G14.

Predeclared integration target: chunked G5 must be >=1.15x faster than original
G5 in both blocks to remain a useful narrow engineering candidate. The strong
control comparison is symmetric and has no presumed winner. A TC-specific
performance thesis needs >=1.25x against the fastest admitted independently
prepared FP32-first control in both blocks before any formal expansion is
considered. A failure does not license a return to MiSTIC-only comparisons.
Report all raw times, marginal and paired estimators and order split; two
blocks do not establish a confidence interval or sustained tail behavior.

## Host/run boundary

Host `gpu-host-8` via `tiaoban`; project
`@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`; no Git root.
Physical GPU 2, UUID `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245`, RTX PRO 6000
Blackwell Server Edition, driver 590.48.01. Python
`@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`, PyTorch 2.11.0+cu130,
Triton 3.6.0, NumPy 2.3.1; installed nvcc 13.1.115. Reverify live state.
No clock/power or shared-service changes. GPU 1 foreign PID 1460887 is
do-not-touch. One campaign lock `/tmp/tensorjoin_gpu2_campaign.lock`, plus
the existing G5 GPU-2 per-child guard/30-second quiescence/250-ms monitor.
Foreign occupancy stops only our child and retains the failed attempt.

All new work is additive under this directory. Raw sources, rejected
observations and original manuscript remain unchanged. Back up mutable route
documents before adding completed evidence. No Overleaf upload is implied;
do not reuse or persist the old session credential.

## Paper-level acceptance ledger

| Surface | Current status | Evidence required |
|---|---|---|
| Specific novelty beyond nearest work | unknown | A non-incremental mechanism/cost removal, not a new label for bounds/refinement |
| Exactness contract | scoped | Reference equivalence stated precisely; no exact-real overclaim |
| Strongest complete comparator | missing | This G15 control plus modern external adapter admission |
| End-to-end attribution | partial G14 | Eliminate hidden preprocessing/packing/export advantages |
| Correctness/code/safety/stress | G5-only historical | Re-establish for changed/new paths and intended shapes |
| Generality and tails | missing | Only after cheap mechanism and baseline gates pass |
| Manuscript coherence | not edited | Lock thesis, contributions and claim map before rewriting/figures |

The paper will be written to the strongest justified claim, not to a desired
acceptance outcome. Report unresolved gates honestly while continuing useful
bounded work without requiring another user approval.
