# G9 decision: small captured compute gain, opportunity gate not met

Verified 2026-09-04 on gpu-host-8 physical GPU7.

**Close these two TC-refinement prototypes at the current scope.** The best
actual-data result is a real but narrow 1.125–1.138x speedup under prebuilt-queue
CUDA Graph replay. It does not meet the predeclared 1.15x gate in both processes,
even before encoding, packing, queue formation, count discovery, Graph capture,
or final pair export is charged. Do not lower the gate retrospectively, extend
the tuning menu, or promote this into the proposed paper story.

This is NOT a claim that TC refinement can never help: the captured diagonal
prototype has a verified component-level gain. Full-cost performance is
UNKNOWN, not measured negative. The point is that the present evidence does
not justify further investment in this story under the agreed opportunity gate.

## Frozen comparison

All actual cases share the same 102079 G3B-R1 uncertain pairs, N4096/D512 FP32
CIFAR-GIST input and exact binary64 threshold. Two INT8 limbs represent each
approximately 14-bit coordinate. Four native INT8 MMAs reconstruct each
quantized dot product exactly; a directed-FP64 interval encloses the original
distance, followed by actual FP32 and interval-guarded FP64 repair.

P16 is the strong G8 batched FP32 component PLUS its required FP64 repair.
Both candidates use the same 16x16 TC kernel. Physical packing computes active
Cartesian tiles; diagonal packing computes 16 requested edges per tile and
discards off-diagonal products. No tile/precision tuning followed measurement.

Table entries are the predeclared paired `P16 time / candidate time`, aggregated
geometrically across two fresh processes within EACH execution mode. Values
above 1 favor TC. Do not combine eager and Graph observations into one ratio.

| Workload | Eager physical TC | Eager diagonal TC | Graph physical TC | Graph diagonal TC |
|---|---:|---:|---:|---:|
| Real / identity | 0.325x | 0.860x | 0.327x | 1.131x |
| Real / random | 0.326x | 0.766x | 0.330x | 1.135x |
| Real / degree-informed | 0.689x | 0.791x | 0.702x | 1.138x |
| Real / PCA1 | 0.521x | 0.858x | 0.531x | 1.125x |
| Synthetic / clustered | 0.970x | 0.967x | 2.998x | 1.082x |
| Synthetic / scattered | 0.364x | 0.985x | 0.367x | 1.124x |

For scale, Graph real_identity medians were 68.003/67.826 microseconds for P16
and 60.339/59.786 microseconds for diagonal TC. This saves about 8 microseconds
in the optimistic captured path. Whether a full implementation can retain any
of that saving has not been measured. The synthetic clustered 3x result is a
positive control, NOT a result on real quantization uncertainty.

## Why a Graph diagnostic was added

The original A0 no-Graph experiment failed and is retained as a negative.
Separate-stage samples did not add up to the full-path samples: short kernels
included appreciable host submission effects. R2 was preregistered BEFORE
Graph testing to isolate the execution-mode effect, without changing any
kernel, input, precision, queue or comparator.

All 18 compiled cubins are identical across eager/Graph correctness, sanitizer,
and timing runs. Changing only capture/replay changed diagonal TC from an eager
loss to a small captured gain. This establishes that submission/launch mode
matters here; it does not identify a single Python frame as the sole cause.
No NCU dynamic critical-path attribution was performed. R2 did not turn A0 into
a pass, and R2 independently failed the same 1.15x actual-data opportunity gate.

## Correctness / safety / numerical boundary

- On every real layout, TC reduces the FP32 repair set from 102079 to 3556;
 97 pairs still require FP64. All final predicates resolve: 59905 reject,
 42174 accept, 0 unknown. The logical final state hash agrees across layouts,
 methods and execution modes.
- Every requested quantized dot product matches an exact CPU INT64 reference.
 Residual norm bounds are checked by exact integer inequalities on an admitted
 2^-40 grid. Directed rounding is explicit, not calibrated against observed error.
- Three signed high-entropy fixtures cover zeros/equal vectors, the last partial
 16-pair group, and thresholds immediately below/equal/above 0.5625. Exact dyadic
 CPU arithmetic verifies interval enclosure and every definitive decision.
 Intended boundary overlaps remain unresolved rather than being guessed.
- Main-workload final predicates match the CPU oracle and are interval-resolved;
 no new full self-join pair export or initial-filter comparison was performed.
- Memcheck and synccheck report zero errors in BOTH eager and Graph modes.
 Guard bytes and output reallocation tests pass; Graph pointers are recaptured
 after reallocation. Each mode has two fresh forward/reverse timing processes
 and separate 500-path sustained diagnostics. No Graph INIT/RACE sweep or broad
 all-input production certification is claimed.
- The TC kernel has 74 registers, no reported spills/stack/local memory, and 128
 static IMMA instruction sites. These are not dynamic execution counts.
 PTX uses documented saturating s8*s8->s32 m16n8k32 instructions. Proven bounds
 keep every partial sum below 2^24, so saturation cannot activate.

## Cost and mechanism interpretation

**Measured:** physical tiles still pay 30.09–74.82x padded products on actual
layouts. Diagonal packing fixes this at approximately 16x; it is much faster
than physical packing on the actual data but still spends four limb products,
INT64 combination, directed certificate work and extra stages.

**Measured:** 3556 FP32 survivors and 97 FP64 survivors are invariant across real
layouts and both TC packings. Native TC execution and complete numerical repair
are included in the measured path. Faster approximate multiplication was not
mistaken for completed exact refinement.

**Limit:** preprocessing, tile metadata and intermediate queues are prebuilt
using untimed real GPU stage masks, not the oracle's final answer. Both baseline
and candidate receive the same free-queue treatment. Capturing known work is an
optimistic diagnostic, not an online algorithm. Individual eager stage timing
samples contain launch/submission gaps and must not be summed as a critical
path or mixed with Graph timings. Two processes are a directional screen,
not publication-grade statistical coverage.

**Novelty:** UNKNOWN. Integer-limb GEMM, TC distance computation and generic
sparse-to-dense blocking are established prior art. A small captured operator
gain does not supply a non-incremental database/systems thesis. The proposed
actual-data distribution-based winner-switch story remains unsupported: the
same diagonal method is best among the TC variants on every actual layout.

## Decisions

| Surface | State | Allowed action |
|---|---|---|
| Numerical component | partial | Retain checked kernels/proof as research infrastructure |
| Eager TC schedules | rejected | No actual-data win; do not adopt or retune from A0 |
| Captured diagonal component | partial | Retain the verified 1.125–1.138x narrow gain, without erasing it |
| R2 opportunity gate | rejected | Below 1.15x in both processes; no fully-costed expansion in this turn |
| Full online join | unknown | No performance claim; no keeper replacement |
| Top-venue research story | unknown | Do not draft/promote from G9; return to novelty-first problem selection |

Reopen only for a separately justified mechanism/workload with a fresh novelty
audit and a decisive falsification test. Not by lowering 1.15 to 1.10, selecting
the synthetic control, adding a generic router, or resampling a favorable run.
G5, G7 baseline-admission pending work and the original G8 evidence are unchanged.
No manuscript/Overleaf edit or synchronization was performed.

## Provenance / failed attempts

GPU1 became occupied after the initial check. The next preflight aborted without
launching; all admitted runs moved to GPU7 under the preregistered R1 revision.
All foreign processes were left alone. Existing GPU7 locking and monitoring
show no foreign process during admitted slots.

Installed nvcc/disassembly tools are CUDA 13.1.115, but the actual Triton JIT
assembler is `ptxas-blackwell` 12.9.86, SHA256
983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461.
Do not confuse these toolchain roles. Target PTX is sm_120a.

Retained setup failures: A0 FP64 scalar-width/threshold bug caught by a boundary
fixture; A1 occupied-GPU skip; A2 missing explicit INT32 dot output dtype;
two static-audit parser assumptions (decimal FP64 immediates and satfinite
spelling). All were fixed before any timing. No failed timing sample was removed.

Sources: `PROTOCOL_G9_TC_REFINEMENT_A0.md`, `NUMERICS_G9_INT14.md`,
`PROTOCOL_G9_GPU7_R1.md`, `PROTOCOL_G9_GRAPH_DIAGNOSTIC_R2.md`.
Results: `results/g9_summary_a0.json`, `results/g9_graph_summary_a0.json`,
`results/g9_static_audit_a0.json`, all corresponding raw samples/manifests.
Evidence manifest: `artifacts/g9_evidence/evidence_manifest.sha256`.
