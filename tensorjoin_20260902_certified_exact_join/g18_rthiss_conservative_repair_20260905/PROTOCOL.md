# G18: two-sided RT-HiSS numerical repair, bounded correctness admission

Declared 2026-09-05 before implementation and measurement. This is modern
baseline normalization, not a new TensorJoin mechanism or a novelty pass.

## Gate 0 and decision

Pinned RT-HiSS already supplies the RT index/filter, native FP32 refinement,
batching and compressed output. Adaptive robust predicates have antecedents;
G10/G16/G17 audits remain active. This small gate tests comparator quality and
correction work, not a claim of new filter/refine, generic precision escalation,
GPU preparation or a new system. No paper drafting or broad/60K timing campaign.
Primary sources: https://github.com/revanthmunugala/rt-hiss (pinned G17 commit),
https://arxiv.org/abs/2609.01975, and NVIDIA floating-point semantics at
https://docs.nvidia.com/cuda/floating-point/index.html.

Positive hypothesis: retain FP32 prefix rejection and use original-order FP64
only when a conservative two-sided band cannot decide; both native error
directions can be corrected without converting all pairs to FP64.
Negative hypothesis: unsafe candidate generation, inaccurate bounds, changed
reference semantics, excessive fallback or unsafe resource use blocks admission.
Both are tested; native G17 and all prior positives/negatives remain unchanged.

## Execution card

Host: gpu-host-8, local -> tiaoban -> root@192.0.2.8:22223, gpu-host-8.
Project: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join; no Git root.
New directory: g18_rthiss_conservative_repair_20260905, separate adapter_a0.
GPU: physical 2, GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245, RTX PRO 6000.
Driver 590.48.01, nvcc13.1.115, SM120, read-only OptiX9.1, original OWL pin.
Python: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python; CMake build -j4.
Locks: /tmp/tensorjoin_gpu2_campaign.lock plus unchanged G5 guard/quiescence.
Do not touch GPU0 PID1754058 (Physion), GPU1 PID1460887 (external-user), other jobs,
power/clocks, services or original G5/G15/G16/G17 sources/binaries.
Raw/result/guard paths: this directory and unchanged root G5 guard receipts.
Rollback: disable new adapter admission; keep originals and failed evidence.
No hardware execution, service restart, publication, manuscript or Overleaf edit.

## Frozen semantics and invariants

Keep G17's nine raw FP32 D512 inputs, epsilons, frozen reference thresholds and
canonical directed IDs, including boundary_zero32. N <= 4096; finite |x| <= 1.
No original threshold, fixture, epsilon or output filtering is changed.
Reference terminal: original-dimension-order sequential round-to-nearest FP64
subtraction, separate multiplication and addition, inclusive T comparison.
Its agreement with the frozen SciPy pair oracle must be measured, not assumed;
any output discrepancy stops the gate. This is not an exact-real predicate.

Only native pointWithinEps is replaced by the documented conservative test.
Upstream default shared/shared ownership, dimension/point order, RT geometry,
candidate generation, query/primitive staging, barriers, mask/compression and
ID decoder stay unchanged. Inverse dimension order and precomputed cutoffs are
loaded explicitly. Diagnostic counters and G17 raw-mask export remain enabled;
all operational times are excluded from performance claims.

Candidate coverage is tested separately from predicate output. Require zero
missing oracle-positive candidates on every full fixture. G17 coverage was dense
on these inputs and is NOT an all-input guarantee. Audit RT source boundaries;
do not claim a general conservative RT index or disable it silently. A coverage
failure stops refinement-only admission and needs a separately declared change.

## Numerical and resource design

Read NUMERICAL_DESIGN.md before CUDA implementation. Derive cutoffs from exact
rational bounds and round outward to FP32 on CPU, never fit a tolerance to G17's
four errors. Native FP32 prefix length/checkpoint order remains four dimensions.
Safe rejection uses the upper cutoff at every checkpoint. Safe acceptance uses
the lower cutoff only after all dimensions. Otherwise replay FP64 in original
order. Count safe reject/accept, fallback accept/reject and FP32/FP64 dimensions.

No new synchronization edge: per-comparison thread owns its prefix and fallback.
Early return exits only the predicate, never its surrounding CTA barrier path.
Inverse permutation and threshold/cutoffs are read-only device constants for the
process. Counters use independent global 64-bit atomics; correctness-only costs.
Reset once before each independent operator/probe. No overlapping configurations.
Role: existing 1024-thread CTA cooperatively stages query AoS and candidate SoA,
then existing comparison owners test assigned pairs and existing mask owners
write bits. Phase live state: FP32 prefix+addresses -> optional FP64 sum+inverse
index -> returned decision+counters -> existing mask update. No TMEM/MMA, warp
count, tiles, buffer ownership, async protocol or shared-memory size change.
Expected extra state <= 16 registers/thread; hard launch admission <=64 registers
for the default 1024-thread kernel, no unexpected stack/local traffic. Inspect
actual resources; a compilation pass does not waive resource or runtime gates.

## Fixed ladder and stopping rule

1. Freeze source/input/binary hashes. CPU rational cutoff checks and deterministic
   8192-pair D512 predicate probe: signed high entropy, near duplicate/cancellation,
   random float exponents, subnormals, zeros, exact boundary and G17 disputes.
   In one launch compare repaired predicate and strict original-order FP64;
   retain stages, prefix sums/dimensions, terminal results and cutoff receipts.
   Any false direct decision or final mismatch stops integration admission.
2. Original nine full inputs in manifest order: independent raw-mask decode,
   native G17 candidate-set identity, original-ID/reference equality, zero
   missing candidates, complete counter accounting. Stop on any failure.
3. Memcheck and synccheck on boundary_zero32 and CIFAR4096, four runs. Repeat
   CIFAR4096 and boundary_zero32 in two additional independent process orders;
   each process allocates fresh buffers. Standalone predicate pointer-churn
   stress: 1000 alternating-buffer invocations with stable complete results.
   This is bounded predicate stress, not sustained full-operator or 60K evidence.
4. NSYS default-kernel identification and selected SASS/resource audit; expect
   refinement changed, compression invariant. Inspect explicit F32/F64 rounding,
   no FTZ on predicate arithmetic and no unexpected Tensor Core instructions.
   OptiX driver-JIT identity is outside this gate.
5. Admit only bounded repaired-baseline correctness if 1-4 pass. Do not promote
   diagnostic timing. Next separately strip diagnostics, validate the new
   artifact, then freeze complete same-scope performance. If >=1% of real4096
   candidates require terminal FP64, flag the cheap-repair hypothesis as failed
   for this screen but retain a correct comparator; do not tune the bound to win.

Implementation, mechanism and thesis decisions must be recorded separately.
