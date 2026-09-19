# Precision-format routing: bounded screen v1 (2026-09-19)

Status at freeze: designed, not measured. This is an isolated prototype, not a
replacement for any archived TensorJoin implementation.

## Gate 0 and hypothesis

Generic adaptive precision, FP8-vs-INT8 comparisons, E3M4 format discovery and
Tensor Core similarity joins are prior art. Relevant sources:
- https://github.com/kacper-daftcode/blackwell-isa (SM120 E3M4 execution/encoding)
- https://arxiv.org/abs/2303.17951 (FP8 vs INT8)
- https://arxiv.org/abs/2309.14592 (FP8 format choices)
- https://thomasahle.com/papers/tcu.pdf (Tensor Core similarity search)
- TensorJoin G19 NEXT_RESEARCH_GATE and the qualified conventional FP16 control
  in the certified-exact-join archive. FP16 already has fewer Stage 1 ambiguities
  than INT8; a tiny FP64 queue is not specific to INT8.

Unresolved narrow question: does a format-conditioned reconstruction certificate
save enough exact refinement to repay native Stage 1, preparation and routing?
Symmetric null: per-row INT8 may have smaller residuals than E3M4, and conventional
FP16 may dominate both. No novelty or paper claim is admitted by this screen.

## Frozen workload and denominator

- One idle RTX PRO 6000 Blackwell Server Edition, SM120; CUDA 13.1 compiler,
  PyTorch 2.11.0+cu130 / Triton 3.6.0; clocks/power unchanged.
- GPU 1 selected after GPU 7 became occupied. Cooperative campaign locks plus
  occupancy supervision. Stop only our child if a foreign process appears.
- Input: contiguous pageable host FP32 N x 512, finite, abs(x) <= 1.
- Output: sorted, unique, directed uint64 i*N+j, including self. Reference:
  direct FP64 subtraction, squared distance, reduction with K blocks of 256;
  inclusive threshold. Not an exact-real predicate claim.
- Existing G17 fixtures: real/signed/onehot/zeros/boundary/cancellation/subnormal
  N=31 and CIFAR N=4096; preserve their thresholds and hashes. Additional
  synthetic exponent-spread/outlier and clustered controls are labeled synthetic.
- Full run: host input -> transfer -> allocation -> quantization/metadata ->
  dense Stage 1 -> compact FP32 queue -> compact FP64 queue -> CPU copy ->
  directed canonical IDs. Compilation and dataset disk I/O excluded. Allocation,
  dispatch, data-dependent host synchronization and output sorting INCLUDED.
- All methods use the same reference and output convention, reusable downstream
  FP32/FP64 kernels, and a shared conservative certificate construction. This is
  a controlled format experiment, NOT an optimized legacy INT8 vs optimized
  cuBLAS FP16 benchmark. The latter remains a required follow-up before promotion.
- Formats: signed INT8 with per-row max/127 scale; E3M4/E4M3FN/E5M2 with
  saturation-safe per-row power-of-two scale; FP16 and IEEE FP32 controls;
  direct FP64 reference. FP8 storage/MMA operands are distinct from INT8.
- FP8 producer rounds nearest, ties-even. E3M4 is an owned byte encoder plus
  a structurally checked private copy of an E4M3 cubin; only format bits change.
  Fail closed if native identity, codebook or static patch checks fail.
- The shared FP certificate uses original norm intervals, reconstruction
  residual norms and a conditional FP32 dot-accumulation envelope. This remains
  conditional on the stated hardware dot envelope, as did the prior FP16 control.

## Gates, sampling and stop rules

1. CPU encoding/codebook tests including ties, saturation and underflow.
2. Fresh documented E4M3/E5M2 controls; structural patch whitelist; all 256
   E3M4 codes on both operands and anti-fallback byte probes; high-entropy dots.
3. Exact canonical-ID equality to direct FP64 on every declared fixture;
   direct small-fixture CPU FP64 oracle too. Check count accounting/capacity.
4. Memcheck, racecheck, initcheck and synccheck on N=31/selected N=512 controls;
   16 repeated full cascades with pointer churn. Data-dependent CPU cascades do
   not expose Graph capture; this missing production gate is explicit.
5. Only then a bounded screen: 4 fresh processes with alternating forward/reverse
   method order, 2 untimed warmups and 5 measured repeats per method/fixture.
   Retain all raw observations. Summarize process medians, p10/p50/p90, and paired
   process log-ratios. A narrow screen win needs >1.10x median INT8 speedup in
   all 4 processes and exact IDs. This is NOT sufficient for production promotion.
6. Dynamic selection extension: sample at most 128 evenly spaced rows; measure
   candidate Stage 1 + refinements on the sample, extrapolate device work by
   pair count, choose a format. Count all sample work and all metadata preparation
   in the public run. A sample-driven global choice is not per-tile routing.
7. Reject the tested configuration if no queue reduction or full-cost benefit;
   retain it as a negative experiment, not an active production dispatch.
   Max N=4096 for this initial screen. Full 60K and broader external baselines
   remain explicitly untested, not silently implied by a subset result.

## Design card

Native INT8 IMMA (integer accumulation); FP8 QMMA m16n8k32, FP16 HMMA,
FP32 IEEE scalar/SIMT dot. Stage 1 uses 32x32 output tiles, K=64, four warps,
two compiler-managed stages; no TMA, TMEM, cluster, persistent state or custom
barrier protocol. All four warps load and compute their own output fragments.

| Phase | Owner/live state | Temporary | Ready / last use / overwrite |
|---|---|---|---|
| Metadata | one CTA per row, FP64 norm/residual reductions | encoded row, scale | input ready -> encode -> store; row temporaries die |
| Dot loop | four output-owner warps, 32x32 accumulator | A/B K64 fragments | staged loads -> compiler wait -> MMA -> next stage reuse |
| Dot epilogue | same output owners | row scales | K loop done -> scaled dot store; accumulator dies |
| Classify | one CTA per flat 1024 pairs | directed intervals + prefix sums | dot store -> same-stream launch -> IDs/counters |
| Refine | one CTA per compact pair | K256 FP32 or FP64 reductions | CPU count read -> launch -> output atomic append |
| Canonicalize | host | mirrored non-self IDs | D2H complete -> sort -> return |

Accumulator lower bound is 8 FP32 registers/thread for 32x32/128; operands and
address state add to it. Staged A/B storage is about 2*64*(32+32)*dtype_bytes,
plus compiler padding. Inspect compiled registers/shared/spills and SASS before
acceptance. No register caps or dynamic reallocation. Residual metadata adds
FP64 work O(ND), while dense work is O(N^2 D); sparse refinement costs scale
with ambiguous pairs. Changing formats preserves pair coverage and downstream
kernels but changes quantization, native instruction geometry, residual error,
and the charged FP32 dot envelope. This coupled numerical contract is explicit.

## Evidence states

Native format identity, conditional certificate, exact-ID tests, sanitizer,
stress, diagnostic timing and public timing are separate gates. Missing evidence
is unknown/unvalidated, not a universal rejection. No old campaign time is reused.

## Pre-timing implementation clarification (v1a)

The first diagnostic probe used Triton's bundled ptxas-blackwell 12.9 despite
system nvcc being 13.1. Retained labels `probe_r1`, `probe_cuda131` and
`validate_r1` are NOT CUDA 13.1 compilation evidence. The experiment now asserts
compiler version 13.1 and the guard sets both Triton compiler overrides. Rerun
labels `probe_final131` and later establish the pinned compiler scope.

The global router prices sample metadata, dense classify/dot and downstream
refinement using ten fixed CUDA Graph replays per component. It extrapolates
O(ND) preparation and O(N^2 D) pair work, and uses sampled ambiguous fractions.
Every calibration, allocation, sample transfer, graph setup/replay, full chosen
preparation and final output remains inside the public router wall time. This
is intentionally a transparent cold-selector prototype, not an amortized
production policy or an oracle selector. FP64 is also a selectable/direct control.
No full data-dependent cascade is Graph captured; only fixed calibration components.

## Pre-timing threshold ABI repair (v1b)

A source review found a scalar-ABI hazard in the initial prototype: a Python
float runtime argument is lowered to FP32 by Triton. Some G17 thresholds are
binary64, not exactly representable as binary32. Old `*_final131` labels only
establish the earlier implementation, not this final predicate. They are
retained as diagnostic records and cannot qualify the repaired implementation.

The owned terminal retains the original FP64 subtraction/reduction but uses an
explicit FP64 constexpr threshold. Stage 1/2 compare FP32 interval endpoints
against floor_FP32(T); since no FP32 endpoint lies strictly between this floor
and T, their inclusive accept / strict reject decisions are identical to
comparing those endpoints against the full binary64 T. Three additional
31-row fixtures test an exact one-coordinate FP64 squared distance at T and
at both adjacent binary64 thresholds. Re-run validation/sanitizers/stress on
this source before any admitted timing. No requested output semantics changed.

Producer admission details: the FP16 control flushes encoded half subnormals
(abs < 2^-14), as did the qualified existing control; the IEEE FP32 dot control
flushes input FP32 subnormals (abs < 2^-126) during preparation. Their effects
are included in residual certificates; neither is a preserve-subnormals control.
INT8 and FP8 use unit scale for rows with max abs < 2^-100 to avoid a denormal
scale ABI; resulting reconstruction loss is also charged to the residual.

Final harness clarification, before admitted timing: stress now varies aligned
GPU input offsets over 16 positions and asserts at least 16 distinct actual
input addresses; fresh allocation alone is not treated as proof of pointer
churn. The benchmark resolves every full-shape candidate before its two
untimed warmups, so reverse order cannot accidentally charge a newly selected
route's first full-shape JIT/load to a measured repetition. Kernel arithmetic,
format producers and certificate parameters are unchanged by these harness fixes.
