# FP16 dense repair: bounded novelty counterfactual and numerical census

Experiment: tensorjoin_20260907_fp16_dense_repair_census_d512.
Frozen before implementation and measurement. No performance promotion,
paper edit, parameter search, or full120-panel operator campaign is authorized
by this record. Preserve the prior1.518533x FP32 comparison unchanged.

## Gate0 and question

FaSTED (ICPP2025) already implements FP16/FP32 dense TC distance self-join;
general robust filters, reconstruction error bounds and adaptive precision are
established mechanisms. No single inspected source is asserted to duplicate
the entire G16 pipeline. Generic TC-scan/quantize-bound-refine novelty remains
unsupported, while the value of a competitive same-contract FP16 control is
unmeasured. A bounded diagnostic may resolve that gap without claiming novelty.

Question: does a conventional FP16 reconstruction plus a conservative dot-error
interval and the *unchanged* FP32/FP64 cascade remove the same numerical work
that the INT8 path avoids? Count both intermediate and final uncertainty,
not just output size. A small FP16 uncertainty set would undermine any claim
that INT8 or its bespoke certificate uniquely enables sparse refinement, but
does not predict complete latency or establish an end-to-end duplicate.

## Frozen paths and exactness

A: the original four G16 cubins, original constants, N60000,D512,T25921/65536.
C: explicit FP32->FP16 round-to-nearest, then set FP16 subnormals to zero;
FP16 GemmEx with FP32 output/accumulation (compute68, math16, alpha1,beta0,
default algorithm-1); original GPU FP64 norm metadata on x, z and r=x-z;
CPU FP64 interval classification; original stage2 and terminal on ambiguous IDs.
The diagnostic CPU classifier is intentionally not a competitive timed operator.
cuBLAS FP16 mode is TC-eligible; actual selected Tensor-Core instructions and
library-internal numerical behavior are separate unresolved admission gates.

The fixed terminal is the reference, not exact-real arithmetic or arbitrary
FP64 reductions. All direct decisions and complete upper-ID sets must match
an exhaustive invocation of that same terminal over each selected panel.
No candidate is admitted just because its approximate hits can be reranked:
rejected pairs are included in the exhaustive check.

## Numerical proposal (conditional, not a completed certificate)

Let Lz_i and E_i bound ||z_i|| and ||x_i-z_i|| from retained norm metadata;
nx_i,nr_i enclose ||x_i||^2. Normal FP16 reconstruction differs from x by an
exact FP32 subtraction under Sterbenz; the explicit zero case also subtracts
exactly. Verify stored subtraction and selected metadata with exact rationals.
Use u=2^-23 (not the RN unit), k=2D+2 and g=ku/(1-ku). Under an admitted
at-least-FP32 dot accumulation model,

    b = g*Lz_i*Lz_j + E_i*Lz_j + Lz_i*E_j + E_i*E_j + 4D*2^-126
    c = nx_i+nx_j-2*p
    m = |nx_i|+|nx_j|+2|p|+nr_i+nr_j+2*b
    R = nr_i+nr_j+2*b+32*2^-52*m+1e-12
        +2^-38*(|nx_i|+|nx_j|+nr_i+nr_j)
    lower=max(c-R,0), upper=c+R

Accept if upper<=T, reject if lower>T, otherwise refine. The old control's
terminal allowance and FP64-evaluation padding are retained; no post-failure
widening. The proposal is not promoted as a portable library-level proof:
PTX leaves MMA rounding/order unspecified and a selected-runtime audit remains
necessary. Finite successes are only diagnostic evidence.

## Workload and stop rule

One fixture pass: first512 rows of a padded N60000 FP32 matrix, containing fixed
seed high-entropy values, duplicates, zero, signed extremes, half-normal and
half-subnormal conversion boundaries, the fixed-terminal/exact-real witness,
and exact/adjacent threshold cases. Exhaustive upper512-square comparison.

Three public panels from the already retained NPY, selected before observations:
4096x4096 at(0,0),4096x2656 at(0,57344),2656x2656 at(57344,57344).
These are22,798,128 unique upper pairs, not the full1,800,030,000-pair workload.
Every panel compares A/C with exhaustive fixed-terminal output and records
stage counts and hashes; collect metadata rational checks for selected rows.

Two distinct device-input allocations on the fixture and one public process;
no speed estimator or latency headline. Reject on any wrong direct decision,
output mismatch, metadata bound failure, capacity violation, identity drift,
foreign occupancy or nonzero exit. Stop first failed process and retain it;
no parameter replacement. A code-only mistake may be repaired in a separately
recorded revision, never silently relabeled. No full baseline admission follows.

## Host, resources, lifecycle

Original gpu-host-8 GPU2 UUID in targets; driver590.48.01; existing torch2.11cu130
environment. Two per-GPU locks,30s quiescence,4GiB peak device-used guard,
30-minute timeout; do not touch the two observed VLLM processes on GPU7.
No custom CUDA kernel or shared-memory synchronization change: retained metadata
and cascade geometry, library-owned GEMM; CPU owns interval arrays/compaction.
Ready graph: input/cast/residual -> GPU metadata/GEMM -> blocking D2H -> CPU
classification -> same-stream stage2 -> counters -> fixed terminal -> host IDs.
Buffers are overwritten only after blocking reads/consumer completion. Separate
half/float views and input allocations are retained until all consumers finish.

Expected effect: narrower FP16 reconstruction than INT8 may reduce ambiguity;
added costs include FP16 conversion, three metadata passes, dense output,
interval arithmetic and queues. The diagnostic deliberately does not estimate
those costs. Proof/SASS/sanitizer/stress/full timed admission are later gates.
