# H2B Design Card: Pedantic cuBLAS Exact-Baseline and Streaming Gate

Date: 2026-09-03

## Decision question

Does the unchanged H1-R1 exact INT8 object-first candidate remain faster than
the fastest admitted **pedantic cuBLAS SGEMM plus certified FP64 repair** keeper,
first on the four H2A cells and then on larger disjoint streamed object
matrices?

H2B is a kill test, not an optimization exercise for the candidate.  Failure
closes the upper performance thesis before any tree/index work.

## Why this comparator is materially stronger

The H2A keeper computes every coordinate difference and reduction in a custom
FP32 Triton kernel.  H2B instead obtains all token dot products from
`cublasGemmEx` with both `CUBLAS_PEDANTIC_MATH` and
`CUBLAS_COMPUTE_32F_PEDANTIC`, then reconstructs squared-distance intervals
using exact-FP64-preprocessed token norms.  NVIDIA documents pedantic math as
using the prescribed precision and standardized arithmetic, and the pedantic
compute type as disabling reduced algorithmic shortcuts.  The relevant API
contracts are:

- <https://docs.nvidia.com/cuda/cublas/#cublascompute-type-t>
- <https://docs.nvidia.com/cuda/cublas/#cublasmath-t>
- <https://docs.nvidia.com/cuda/cublas/#cublassetmathmode>

The implementation calls cuBLAS directly rather than inferring the math mode
from a high-level framework flag.  The selected runtime kernel must later be
resolved with Nsight tooling; an API setting alone is not generated-code
evidence.

## Numerical hypothesis to falsify

For finite normalized FP32 token vectors `x,y` of dimension `D`, let
`p_hat` be the FP32 result returned by the pedantic SGEMM and let exact
preprocessed FP64 norms be `nx=sum(x_i^2)` and `ny=sum(y_i^2)`.  Define

```text
u       = 2^-24
k       = 2D + 2
gamma_k = k*u / (1-k*u)
Bxy     = sqrt(nx*ny)
r_dot   = gamma_k*Bxy + 4D*tiny32
c       = nx + ny - 2*p_hat              # evaluated in FP64
r_d2    = 2*r_dot + 32*eps64*(nx+ny+2*abs(p_hat)) + 1e-12
[L,U]   = [max(c-r_d2,0), c+r_d2]
```

`gamma_k` deliberately budgets separate FP32 products plus an arbitrary
addition path rather than assuming a particular FMA tree.  `4D*tiny32`
budgets flushed subnormal products/partials on the frozen normalized domain;
the FP64 term and absolute guard cover interval reconstruction.  This is a
design-time bound, not yet an accepted certificate.  Admission requires:

1. resolving the actual cuBLAS kernel and confirming no TF32 or lower-precision
   MMA path;
2. exhaustive actual-data token/object containment;
3. same-shape adversarial/metamorphic containment, including cancellation,
   zeros, scale, signs, and ragged tails;
4. rejecting rather than widening the formula if any frozen gate fails.

The CUDA floating-point/FMA behavior used by this hypothesis follows the
official programming guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>.

## Phase P: frozen four-cell proof and performance contract

- Host/device: `gpu-host-8`, physical GPU3, RTX PRO 6000 Blackwell Server
  Edition, SM120; GPU3 lock plus foreign-PID refusal before every GPU process.
- Data, split, normalization, object semantics, target-1/8 thresholds, and
  canonical IDs: byte-for-byte H2A dependencies.
- Keeper preprocessing outside timing: exact FP64 token squared norms and
  norm square roots, device allocation, compilation, and cuBLAS handle setup.
- Keeper timed path: pedantic FP32 SGEMM dot matrix -> fused object interval
  aggregation/compaction -> charged dynamic ambiguity count -> unchanged H1
  direct-FP64 panels -> final count/IDs -> host canonical sort.
- Candidate: byte-for-byte accepted H1-R1 code and preprocessing boundary.
- Screen denominator: outer monotonic wall time, identical included/excluded
  work to H2A except that the keeper's dot product comes from cuBLAS.

## Phase S: larger streamed contract

Phase S is frozen only after Phase P passes.  It must use larger disjoint
objects from the same public caches, deterministic object-level splits, and
base-object chunks that prevent an uncharged full token-pair materialization.
Both keeper and candidate must charge:

- every streamed token panel and object reduction;
- dynamic ambiguity discovery and exact repair;
- per-chunk scheduling/synchronization;
- final canonical ID transfer and sort.

Allocation, one-time feature normalization, candidate quantization, keeper
norm preprocessing, and compilation remain outside the operator denominator
for both variants.  The exact object counts, chunk width, thresholds, memory
cap, and output-capacity proof must be written into a separate Phase-S protocol
before any scale timing.

## Gate ladder

1. **P0 host opportunity:** ideal-centered theoretical intervals on all four
   H2A cells; record ambiguity without using it as correctness evidence.
2. **P1 API/correctness:** direct cuBLAS status checks, queried pedantic math
   mode, exhaustive actual-data containment, exact IDs, and adversarial cases.
3. **P2 generated code:** runtime-selected cuBLAS function/kernel identity,
   library/container hashes, normalized selected-function SASS hash, resources,
   and zero TF32/lower-precision MMA evidence.
4. **P3 safety/stress:** actual-data memcheck, synchronization checks where
   applicable, output stability, and long-loop count/hash stability.
5. **P4 timing:** direction-balanced fresh-process outer-wall A/B plus a
   sustained sequence; preserve every observation and process order.
6. **S0--S4 scale:** freeze and repeat the same ladder with charged streaming.

No timing result can promote before P1--P3.  Profiler duration and SASS are
mechanism evidence only; public latency comes from the declared outer wall.

## Promotion and stop rules

- Phase P passes only if all exactness/safety/generated-code gates pass and the
  candidate's predeclared process-paired lower confidence bound exceeds 1.25x
  in every cell, with all fresh-process medians and sustained sequences won.
- Phase S passes only if the same exactness gates hold and the aggregate
  candidate lower confidence bound exceeds 1.25x without any cell below 1.10x.
- Any exactness failure rejects the implementation.  Any performance failure
  against the fastest admitted pedantic keeper closes the upper performance
  thesis; broader packaging or a tree cannot rescue it.
- Tree/index work is admitted only after both Phase P and Phase S pass.

## Immutable dependencies

- H1 runner/kernel SHA-256:
  `11aaa72447266fef26286d80cbbe0bd22bc1626af437c0c9a0b121352600713e` /
  `c24d6e94b81dc94f8ad50699eadac82a8a593e175c519a8d704947ab8e5db0db`.
- H2A correctness/safety/timing SHA-256:
  `d773c3cf08578af1b4845d5dc3bdddced4138f167659e24c7bf25eb3a3b5c9be` /
  `d10f7dab881175cb742325104d1f93479a0c76aacf5f107a5768505b42e39da4` /
  `a97efa258375b3bc87949fc6a0efaf15fb863958dc8d099aa6f1344952cfdcb4`.

## Current state

P0 opportunity, P1 correctness, P2 selected-function precision, and P3-R3
bounded access/stress gates passed.  P4 then rejected Phase P in its first
fresh process: the candidate failed the 1.25x gate in all four cells and was
slower in both target-8 cells.  The frozen stop rule blocks Phase S streaming
and tree/index work.  No positive H2B performance claim is admitted.
