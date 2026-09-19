# Numerical scope: comparator admission, not unconditional certification

## Reference and data contract

Inputs are finite binary32, |x_k|<=1, D512. The output contract is agreement
with the retained G16 FP64 terminal and T25921/65536, not membership under
exact-real squared Euclidean distance. The previously retained threshold
counterexample remains valid. Neither small FP64 queues nor Tensor Core use is
claimed novel here; conventional low-precision filtering is the comparator.

## Explicit conditional interval

Let z be nearest FP16 reconstruction with half subnormals explicitly cleared,
r=x-z, and p the actual owned-cuBLAS FP32 dot output. GPU metadata encloses
||x_i||^2 in [a_i,b_i], and provides L_i>=||z_i||, E_i>=||r_i||.
The independently stated library premise is

    |p - z_i^T z_j| <= g L_i L_j + 2048*2^-126,
    g = 1026/(2^23-1026).

This premise is NOT derived from the presence of HMMA.16816.F32. Actual
FP16/FP32 Tensor Core reduction behavior is not fully specified by the API.
The coefficient is inherited before this implementation, not fitted to tests.

By bilinearity and Cauchy-Schwarz,

    |x_i^T x_j - p| <= B_ij
    B_ij = g L_i L_j + E_i L_j + L_i E_j + E_i E_j + 2048*2^-126.

With the inherited terminal allowance delta=2^-38*(b_i+b_j), the ideal
reference-enclosing interval is

    lower = max(0, a_i+a_j-2p-2B_ij-delta-pad),
    upper = b_i+b_j-2p+2B_ij+delta+pad,
    pad = FP32_up(1e-12).

The GPU uses FP32_up(g), directed-up additions/multiplications for every radius
term, directed-down lower-endpoint additions, and directed-up upper-endpoint
additions. `-2p` is exact for the admitted finite dot range. Thus directed
rounding can only widen the ideal interval, assuming ordinary non-FTZ FP32
instruction semantics. It accepts if upper<=T, rejects if lower>T, otherwise
invokes unchanged stage2/terminal. Finite checks test implementation of this
conditional statement, not all possible cuBLAS dot outputs.

## Metadata realization and proof boundary

FP32->FP16 round-to-nearest and explicit subnormal clearing define z. Original
norm, reconstructed norm, and residual norm squares are reduced in FP64 once
per vector. In the admitted range FP32 products fit FP64; for nonzero z,
Sterbenz cancellation makes x-z exact, while z=0 leaves x unchanged. Positive
reductions use an intentionally loose G=(512*2^-52)/(1-512*2^-52) radius,
1024*2^-1022 padding, a (1+8*2^-52) root multiplier, FP64 nextafter toward
+infinity, then FP32 outward conversion. Exact-rational saved rows test the
actual metadata. A source-level argument still requires checking actual emitted
reduction/conversion instructions and their round/FTZ behavior; it is not a
substitute for that audit or an independent complete theorem.

## Sources and limits

- NVIDIA PTX ISA, warp-level MMA numerical semantics:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
  FP16 products and FP32 accumulators do not imply a uniquely specified
  accumulation order, rounding mode, or subnormal treatment.
- NVIDIA cuBLAS math modes:
  https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t
  DISALLOW_REDUCED_PRECISION_REDUCTION is an execution configuration, not a
  complete roundoff theorem. Record installed library version130100 and full
  container hashes; online documentation can describe a newer toolkit.
- Khattak and Mikaitis, Accurate Models of NVIDIA Tensor Cores (2025):
  https://arxiv.org/html/2512.07004v1
  Tested architecture-specific models must not be transplanted to SM120;
  that paper's tested-GPU inventory does not establish this RTX model.
- Fasi, Higham, Mikaitis and Pranesh, Numerical Behavior of NVIDIA Tensor Cores:
  https://research.manchester.ac.uk/en/publications/numerical-behavior-of-nvidia-tensor-cores/
  Prior numerical analysis is nearest methodological context, not permission
  to claim a new universal certificate from engineering this comparator.

Current permitted status: tested fixed-reference comparator under an explicit
library numerical assumption. Do not write "unconditionally certified FP16",
"exact-real join", "new numerical theory", or a speed ratio from diagnostic
or profiler durations. A selected-kernel capture and finite adversarial checks
narrow an empirical risk; they do not eliminate the stated assumption.
