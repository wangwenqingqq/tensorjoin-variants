# D1 Design: 2048-D Compact Certified Join

## Purpose

D1 tests whether C0B's end-to-end advantage survives a real pretrained audio
representation, doubled dimensionality, exact embedding ties, and a materially
larger output set.

## Data path

1. Load the fixed 10,000x2048 FP32 PANNs cache and reproduce D0's clip-disjoint
   512x4096 split.
2. Build the direct-difference FP64 oracle outside timing.
3. Place each radius at the midpoint between the requested rank distance and
   the next strictly greater distance, retaining all exact ties while keeping
   the decision boundary away from observed data.
4. Quantize each query and base vector independently to signed INT8, retaining
   FP32-upward-rounded residual norms for the GPU certificate.
5. Keep original FP32 vectors resident for exact FP64 refinement.

## Candidate path

The 64x64 scan tile accumulates 2048 products through 32 K=64 Tensor-Core
steps. Its epilogue computes padded lower/upper distance bounds. Direct accepts
and ambiguous pair IDs are appended to separate compact arrays; definite
rejects emit nothing. After reading the ambiguous count, one refinement program
per ambiguous pair sums direct squared differences in eight K=256 blocks using
FP64 and appends the exact accepts.

The 65,536-entry capacity covers both the observed D0 ambiguity and output
counts without changing the decision logic. Overflow is counted and rejects a
run rather than truncating silently.

## Keeper path

The keeper executes a non-TF32 FP32 GEMM over every pair. Values more than
`1e-3` below or above the threshold are accepted or rejected directly; all
boundary values are recomputed as direct FP64 squared differences. Output is a
GPU-resident compact ID tensor.

## Timing boundary

Both paths start with resident vectors and end with resident compact pair IDs.
Candidate timing includes the host synchronization needed to size refinement.
Oracle construction, quantization, allocation, and result validation are
outside the timed region for both paths.

## Known boundary

This remains exhaustive scanning at one query/base shape on one GPU. It does
not yet use the user's tree-indexing asset, compare to FaSTED or another
specialized implementation, or prove the FP32 certificate pad at bit level.
