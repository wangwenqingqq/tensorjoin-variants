# Gate 0: Certified Low-Precision Exact Similarity Join

Date: 2026-09-02

## Decision first

The broad idea, "use Tensor Cores for similarity join," is already subsumed.
The surviving direction is a conditional extension rather than a clean-slate
novelty claim:

> Accelerate exact high-dimensional radius joins by evaluating a dense
> per-vector-quantized INT8 distance surrogate on Tensor Cores, certifying each
> pair as accepted, rejected, or ambiguous, and refining only ambiguous pairs
> in the original precision.

Proceed only if the real-data ambiguity rate is low and a same-contract GPU
implementation later beats a full-precision exhaustive join end to end.

## Frozen problem

- Input: a query relation `Q` and base relation `X`, each containing dense
  real-valued vectors of dimension `D`.
- Predicate: emit every pair `(i, j)` for which
  `||Q[i] - X[j]||_2^2 <= epsilon^2`.
- Contract: exact with respect to the stored FP32 feature vectors and a FP64
  distance oracle. Approximate recall is not admissible.
- First workload: disjoint-clip audio windows derived from ESC-50.
- Extension targets, only after the mechanism passes: video embeddings and
  scientific tensors under the same rectangular radius-join contract.

## Mechanism

For every vector `v`, symmetric per-vector INT8 quantization stores scale `s`,
code `z`, reconstruction `v_hat = s*z`, and residual norm
`e_v = ||v-v_hat||_2`. For a pair `(q, x)`, let
`d_hat = ||q_hat-x_hat||_2`. The triangle inequality gives:

```text
LB = max(0, d_hat - e_q - e_x)^2
UB = (d_hat + e_q + e_x)^2
LB <= ||q-x||_2^2 <= UB
```

- `UB <= epsilon^2`: accept without original-precision distance work.
- `LB > epsilon^2`: reject without original-precision distance work.
- Otherwise: refine the pair using the original FP32 vectors and FP64 oracle.

The INT8 dot product is shared by all three outcomes and maps to dense matrix
multiplication. Per-vector scale products are separable row/column metadata.

## Nearest prior art and kill result

| Work | What it establishes | Consequence for this project |
|---|---|---|
| Ahle and Silvestri, *Similarity Search with Tensor Core Units* | Tensor-Core algorithms for dimensionality reduction and similarity join | Kills the broad Tensor-Core join claim. |
| Gallet and Gowanlock, *Leveraging GPU Tensor Cores for Double Precision Euclidean Distance Calculations* | Exact FP64 Tensor-Core distance self-join, strongest in very low dimensions | Kills an "exact join on Tensor Cores" claim without a distinct precision/refinement mechanism. |
| FaSTED, *Fast Similarity Search via Tensor Cores* | High-dimensional Euclidean self-join using low-precision Tensor Cores and dense tiling | Kills low-precision dense Tensor-Core join as the thesis. Its reported output overlap is near, but not equal to, one. |
| RaBitQ | Quantized high-dimensional ANN with theoretical error bounds | Kills a generic "quantization with bounds" claim. It does not establish this exact radius-join execution contract. |
| Certified Cosine | Certificates for exact nearest-neighbor search | Kills a generic "certified exact vector search" claim. It does not establish the proposed Tensor-Core three-way radius join. |
| NVIDIA cuVS refinement | Quantized candidate generation followed by exact reranking is standard practice | Kills "quantize then refine" as a contribution by itself. |

**Gate-0 novelty status: conditional pass.** The claim must remain the coupled
execution mechanism and exact threshold-join contract. Broader wording is not
allowed. A closer prior work that already combines low-precision dense
Tensor-Core scoring, deterministic pairwise bounds, and exact output-sensitive
radius-join refinement would kill the direction.

## Cheap kill test A0

Before CUDA implementation, measure the certificate geometry on real ESC-50
audio windows. The predeclared gates are:

1. Zero containment violations after the declared numerical guard.
2. Zero pair-classification mismatch after exact refinement.
3. Ambiguous pairs are at most 5% of all pairs at every tested selectivity.
4. Median ambiguous fraction across tested selectivities is at most 1%.
5. The frozen INT8 accumulator cannot overflow for the tested dimension.

Failure of either exactness gate rejects the implementation. Failure of either
ambiguity gate rejects this quantizer/certificate as an efficient mechanism for
the frozen workload; it must not be rescued by writing a CUDA kernel first.

## Material caveats

- Exact refinement still requires access to the original vectors. INT8 codes
  reduce the hot scan footprint to about one quarter of FP32, but do not reduce
  total retained storage if original vectors remain resident.
- A handcrafted log-spectrogram tests numerical feasibility, not semantic audio
  retrieval quality. Learned audio/video embeddings are an extension gate.
- A mathematical triangle-inequality certificate and a bit-level GPU
  implementation certificate are different gates. A0 validates the former;
  later CUDA work must provide outward-rounded arithmetic or a conservative
  numerical pad, adversarial correctness, and sanitizer evidence.
- Low ambiguity is necessary but not sufficient. Promotion requires at least
  1.5x end-to-end speedup over a same-contract FP32 exhaustive GPU baseline on
  the target RTX PRO 6000 Blackwell GPU.

