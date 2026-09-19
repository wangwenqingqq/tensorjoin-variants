# Gate 0: Index-Aware Exact Tensor-Core Join

Date: 2026-09-03

## Conclusion

The broad combination **"GPU tree index + Tensor-Core similarity" is not a
safe novelty claim**. The nearest prior work already covers the two halves:
GTS provides exact batched GPU tree search in metric spaces, while prior
Tensor-Core work and FaSTED cover dense mixed-precision similarity joins.

A narrower mechanism remains worth a cheap kill test: use exact residual-error
certificates to execute a GTS-like pivot tree as regular Tensor-Core panels,
prune or bulk-accept whole leaves, and invoke the already validated exact
compact/refine kernel only on boundary panels. This is treated as a
high-extensibility systems mechanism, not as established novelty.

## Nearest prior art

1. [GTS (PACMMOD 2024)](https://arxiv.org/abs/2404.00966) is the closest tree
   system. It supports exact metric range and kNN queries, organizes pivot-tree
   nodes in levelwise list tables for GPU parallelism, and batches concurrent
   queries. Therefore GPU-resident tree traversal, pivot pruning, and batched
   range search are already claimed.
2. [Similarity Search with Tensor Core Units](https://arxiv.org/abs/2006.12608)
   already gives Tensor-Core algorithms for similarity join. Tensor Cores alone
   cannot be the contribution.
3. [FaSTED](https://github.com/bwcurless/FaSTED) implements an A100 FP16/FP32
   all-pairs Euclidean join with custom Tensor-Core tiling, but reports up to
   0.03% result loss rather than exact output. It is the nearest dense-kernel
   comparator, not evidence for exact certification.
4. [GPU-SJ](https://arxiv.org/abs/1809.09930) and the earlier
   [LSS GPU join](https://www.cs.umd.edu/~hjs/pubs/GPUicde2008.pdf) establish
   exact GPU index-supported similarity joins outside the Tensor-Core setting.
5. [High-Dimensional Similarity Joins](https://research.google/pubs/high-dimensional-similarity-joins-2/)
   and dual-tree BallTree search establish that hierarchical pruning for joins
   is longstanding.

## Surviving differentiator

The candidate contribution must be the **compositional exact execution
contract**, not the components in isolation:

- per-vector and per-pivot signed-INT8 representations with explicit residual
  errors;
- one regular query-by-pivot Tensor-Core panel that safely bounds all tree
  routing decisions;
- query groups gathered per fixed 64-vector leaf, permitting block-sparse
  Tensor-Core leaf panels rather than query-private point lists;
- whole-leaf reject/bulk-accept decisions from pivot interval bounds;
- exact FP64 refinement and compact output only for boundary pairs;
- a router that falls back when padding, pruning, or output selectivity makes
  the Tensor-Core path uneconomic.

This differs from GTS in arithmetic and execution granularity, and from
FaSTED in exactness and index pruning. The combination may still be judged an
obvious composition unless it yields a decisive same-contract system result
and exposes a nontrivial routing law.

## Cheap kill

Before another CUDA implementation, run `PROTOCOL_E0.md`. It grants the tree
exact FP64 pivot distances and excludes packing time, so it is an optimistic
ceiling. If even this ceiling cannot reduce padded Tensor-Core work enough on
both learned audio and learned video, stop the tree line. If it passes, E1 must
replace exact pivot distances with the same quantized certificate before any
performance timing.
