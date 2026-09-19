# Gate 0 H0: Exact Multi-Vector Similarity Join

Date: 2026-09-03

## Decision

**Conditional novelty pass for a cheap structural kill test; no paper-level
novelty claim is admitted yet.**  A 2026 vector-query survey treats
multi-vector similarity search and similarity join as separate query families,
and the literature search below did not find an exact all-pairs join over
variable-cardinality dense-vector objects under Chamfer/MaxSim semantics.
That gap is more defensible than adding another conventional tree to the
single-vector operator.

The broad ingredients are not novel: multi-vector retrieval, exact metric and
set joins, Tensor-Core similarity joins, and error-bounded quantization all
exist.  The surviving thesis candidate is narrower: **lift certified
pairwise low-precision intervals through the non-linear min/max-and-sum
aggregation of a multi-vector predicate, and refine only object pairs and token
cells that can still change the exact join decision.**

## Problem contract

Each database object is a variable-cardinality set of normalized dense vectors.
For object sets `A` and `B`, use symmetric Chamfer squared distance:

```text
C(A,B) = 0.5 * (
    mean_{a in A} min_{b in B} ||a-b||^2
  + mean_{b in B} min_{a in A} ||a-b||^2)
```

For unit vectors this is the distance form of bidirectional MaxSim.  The join
returns every object pair whose `C(A,B) <= tau`, with exact deterministic
semantics over the stored float32 vectors and an FP64 direct-difference oracle.

## Nearest prior art and kill boundaries

| Work | Established result | Consequence for H0 |
|---|---|---|
| Xie, Liu, and Yu, *A Survey on Query Processing in Vector Databases* (2026) | Separately taxonomizes multi-vector search and similarity join; describes the former as query-to-corpus retrieval and the latter over single vectors. | Supports a problem-level gap, but a survey omission is not proof of novelty. |
| MUVERA, NeurIPS 2024 | Approximates Chamfer/MaxSim with fixed-dimensional encodings for query-to-document retrieval, followed by candidate generation/reranking. | Kills any claim that MaxSim indexing or fixed-vector reduction is new. It does not provide an exact all-pairs threshold join. |
| PLAID / EMVB / IGP / GEM | Centroid, bit-vector, PQ, or graph acceleration for multi-vector top-k retrieval; GEM is a native multi-vector graph index. | Kills a generic multi-vector index contribution. H0 must be exact and join-native, not another approximate retrieval index. |
| AMES (2026) | Approximate multimodal candidate generation followed by accelerator-optimized exact MaxSim reranking. | Kills “first accelerator MaxSim” and “first multimodal late-interaction system.” The end-to-end candidate path is approximate and query-to-corpus. |
| Leybovich and Shmueli (2021) | Exact reduction and approximate search for sets of vectors under a max-plus-average cosine measure. | Kills “first vector-set search” and requires a same-semantics distinction. It is not an all-pairs threshold join and does not expose certificate-driven cell refinement. |
| Jacox and Samet, *Metric Space Similarity Joins* (2008) | Exact similarity joins for arbitrary metric objects; explicitly notes Hausdorff distance for shapes/images. | Kills “first exact join over set-valued objects.” Symmetric Chamfer/MaxSim is non-metric, so H0 must claim the exact non-linear aggregate and execution mechanism, not set-valued joins generally. |
| Exact set-similarity joins | Exact filter/verify joins for discrete token sets under Jaccard/Dice/cosine-like overlap. | Kills a generic set-join framing; these are not dense vector-set Chamfer joins. |
| Tensor-Core similarity join and TensorJoin G4C | Dense low-precision similarity computation and the existing exact single-vector certificate router. | Kills hardware-only novelty. H0 must show a new certificate algebra and decisive object/cell-level work elimination. |

## Candidate mechanism

For every token pair `(a,b)`, the existing quantizer gives a certified interval
`L_ab <= ||a-b||^2 <= U_ab`.  Monotonicity gives object-level bounds:

```text
mean_a min_b L_ab <= mean_a min_b d_ab <= mean_a min_b U_ab
```

and analogously in the reverse direction.  Their symmetric average is a safe
interval `[L(A,B), U(A,B)]` for the exact object score.

- Bulk accept an object pair if `U(A,B) <= tau`.
- Bulk reject it if `L(A,B) > tau`.
- Refine only ambiguous object pairs.
- Within an ambiguous directed row, a token cell cannot attain the minimum if
  its lower bound exceeds the best available upper bound in that row.  Refine
  only the union of forward- and reverse-competitive cells.
- Route variable-cardinality object pairs by padding cost, ambiguity, and
  output density; Tensor-Core tiles are the physical unit, not a point-wise
  graph traversal.

The interval lifting itself is elementary.  A publishable contribution requires
the combination to produce a nontrivial exact execution law and a decisive
same-contract system result.  Merely implementing batched MaxSim is not enough.

## Cheap novelty and mechanism kill

Run `PROTOCOL_H0_MULTIVECTOR_CERTIFICATE.md` on two existing public-derived
caches:

1. ESC-50/PANNs: five one-second embeddings per audio clip.
2. UCF101/R3D-18: four to seven clip embeddings per source group.

Stop the direction before GPU work if either workload violates exactness or if
the object-level certificate/refinement gate fails.  A pass admits only an H1
GPU design and a deeper forward/backward citation audit.  It does not admit
“first exact multi-vector join” wording.

## Novelty status

- **Established:** all broad components above.
- **Inferred:** exact join-native interval lifting for variable-cardinality
  Chamfer objects is not represented in the inspected closest work.
- **Unknown:** whether unindexed or unpublished work subsumes the mechanism.
- **Required next evidence:** H0 structural selectivity, then exact external
  baselines and a citation audit centered on dense set-to-set joins.

## Primary sources checked

- <https://xiejiadong.github.io/files/paper/vector_survey.pdf>
- <https://papers.neurips.cc/paper_files/paper/2024/file/b71cfefae46909178603b5bc6c11d3ae-Paper-Conference.pdf>
- <https://arxiv.org/abs/2603.20336>
- <https://arxiv.org/abs/2404.02805>
- <https://arxiv.org/abs/2107.06817>
- <https://doi.org/10.1145/1366102.1366104>
- <https://arxiv.org/abs/2603.13537>

