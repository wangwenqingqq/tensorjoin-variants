# G11: Nearest-Prior-Art Ledger

Checked 2026-09-04. This is an ideation audit, not a manuscript contribution
claim. Missing coverage in this bounded search is uncertainty, not novelty.

## 1. TigerVector: graph-pattern vector joins already exist

- Primary paper: https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD25_TigerVector.pdf
- Published at SIGMOD Companion 2025; DOI 10.1145/3722212.3724456.
- Section 5.4 describes top-k endpoint-pair similarity over graph patterns.
  Its stated execution enumerates matching paths, computes similarities, and
  maintains a global heap. Section 5.3 also treats graph patterns as filters.
- Consequence: neither graph-plus-vector composition nor graph-pattern vector
  join is a new problem. A possible distinction must concern repeated endpoint
  evaluation and intermediate construction, not changing the output from top-k
  to a threshold and calling the query novel.
- Boundary: the paper's described operator is not an admitted same-contract
  threshold-join benchmark. No TigerVector runtime comparison was performed.

## 2. VecFlow: GPU filtering and posting-list scans already exist

- Primary text inspected: https://arxiv.org/html/2506.00812v1
- Author project: https://supercomputing-system-ai-lab.github.io/projects/vecflow/
- Sections 4.1–4.3 give label-centric IVF, graph/brute-force routing, memory
  layouts, and AND/OR filtering. The low-specificity scan assigns a query and
  posting list to a block. Section 4.2.2 explicitly discusses dense bitmap
  filtering versus CSR masked matrix multiplication and their costs.
- Consequence: predicate-first execution, label grouping, avoiding irrelevant
  distances, and GPU-friendly scans are not new. A stronger control must use
  deduplicated eligible endpoints, not only a wasteful dense post-filter.
- Boundary: the inspected version targets filtered ANNS with recall metrics;
  it must not be mislabeled an admitted exact full-join comparator. The census
  does not measure its code or infer that all current versions lack batching.

## 3. Kùzu: preserving factorization is established

- Primary paper: https://mail.vldb.org/cidrdb/papers/2023/p48-jin.pdf
- Section 3 represents intermediate relations by Cartesian products of vector
  groups and develops joins that preserve this representation. This avoids
  repeated flat tuples in many-to-many joins.
- Consequence: keeping endpoint lists rather than expanding their product is
  not itself a contribution. Our factor-payload counts describe a familiar
  representation and cannot be advertised as a new compression result.
- Unresolved: an overlapping existential projection followed by a dense-vector
  predicate needs a precise duplicate-free execution contract. A generic
  factorization argument alone neither solves this nor proves it impossible.

## 4. TCUDB: relational joins on Tensor Cores are established

- Primary paper: https://arxiv.org/pdf/2112.07552
- Section 3 translates relational operators to matrix operations, supports
  multiway compositions, and discusses comparison-based non-equijoins.
- Consequence: SQL-to-GEMM, GPU join acceleration, and a matrix-operator
  optimizer are not new claims. A different consumer or larger dataset does
  not erase this prior art.
- Boundary: this audit does not claim TCUDB implements the complete proposed
  graph-restricted vector predicate. Its role is mechanism-overlap evidence.

## 5. FlashMask: interval masks and block skipping are established

- ICLR 2025 primary paper:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/44c624d8f4e802453ad12593b2aa8e38-Paper-Conference.pdf
- Section 4 uses a compact column-interval mask representation and block-level
  classification to avoid fully masked score tiles.
- Consequence (inference): sorting scalar metadata into ranges and skipping
  empty dot-product tiles is too close to established machinery to be our
  standalone new mechanism. Attention and threshold joins have different
  consumers; this is a mechanism-overlap finding, not an end-to-end equivalence.
- Also inspected: PyTorch BlockMask documentation at
  https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html .
  A block-mask baseline must not be credited to this project.

## 6. Beyond Equi-joins: factorized range predicates are established

- PVLDB 2021 primary paper:
  https://www.vldb.org/pvldb/vol14/p2599-tziavelis.pdf
- Introduces tuple-level factorization graphs and compact representations for
  inequality/band predicates and combinations, supporting ranked enumeration
  under the paper's ranking and query restrictions.
- Consequence: a segment tree or inequality-based factorized representation is
  not a sufficient new database mechanism.
- Boundary: do not extend its ranking guarantees to an arbitrary dense-vector
  distance across endpoints. Full queries, projected set queries, and bag
  queries have different conditions; no general impossibility is inferred.

## 7. Other routes screened, not selected for implementation

| Route | Evidence / nearest source | Decision and limitation |
|---|---|---|
| Consumer-specific DBSCAN/connectivity without exporting all neighbors | https://arxiv.org/abs/2103.05162 ; GPU tree-based DBSCAN | Generic consumer pruning is already developed. No differentiated new mechanism identified here. This is not proof that every high-D variant is subsumed. |
| Maintain a join across evolving embeddings using small residual movement | https://arxiv.org/abs/2105.01818 ; Dynamic Enumeration of Similarity Joins | Dynamic joins are established. Factoring out a global rotation is only an untested possible distinction; no real multi-version embedding trace was located or measured. Park, do not call the hypothesis false. |
| Ordinary geometric tree plus TC precision filtering | Local E0, H2B, G5, G8/G9, G10 | Existing negative evidence and novelty boundaries remain. Do not revive via data breadth, naming, or another leaf/fanout sweep. |

## 8. Public-data provenance

- OGB dataset documentation: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
- Official download routing:
  https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv
- `ogbn-arxiv` supplies directed citations and 128D averaged word embeddings,
  with years and node IDs. The public license is ODC-BY; retain attribution.
- The G11 census uses real citations and deterministic sampled endpoint sets.
  Its two-hop join queries are researcher-constructed; they are not OGB's
  official prediction task or a production log. Vectors are not evaluated.

## Gate-0 conclusion before the census

No unconditional novelty pass. The one remaining bounded question is whether
real overlapping relationship witnesses provide enough redundant and dense
work to justify designing a new duplicate-free, factor-preserving vector
operator. G11 only measures that premise. Even a positive census cannot turn
known grouping, masking, or factorization into a contribution.
