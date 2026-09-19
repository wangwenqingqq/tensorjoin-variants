# G11: Relational Eligibility Census (CPU-only)

Frozen: 2026-09-04, before downloading or inspecting the dataset.

## Purpose and novelty boundary

Test a necessary workload premise, not a GPU performance claim or a novelty
pass. The tentative target is a vector threshold join restricted by an
existential graph relationship. Multiple graph witnesses can name the same
endpoint pair. A useful new mechanism would have to preserve compact
eligibility while preventing repeated vector evaluation, without first
materializing every eligible pair.

TigerVector already supports graph-pattern vector joins. Factorized execution,
theta-join factorization, GPU filtered vector search, relational Tensor-Core
joins, and interval/block mask skipping already exist. Merely composing these
does not pass Gate 0. No GPU implementation or paper drafting is admitted here.
The census is a cheap ideation falsification test; a novel algorithm is not
claimed to have been implemented.

## Frozen public data and sampling

- Source: OGB `ogbn-arxiv`, official version 1 archive:
  https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
- Attribution and description: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
  (ODC-BY). The source supplies real directed citations, years, and 128D
  averaged word embeddings. This census uses citations only.
- Expected published inventory: 169343 nodes and 1166243 edges. Audit actual
  archive shape, ranges, duplicate edges, and hashes before proceeding.
- Independent NumPy PCG64 seeds: 20260904 and 20260905. For each seed, permute
  all node IDs, take 4096 query endpoints, then 16384 disjoint database
  endpoints. Keep all possible witness nodes, not just selected endpoints.
- Two predicates, each with existential/set semantics:
  1. common_reference: both endpoints cite the same paper;
  2. common_citer: the same paper cites both endpoints.
- These are constructed research query workloads on real data, NOT observed
  production query logs and NOT the official OGB node-classification task.
- No class/year filtering, embedding changes, favorable degree sampling,
  radius tuning, or dropping empty-query rows.

## Quantities and comparators

For witness w, Qw and Bw are selected incident endpoint sets. Record:

- W = sum_w |Qw| |Bw|: witness-expanded pair occurrences.
- E = |union_w Qw x Bw|: unique eligible endpoint pairs.
- W/E: maximum dot-work reduction from removing witness duplicates alone,
  relative to a witness-expanded evaluator, NOT relative to a deduplicated
  keeper and NOT a wall-time speedup.
- Compact factor payload: int32 endpoint IDs and 24-byte metadata per active
  witness. Charge the original graph separately; this is a representation
  size, not measured GPU memory or zero-cost construction.
- Unique CSR payload: 4E + 8(Q+1) bytes. Flat uint64 pair list: 8E bytes.
- Literal witness-panel physical work for 16x16 and 64x64 tiles:
  sum_w ceil(|Qw|/t) ceil(|Bw|/t) t^2, including padding and duplicate pairs.
- A simple duplicate-free fixed-layout block-mask control: evaluate each
  active t x t block once, including noneligible elements. Mask construction
  is NOT free. Do not present this control as cuVS/FlashMask performance.
- Exact equal-neighborhood row grouping is another standard control. It
  requires discovering neighborhoods; charge construction in any later run.
- Fat-factor coverage: unique eligible pairs covered by at least one witness
  having >=16 query and >=16 database endpoints.

## Predeclared opportunity rule (not a performance gate)

Only retain a predicate for further algorithm design when BOTH independent
seeds satisfy all three: W/E >=1.5, factor payload <=25% of unique CSR payload,
and fat-factor coverage >=50% of E. This is a conservative research-budget
screen for this duplicate-removal thesis, not an impossibility theorem. Keep
all four cells and all padding regressions regardless of outcome. If no
predicate passes, stop this workload route without GPU tuning or favorable
resampling. If a predicate passes, novelty and a complete duplicate-free
panel construction remain unresolved; do not authorize a GPU campaign.

## Correctness and resource contract

- Local Mac, `/usr/bin/python3`, NumPy; no GPU, driver, CUDA, SASS, sanitizer,
  clock locking, or library engine is involved (all N/A).
- Freeze and hash this protocol before implementation. Log machine/OS,
  Python/NumPy versions, source/data/protocol hashes and wall-clock diagnostics.
- Deduplicate directed input edges explicitly and report the count removed.
- Independently verify each row's eligible endpoints by adjacency-list unions
  against the witness-rectangle count matrix. Verify sum(count matrix)=W.
- The dense count matrix and full pair scans are CPU audit oracles, NOT an
  alleged efficient candidate implementation. Counts are uint32, with an
  input-degree bound checked before accumulation.
- Maximum 4 sequential cells, one archive download (~83 MB), no remote/GPU
  process and no model training. Fail on provenance/correctness/resource error;
  do not silently shrink dimensions. Do not rank variants by Python runtime.

## Later acceptance requirements, only if this screen survives

1. A differentiated ownership/coverage algorithm, audited against factorized
   databases, graph joins, set joins, sparse matrix kernels, and block masks.
2. Exact set semantics; no duplicate outputs, missed pairs, or hidden path
   expansion. Freeze numeric predicate separately; G10's FP64-versus-exact-real
   distinction still applies.
3. Compare complete vector pipelines to a deduplicated CSR/gather keeper,
   factorized grouped scan, and masked dense/block GEMM; ANN methods are only
   recall-matched context, not automatically exact comparators.
4. Charge graph filtering, factor construction, ownership, packing, vector
   reads, numeric refinement, dynamic output allocation, deduplication, and
   host export. Prebuilt masks/factors must be a separate denominator.
5. A large same-contract advantage and adversarial controls before paper claims.
