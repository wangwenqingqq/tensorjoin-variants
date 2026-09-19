# Protocol H0: Multi-Vector Certificate Opportunity

Experiment ID: `tensorjoin_20260903_multivector_certificate_h0`

## Purpose and evidence level

H0 is a CPU-only structural kill test.  It asks whether the existing per-vector
INT8 residual certificate remains selective after exact interval propagation
through a variable-cardinality symmetric Chamfer object score.  It is not GPU
correctness, generated-code, latency, end-to-end, or paper-level novelty
evidence.

## Frozen input and object semantics

| Cell | Cache | Object key | Token count |
|---|---|---|---|
| learned audio | `data/esc50_panns_cnn14_1s_2048d.npz` | ESC-50 clip | exactly 5 one-second PANNs embeddings |
| learned video | `data/ucf101_r3d18_512d.npz` | UCF101 source group | 4--7 R3D-18 clip embeddings |

- L2-normalize every stored float32 token in FP64 and round the normalized
  token once to float32.  Reject zero or non-finite norms.
- Deterministically sort object IDs, permute them with seed `20260903`, take
  128 query objects followed by 512 disjoint base objects, and retain token
  order from the cache.
- Score each object pair using symmetric Chamfer squared distance as defined in
  `GATE0_H0_MULTIVECTOR_JOIN.md`.
- Direct FP64 differences on the normalized stored float32 tokens define the
  oracle; GEMM cancellation is not the oracle.

## Thresholds

For target densities 1 and 8 results per query, stable-sort all object-pair
oracle scores by `(score, query_object_id, base_object_id)`.  Use a strict
mid-gap threshold between positions `128*target-1` and `128*target`; fail if
there is no strict gap.  The threshold is selected only once from the exact
oracle and applies unchanged to every candidate classification.

## Candidate certificate

1. Quantize each normalized token independently to signed INT8 using its
   float32 maximum absolute value divided by 127.
2. Reconstruct in float32, compute the residual norm in FP64, and round its
   stored upper bound outward with `nextafter`.
3. Reconstruct token-pair L2 distances from INT32 dots and per-token scales.
4. Form outward pairwise squared-distance intervals with the sum of the two
   residual norms.
5. Propagate pairwise intervals through directed `min`, directed `mean`, and
   the symmetric average to obtain the object interval.
6. Directly accept/reject by the object interval.  For ambiguous objects,
   identify the union of forward- and reverse-competitive token cells, where a
   cell's lower bound does not exceed the best upper bound in its directed row.
   Exact refinement of those cells must reproduce the oracle object decision.

## Denominators and reported counts

For every dataset/target cell report:

- total object pairs and exact output pairs;
- direct accepts, direct rejects, and ambiguous object pairs;
- object ambiguity fraction over all object pairs;
- total token cells over all object pairs;
- competitive token cells inside ambiguous object pairs;
- exact-refinement fraction over all token cells and over token cells belonging
  to ambiguous objects;
- lower/upper containment violations, unsafe accepts/rejects, and final output
  mismatches.

The low-precision scan still touches every token cell.  H0 does not count those
cells as pruned and makes no latency claim.

## Frozen pass/stop rule

All four dataset/target cells must satisfy:

1. zero non-finite values, interval-containment violations, unsafe direct
   decisions, and final mismatches;
2. object ambiguity fraction at most 5%;
3. exact competitive-cell refinement at most 2.5% of all token cells;
4. INT32 dot-accumulator worst case `D*127^2` within signed INT32.

Any exactness failure rejects the certificate construction.  A selectivity
failure closes H0 for the current representation; do not retune thresholds,
object definitions, or token grouping on these measured caches.  A pass admits
an H1 GPU screen, not a novelty or performance claim.

## Reproducibility

The runner must refuse overwrite, record source/cache hashes, emit a raw log,
and use `CUDA_VISIBLE_DEVICES=""`.  No GPU lock is required because H0 launches
no CUDA work.

