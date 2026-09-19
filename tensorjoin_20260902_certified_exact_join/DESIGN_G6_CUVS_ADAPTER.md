# G6 cuVS brute-force adapter design

Date: 2026-09-04

Purpose: expose the latest installable cuVS brute-force implementation under
the frozen TensorJoin self-join denominator.  cuVS is a current vendor
comparator, not a tree index and not a replacement for unavailable RT-HiSS.

## Frozen implementation

- `cuvs-cu13==26.8.1`, `libcuvs-cu13==26.8.1`;
- RAFT/RMM 26.8.0, CuPy 14.1.0, NumPy 2.5.2;
- package-native `brute_force.build(..., metric="sqeuclidean")` and
  `brute_force.search`;
- contiguous row-major float32 input and `k=4096`;
- query batches of 256 rows for both G2A and G2B.

All Linux wheels and SHA-256 receipts are retained under
`artifacts/g6/wheels_linux`.  The environment is project-local at
`envs/cuvs_py312`; no shared environment is modified.

## Data flow and denominator

```text
pageable host float32 matrix (timer starts)
  -> one CuPy H2D copy
  -> cuVS brute-force build
  -> batched exhaustive kNN over device views of the same matrix
  -> distances and IDs D2H per batch
  -> host float64 comparison of returned float32 squared distances
  -> canonical uint64 pair encoding and host sort (timer stops)
  -> output file, hash, and oracle comparison outside timer
```

The host comparison widens returned distances but does not recover precision
lost inside cuVS.  Therefore exact admission still requires the complete pair
hash, not merely exhaustive candidate enumeration.

## Completeness and capacity proof

The frozen G2B oracle's maximum row cardinality is 2,855.  Since `k=4096`, an
exact top-k result contains every qualifying G2B neighbor.  The runner also
records the observed maximum accepted row count and fails closed if any row
reaches 4,096.  G2A uses `k=N=4096` and is exhaustive by construction.

Peak explicit result storage is one 256 x 4,096 float32 distance matrix and one
256 x 4,096 int64 ID matrix, approximately 12 MiB, plus cuVS workspace.  The
dataset itself is approximately 8 MiB for G2A or 117 MiB for G2B.

## Gates and interpretation

1. Import/version/pip-check receipt.
2. G2A complete output, no invalid IDs, no duplicates, no saturation, exact
   count and hash.
3. G2B single fresh-process diagnostic only if G2A passes.
4. A timing may be compared descriptively only when the same process passes
   exactness.  Formal claims require a later balanced multi-process campaign.

If cuVS returns a numerically different threshold set, retain the exact
missing/extra IDs and classify it as a brute-force FP32 context baseline rather
than an exact keeper for the float64-oracle contract.

