# Protocol E0: GTS-Style Pivot-Tree Tensorization Ceiling

Experiment ID: `tensorjoin_20260903_pivot_tree_ceiling_e0`

## Purpose and evidence level

E0 is a CPU-only structural kill test. It measures an optimistic upper bound on
the value of adding a GTS-style tree before the D1/D2 exact Tensor-Core scan.
It is not a latency result: tree construction, query gathering, and panel
launch costs are excluded, while pivot distances are exact FP64.

## Frozen workloads

- Learned audio: D1 PANNs Cnn14 cache and its seed-`20260902`, clip-disjoint
  512x4096 split.
- Learned video: D2 R3D-18 UCF101 cache and its seed-`20260903`, source-group-
  disjoint 512x4096 split.
- For each modality, reuse tie-aware mid-gap radii at nominal 1 and 64
  results/query. Direct blocked FP64 differences remain the oracle.

## Deterministic pivot tree

- Build on the 4,096 base vectors only, in local base-ID order.
- Fanout 8 and maximum leaf size 64. For an internal node, choose the member at
  the midpoint of its current stable order as pivot, compute exact FP64 L2
  distances, stable-sort by `(distance, local_base_id)`, and split into eight
  contiguous near-equal chunks.
- Each child stores its exact minimum and maximum member distance to the parent
  pivot. The resulting 4,096-point tree has fixed 64-vector leaves when no
  construction invariant is violated.

## Exact search simulation

For query `q`, parent pivot `p`, radius `r`, and a child whose member-to-pivot
distances lie in `[a,b]`, use:

```text
lower = a-d(q,p)        if d(q,p)<a
        d(q,p)-b        if d(q,p)>b
        0               otherwise
upper = d(q,p)+b
```

- Reject the whole child if `lower > r`.
- Bulk-accept it if `upper <= r`.
- Descend otherwise; a boundary leaf sends all of its points to exact
  per-vector certification/refinement.
- Validate every visited lower/upper interval against actual member distances,
  and require exact final pair IDs.

## Tensorization ceiling

- All internal pivots are granted one dense `512 x P` query-pivot panel, with P
  padded upward to 64.
- For each boundary leaf, gather its active queries in groups padded upward to
  16 and its members upward to 64. Packing and scheduling costs are excluded.
- Charge every padded cell, not only useful pairs. Report pivot cells, useful
  leaf pairs, padded leaf cells, bulk-accepted pairs, rejected pairs, leaf-panel
  utilization, and total padded-cell fraction versus the 512x4096 full scan.

## Pass/stop rule

All four modality/radius cells must have zero output mismatches and bound
violations, leaf-panel utilization at least 70%, and:

- nominal-1 total padded-cell fraction <=20%;
- nominal-64 total padded-cell fraction <=35%.

Both modalities must pass both radii. A failure closes this fixed pivot-tree
front end; do not tune fanout, leaf size, or gates on the measured caches. A
pass admits E1 quantized-pivot certificates, not GPU performance claims.
