# Paper-Facing Main Results Tables

Date: 2026-09-03

These tables are numeric inventory for the paper spine.  They are not a prose
draft.  Every repeated number must remain identical to the frozen decision
artifact named below.

## Table 1: Full public-denominator external comparison

Frozen source: `DECISION_G2B_FORMAL.md`.

Workload: CIFAR-GIST-512, 60,000-vector self-join, epsilon 0.62890625.
Denominator: pageable host FP32 source through sorted canonical host `uint64`
IDs; 3,926,078 exact directed pairs.

| Method | Arithmetic/output contract | p10 (s) | Median (s) | p90 (s) | Exact output | Relative to TensorJoin |
|---|---|---:|---:|---:|:---:|---:|
| TensorJoin | certified mixed precision, exact | 0.924369 | 0.926153 | 0.928841 | yes | 1.000x |
| MiSTIC | FP64, exact | 4.631841 | 4.732177 | 4.770933 | yes | 5.083x slower, 95% CI [5.030x, 5.133x] |
| GDS-Join | FP64, exact | 10.820870 | 10.927797 | 10.990203 | yes | 11.783x slower, 95% CI [11.716x, 11.846x] |
| FaSTED | FP16 input/FP32 accumulation | 0.200003 | 0.229410 | 0.231410 | no: 598 FN, 1,130 FP | 4.037x faster by marginal median; approximate context only |

The FaSTED relative number is `0.926153 / 0.229410 = 4.037x`; it is not an
acceptance comparison because the output differs.

## Table 2: Dynamic-router mechanism attribution

Frozen source: `DECISION_G4C.md`.

Denominator: resident-input/output, host-dispatched operator with actual
dynamic count reads and launch extents; N=4096, target degree 64.

| Dataset | D | All-FP64-refinement keeper median (us) | Dynamic router median (us) | Paired speedup | 95% process CI | Sustained speedup | Process/sustained wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT-128 | 128 | 264.449 | 207.959 | 1.270875x | [1.267100x, 1.274316x] | 1.269737x | 8/8 + 8/8 |
| CIFAR-GIST-512 | 512 | 826.680 | 355.435 | 2.328925x | [2.299469x, 2.358065x] | 2.340438x | 8/8 + 8/8 |
| Fashion-MNIST-784 | 784 | 487.019 | 287.777 | 1.709810x | [1.687169x, 1.735672x] | 1.708095x | 8/8 + 8/8 |

This table attributes the routing mechanism.  It must not be labeled an
external end-to-end comparison.

## Table 3: Same-artifact unification boundary

Frozen source: `DECISION_G5_P4.md`.

Denominator: CIFAR-GIST-512 60K, pageable-host FP32 source through sorted
canonical host IDs; two direction-balanced diagnostic rounds on the exact,
audited G5 router.  MiSTIC is the faster exact keeper.

| Round and order | TensorJoin (s) | MiSTIC (s) | GDS-Join (s) | Speedup over MiSTIC | Frozen gate |
|---|---:|---:|---:|---:|---|
| 0: GDS, MiSTIC, TensorJoin | 1.178670 | 4.822746 | 10.823552 | 4.091684x | pass |
| 1: TensorJoin, MiSTIC, GDS | 5.284857 | 5.024050 | 11.108409 | 0.950650x | **fail: below 1.50x** |

All six outputs, guards, source hashes, stage counts, and binary identities
pass.  The paired geometric mean is 1.972247x, but the predeclared every-round
1.50x gate fails.  This forbids formal same-artifact performance wording and
requires Tables 1 and 2 to remain separately scoped.

## Table 4: Multi-vector boundary result that must not be hidden

Frozen source: `DECISION_H2B_P4.md`.

Denominator: complete resident 128x512 multi-vector object join through final
host IDs/sort; first predeclared fresh process, 100 observations/variant/cell.

| Dataset | Target | Pedantic exact keeper (us) | INT8 certificate candidate (us) | Keeper/candidate | Outcome |
|---|---:|---:|---:|---:|---|
| PANNs D2048 | 1 | 672.221 | 575.269 | 1.169x | below 1.25x gate |
| PANNs D2048 | 8 | 674.442 | 1,012.429 | 0.666x | candidate slower |
| R3D-18 D512 | 1 | 505.183 | 497.300 | 1.016x | below 1.25x gate |
| R3D-18 D512 | 8 | 587.617 | 783.151 | 0.750x | candidate slower |

This boundary belongs in limitations or an honest design-scope discussion if
multi-vector extension is mentioned.  It forbids a multi-vector performance
contribution.

## Table 5: Latest-baseline audit

Frozen source: `DECISION_G6_LATEST_BASELINES.md`.

Contract: same CIFAR-GIST-512 60K source, radius, FP64 oracle, canonical
directed-ID output, complete public denominator, and isolated SM120 GPU.  GTS
and cuVS failed the G2A exactness gate, so each G2B point is one diagnostic run
and cannot enter an exact-speedup column.

| Method | Artifact | G2A delta | G2B disagreement | G2B one-run time | Paper role |
|---|---|---:|---:|---:|---|
| RT-HiSS | no public artifact found | -- | not measured | not measured | newest direct prior art and explicit limitation |
| cuVS brute force | `cuvs-cu13==26.8.1` | -4 | 24 FP, 26 FN; F1 0.999993632 | 3.551750 s | accuracy-latency context only |
| GTS | upstream `3bac1b7` + export adapter | +2 | 4 FP, 0 FN; F1 0.999999491 | 1111.520 s | accuracy-latency context only |

The table supports a numerical-contract result, not a claim that the external
implementations are approximate algorithms.  MiSTIC and GDS-Join remain the
only measured same-contract exact keepers in the headline table.
