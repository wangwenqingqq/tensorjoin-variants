# Baseline Audit: RT-HiSS and Current Comparator Set

Date: 2026-09-04

## Decision

RT-HiSS is the newest direct algorithmic baseline for TensorJoin found in the
2026-09-04 search. It is an SC 2026 exact high-dimensional GPU distance
similarity-search/self-join system. The paper's appearance after the original
TensorJoin Gate 0 means that MiSTIC and GDS-Join alone are no longer a
sufficient external comparison set.

RT-HiSS is not yet a runnable comparator in this workspace. The arXiv paper and
source bundle contain no public implementation or artifact URL, and title- and
author-based public repository searches found no release. This is a missing
artifact, not evidence that TensorJoin is faster.

The two newest executable additions also do not enlarge the exact comparator
set under TensorJoin's frozen numerical contract. GTS commit `3bac1b7` and
cuVS 26.8.1 both disagree with the FP64 oracle at threshold-boundary pairs.
Their completed full-scale runs are reported below as accuracy-latency context,
not exact speedups.

## Nearest contract

RT-HiSS reports:

- exact epsilon distance search, evaluated as a self-join;
- a three-dimensional RT-core BVH filter followed by a CUDA-core refinement;
- six real datasets with 288K--11.6M vectors and D18--D128;
- target mean neighbor counts of 256, 1,024, and 4,096;
- response time including reorder, kd-tree grouping, RT index construction,
  CUDA refinement, and device-to-host result copy, excluding disk input;
- a Quadro RTX 5000 16 GiB platform, CUDA 13.0, and OWL 7.4.0;
- comparisons against GDS-Join, COSS, GTS, cuVS-Brute, PyTorch3D, and RTNN
  for the separate D3 regime.

The paper reports up to 9.19x over GDS-Join and up to 8.37x over COSS, but also
loses to GDS-Join in one SuSy cell and to COSS in one WEC cell. These numbers
cannot be compared numerically with TensorJoin's RTX PRO 6000 Blackwell
CIFAR-GIST-512 60K result.

## Material contract gaps

1. RT-HiSS calls the algorithm exact but does not state a numerical arithmetic
   contract or a certified threshold-decision analysis. Its output must be
   checked against the same FP64 oracle used by TensorJoin.
2. RT-HiSS copies a compressed result-mask representation; TensorJoin's G2B
   denominator ends at sorted canonical host pair IDs. Output materialization
   must be normalized before timing.
3. RT-HiSS evaluates D18--D128, whereas TensorJoin's current external anchor is
   D512. Both systems should be run on both workload families.
4. RT-HiSS uses much larger inputs and mean output degrees (256--4,096) than
   TensorJoin's 60K target-degree-64 anchor.
5. RT-HiSS was measured on Quadro RTX 5000, not the target SM120 GPU.

## Measured G6 disposition

| Baseline | G2A delta | G2B exact-only / baseline-only | G2B one-run public time | Disposition |
|---|---:|---:|---:|---|
| cuVS 26.8.1 brute force | -4 | 26 FN / 24 FP | 3.551750 s | non-exact under frozen oracle; context only |
| GTS `3bac1b7` | +2 | 0 FN / 4 FP | 1111.520 s | non-exact under frozen oracle; context only |

GTS's four G2B false positives are two unordered pairs accepted by its
sequential FP32 accumulation but rejected by the FP64 oracle. cuVS's G2A
mismatches are shape dependent: its single-query path accepts the boundary
pairs while the frozen 256-query batch path rejects them. Neither result is an
export, duplicate, invalid-ID, or top-k-saturation artifact. Full evidence and
allowed wording are frozen in `DECISION_G6_LATEST_BASELINES.md`.

## Required comparator tiers

### Headline exact, same-contract and measured

1. MiSTIC -- current fastest exact measured keeper in G2B.
2. GDS-Join -- exact grid-index lineage and existing G2B comparator.

### Required but pending or rejected for this exact contract

1. RT-HiSS -- newest direct exact high-dimensional GPU self-join baseline;
   pending code/artifact access.
2. COSS -- exact coordinate-oblivious GPU index and an RT-HiSS comparator;
   not yet measured under G2B.
3. GTS -- executable and measured, but rejected as a same-contract exact
   keeper due to FP32 threshold disagreement.
4. cuVS brute-force search -- executable and measured, but rejected as a
   same-contract exact keeper due to shape-dependent threshold disagreement.

### Approximate context only

- FaSTED -- nearest Tensor-Core throughput comparator, not exact.
- SimJoin -- SIGMOD 2025 approximate join algorithm.
- DiskJoin -- SIGMOD 2025 approximate, SSD/billion-scale system with a
  different memory hierarchy and recall contract.

## Cheap next gate

Do not alter the paper's performance claim using RT-HiSS's published numbers.
First obtain the RT-HiSS artifact or a runnable implementation and locate a
runnable COSS artifact. GTS and cuVS-Brute have now completed the frozen G2B
context protocol and failed same-contract exact admission. A future baseline is
admitted to the headline only after matching input semantics, epsilon, exact
FP64 oracle, output representation, timing boundaries, hardware, and process
isolation.
