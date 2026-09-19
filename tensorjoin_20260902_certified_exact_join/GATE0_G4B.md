# Gate 0 G4B: public breadth matrix for the dynamic precision router

Date: 2026-09-03

## Decision

**Proceed as an extensibility gate, not as a new novelty claim.**  The nearest
prior art already prevents a broad claim that low-precision Tensor-Core search,
filter/refine, or certified retrieval is new.  G4B is justified only because it
tests whether the G4A-R1 mechanism survives across standard public vector
benchmarks, dimensions, output densities, and scales under one exact-output
contract.  A collection of application ports would not justify the work.

The cheap kill test is frozen before implementation.  If the analytic
INT8-to-FP32-to-FP64 cascade is not both exact and materially selective outside
the existing CIFAR cell, stop before formal timing and retain the failed cells.

## Public-data audit

ANN-Benchmarks publishes dense, pre-split benchmark files and identifies SIFT
as 128-dimensional Euclidean data with 1,000,000 train vectors, Fashion-MNIST
as 784-dimensional Euclidean data with 60,000 train vectors, and GIST as
960-dimensional Euclidean data with 1,000,000 train vectors:

- <https://github.com/erikbern/ann-benchmarks#data-sets>
- <https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py>

Faiss independently treats SIFT1M as a standard 1M-vector, 128-dimensional
benchmark and records its original corpus source:

- <https://github.com/facebookresearch/faiss/blob/main/contrib/datasets.py>
- <https://github.com/facebookresearch/faiss/tree/main/benchs#getting-sift1m>

The official Fashion-MNIST repository publishes the 60,000 training images,
their checksums, and an MIT license:

- <https://github.com/zalandoresearch/fashion-mnist>

The ANN-Benchmarks and Faiss *software* repositories are MIT-licensed.  That
does not automatically grant redistribution rights for every upstream dataset.
G4B may download and use the public benchmark artifacts for research, but the
SIFT artifact must not be redistributed until its upstream data terms are
separately confirmed.  Dataset bytes are excluded from any future code bundle
by default.

## Frozen datasets and axes

Use three Euclidean public sources with different native dimensions:

| Dataset ID | Source rows | Native dimension | Acquisition |
|---|---:|---:|---|
| `sift128` | original SIFT1M base split | 128 | reuse the read-only fvecs already present under the user's GTS data; record path and SHA-256 |
| `cifar_gist512` | existing frozen Cifar60K base fvecs | 512 | reuse the G2 source and SHA-256 |
| `fashion784` | official Fashion-MNIST train images | 784 | official gzip; verify published MD5 and local SHA-256 |

GIST1M-960 is deferred at this gate: its 3.6 GB HDF5 artifact adds substantial
download and storage cost while Fashion-MNIST already exercises a high native
dimension below the current exact INT32-to-FP32 conversion limit.  It may be a
later confirmation cell, not a rescue cell.

Pre-execution acquisition audit found an existing complete SIFT1M base file at
`@TENSORJOIN_ROOT@/project/GTS/Datasets/sift1m/sift_base.fvecs` with
1,000,000 128-dimensional records and SHA-256
`21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`.
Its colocated README identifies the original TexMex source.  G4B therefore
reuses this file read-only instead of downloading a duplicate 525 MB HDF5
container.  This acquisition change occurred before opportunity measurement
and does not alter sampling, dimensions, cells, or gates.

For every source, construct one seeded permutation of the full train/base
split with NumPy `PCG64` seed `20260903`.  Use nested prefixes of that fixed
permutation at `N in {1024, 2048, 4096}`.  Do not choose rows after observing
certificate or timing behavior.

For every dataset/scale pair, define three strict self-range-join radii targeting
average non-self directed degrees `k in {1, 16, 64}`.  Select the radius at the
midpoint of adjacent exact-FP64 upper-triangle squared-distance order
statistics.  If the target falls inside a tie, move to the next strict gap and
report the resulting tie-expanded degree.  Preserve self pairs and canonical
upper-triangle IDs.

This yields a `3 datasets x 3 scales x 3 densities = 27` cell opportunity
matrix.  Dataset and dimension are confounded across rows; G4B may claim
cross-dataset breadth, not a causal dimension-only effect.  Scale and density
effects are interpretable within each dataset because the sampled prefixes are
nested and only the radius or prefix changes.

## Exactness and analytic-domain preconditions

The G3B-R1 integer dot path requires both exact INT32 accumulation and exact
conversion of the accumulator to FP32.  Before any GPU cell is admitted:

- inputs must be finite float32 rows;
- per-vector symmetric INT8 quantization must preserve the G3B-R1
  normal-or-zero residual-bound rule;
- `D * 127^2 <= 2^24` must hold; and
- all threshold and radius endpoints must be outward-rounded as in G3B-R1 and
  G3C-B-R1.

The frozen native dimensions 128, 512, and 784 satisfy the conversion bound.
No padding, projection, learned preprocessing, or dimension truncation is
allowed in the primary cells.

## Cheap opportunity kill test

The first G4B execution is correctness and routing geometry only; it is not
performance evidence.  For every one of the 27 cells, retain:

- exact FP64 output count, actual directed degree, and sorted upper-ID hash;
- G3B direct-accept, reject, and ambiguity counts;
- G3C direct-accept, reject, and residual-FP64 counts;
- zero unsafe G3B/G3C decisions, zero final mismatch, zero duplicate or
  lower-triangle IDs, and zero overflow;
- the fraction of all upper pairs left ambiguous by G3B; and
- the fraction of G3B ambiguity left for FP64 after G3C.

The opportunity gate passes only if:

1. all 27 cells are exact and structurally valid;
2. all source, sample-index, threshold, and output hashes are recorded;
3. at `N=4096`, both non-CIFAR datasets have at least two of three density
   cells with G3C residual FP64 work at or below 25% of G3B ambiguity; and
4. at least one low- or medium-density `N=4096` cell on each non-CIFAR dataset
   leaves no more than 25% of all upper pairs ambiguous after G3B, so the
   Tensor-Core certificate is not merely forwarding the full join.

Failure of an individual performance-opportunity cell is evidence for router
selection, not permission to delete the cell.  Failure of exactness stops G4B.
Failure of either cross-dataset selectivity condition stops the generic GPU
timing implementation; additional data or per-dataset guard tuning may not
rescue the gate.

## Promotion after the opportunity gate

If the gate passes, implement one shape-parameterized GPU operator without
dataset-specific constants.  It must preserve the accepted G3B/G3C arithmetic
and dynamic count discovery, then run a compact timing subset selected by a
predeclared rule from the matrix.  Comparisons must include the unchanged
two-stage exact keeper and the three-stage router under the same denominator.
Any cost model or route selector is a later, separately frozen gate.

G4B cannot inherit the G2B external-system speedup or call its subset results
full-scale.  G2B remains the external headline; G4B supplies breadth and
boundary evidence only.
