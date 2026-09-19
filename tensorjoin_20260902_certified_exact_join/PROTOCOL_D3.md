# Protocol D3: Hyperspectral Tensor Breadth Gate

Experiment family: `tensorjoin_20260903_indian_pines_d3`

## Purpose

D3 tests extensibility to a scientific tensor representation after learned
audio passed and learned video exposed selectivity-sensitive performance. It is
not a new novelty claim and does not reopen the previously rejected progressive
HSI top-k/tree direction.

## Frozen data and representation

- Source: `IndianPines.mat` from Zenodo record 10638886, required MD5
  `cea396f8a7bdf947f26a7a36c7b7c81a`.
- Select the largest numeric 3-D array and canonicalize it to spatial x spatial
  x spectral layout; require `145x145x220`.
- Compute one global mean and standard deviation per spectral band over the
  full cube. Standardize every pixel, extract all valid `3x3x220` patches,
  flatten in C order, and L2-normalize each patch.
- Append four zero coordinates, producing a fixed 1,984-D FP32 vector aligned
  to K=64 Tensor-Core steps. Record source, script, and cache hashes.

## Spatially disjoint join split

- Patch top-left coordinates are `0..142` on both spatial axes.
- Query candidates have top-left row `0..47`; base candidates have row
  `51..142`. Thus no source pixel occurs in both a query patch and a base patch;
  rows 48--50 form a deterministic gap.
- Seed `20260903`; sample 512 query and 4,096 base patches without replacement.

## D3A certificate geometry

- Direct blocked FP64 differences define the oracle.
- Use tie-aware FP64 mid-gap radii at nominal 1 and 64 results/query.
- Apply unchanged per-vector signed-INT8 quantization and `1e-4` padded
  certificate.
- Pass: exact final output, zero containment/unsafe decisions, safe INT32
  accumulation, and ambiguity <=10% at both radii.

## D3B performance admission

D3B is admitted only if D3A passes. It must reuse D1's guarded-FP32 keeper and
fused compact/refine candidate with only D specialized to 1,984. A cheap smoke
at both radii precedes sanitizer/stress/formal work. If either smoke ratio is
below 1.35x, stop before formal timing; otherwise use the D1 eight-process
>=1.5x-lower-bound contract.
