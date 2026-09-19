# G6 latest-baseline diagnostic protocol

Date frozen: 2026-09-04 Asia/Shanghai

Status: frozen before baseline adapter correctness or timing observations.

## Question and decision boundary

G6 asks whether the current TensorJoin result survives the strongest recent
baseline evidence that is reproducible on the same RTX PRO 6000 Blackwell host.
It is a diagnostic campaign, not a formal promotion campaign.

The nearest new exact system is RT-HiSS (SC 2026, arXiv:2609.01975).  The paper
describes an RT-core BVH filter plus CUDA refinement, but the audited paper and
project searches exposed no public implementation or artifact.  G6 therefore
does not invent an RT-HiSS reimplementation.  It records RT-HiSS as the newest
unavailable direct comparator and measures the newest available implementations
used by or adjacent to that paper:

1. GTS, upstream Git commit
   `3bac1b725e92e98e3b69d5cb79ad9c56ccb5e639`;
2. NVIDIA cuVS brute force, Python package `cuvs-cu13==26.8.1`.

SimJoin and DiskJoin remain approximate/out-of-core context, not exact keepers.

## Frozen workload and output contract

Two deterministic workloads are used in sequence:

| Gate | Dataset | Shape | Epsilon | Exact directed count | Exact raw-u64 SHA-256 |
|---|---|---:|---:|---:|---|
| G6-S | G2A CIFAR-GIST subset | 4,096 x 512 | 0.7541135250198396 | 262,144 | `da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d` |
| G6-D | G2B CIFAR-GIST | 60,000 x 512 | 0.62890625 | 3,926,078 | `13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495` |

The semantic contract is Euclidean epsilon self-join over the exact values of
the stored float32 coordinates widened to float64.  Output is every directed
pair including self, encoded as sorted little-endian
`uint64(i) * N + uint64(j)`.  Passing exactness requires zero missing, extra,
duplicate, or invalid IDs and equality with the frozen count and hash.

The distinct G2A summary JSON hash is
`92b7975a5b4d7a8a7862d2797a636385318cf3e2822f5726e7daccde05f25c76`;
it must not be used as the pair-file hash.  G6 attempt GTS-G2A-a0 stopped before
GPU execution when the initial protocol confused these two receipts.  The
failure is retained and the corrected pair receipt above is frozen before a1.

## Public denominator

The timed boundary starts with a pageable-host float32 matrix already loaded
from disk and ends after all qualifying pair IDs have reached pageable host
memory and have been sorted into the canonical representation.  It includes:

- device allocation and input H2D transfer;
- index/build/setup that is specific to this dataset;
- every search/refinement kernel;
- result materialization and D2H transfer;
- host canonicalization and sort.

It excludes disk input, process startup, CUDA context creation, compilation,
post-return hashing, JSON serialization, and correctness-set diagnostics.  A
timing is descriptive only unless its output passes the exact hash in that same
process.

## Baseline-specific contract

### GTS

- Preserve the upstream algorithm's float32 coordinates, bounds, distance
  accumulation, and radius comparison.  A compatibility-only C++17/SM120 build
  patch and a pair-ID export adapter are allowed and must be diffed and hashed.
- Freeze the exposed upstream configuration to `TREE_ORDER=10`, `MAX_SIZE=20`,
  and `MAX_H=5`, which is deep enough for 60,000 rows.  Cap the V2 traversal
  workspace at 7 GiB: upstream stores its element capacity in a signed `int`,
  so its default of half the free memory overflows on a 96 GiB-class device.
  The cap is a capacity/toolchain compatibility patch, not a search-semantic
  change.
- Do not label upstream GTS exact merely because it completes.  Its result must
  pass the float64-oracle count and pair hash.
- Any precision-changing repair is a separately named derived implementation,
  not the upstream GTS baseline.
- The cloned upstream snapshot currently exposes no license file.  Use it for
  this internal experiment only and do not redistribute the patched source.

### cuVS brute force

- Use exhaustive brute-force kNN with squared Euclidean distance and filter at
  the frozen squared-radius threshold.
- Freeze `k=4096`.  G2B's existing exact reference has maximum row cardinality
  2,855, so this is sufficient for this frozen dataset.  Nevertheless, fail
  closed if any row has 4,096 accepted neighbors or if the exact hash differs.
- Search may be query-batched for memory capacity, but every query row must be
  processed exactly once.  Record batch size and package/runtime versions.
- cuVS is not an index-system peer; report it as a current vendor brute-force
  lower-level comparator and never as an RT-HiSS substitute.

## Execution order and gates

1. Retain the untouched-upstream GTS CUDA-13.1 build attempt and its failure or
   success log.
2. Build the minimal GTS compatibility/export adapter; record the full diff,
   source hash, binary hash, compiler, and target architecture.
3. Create a dedicated cuVS virtual environment; record the fully resolved
   package versions and hashes.
4. Run G6-S once per baseline.  Stop a baseline if it crashes, exceeds capacity,
   emits invalid IDs, or fails the exact count/hash.
5. Run one fresh-process G6-D diagnostic per baseline only after its G6-S gate
   passes.  If GTS fails solely at threshold-boundary precision, retain the
   mismatch ledger and do not time it as an exact comparator.
6. Recheck exact output, GPU isolation, and artifact hashes after every run.

One run is not a formal latency estimate.  A baseline that changes the current
paper decision must later enter a direction-balanced, fresh-process campaign
under the unchanged denominator.

## Safety and reproducibility

- Host: `gpu-host-8`; project:
  `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`.
- Physical GPU1 is the intended device.  Hold
  `@TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock` for every GPU run.
- Immediately before execution, require GPU1 to have no foreign compute process
  and record `nvidia-smi`.  The child sees exactly one remapped CUDA device.
- Do not modify or terminate processes on GPU0 or GPU3; do not kill any foreign
  process on any device.
- Do not change application clocks, power limits, persistence mode, or host
  services.
- Raw logs go to `raw/g6_*`, structured outputs to `results/g6_*`, immutable
  receipts/diffs to `artifacts/g6_*`, and adapters outside `external/GTS`.
