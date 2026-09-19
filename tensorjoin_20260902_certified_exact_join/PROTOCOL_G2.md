# Protocol G2: Cifar60K Exact Self-Join External Validation

Experiment family: `tensorjoin_20260903_external_exact_selfjoin_g2`

This protocol is frozen before G2A data preparation or adapter execution.

## Roles

- **External exact keepers:** GDS-Join FP64 and MiSTIC FP64, independently.
- **Candidate:** TensorJoin certified INT8 -> guarded FP32 -> FP64 refinement
  cascade with scalable output compaction.
- **Context only:** FaSTED FP16-input/FP32-accumulation. It is not an exact
  keeper and has no pass/fail vote on the exact comparison.
- **Oracle:** FP64 direct-difference squared Euclidean distance over the exact
  widening of the source float32 vectors.

No method may silently use a different input subset, radius, output convention,
or quality target.

## Canonical operation and output

For an input matrix `X` with `N` rows and 512 columns, return every **directed**
pair `(i,j)` satisfying

```text
sum_k (float64(X[i,k]) - float64(X[j,k]))^2 <= epsilon^2.
```

- IDs are subset-row IDs for G2A and original Cifar60K base-row IDs for G2B.
- Encode a pair as `uint64(i) * N + uint64(j)` and sort ascending for
  validation.
- Self-pairs are included.
- Symmetric methods that compute one triangle must expand non-self pairs into
  both directions before validation.
- Duplicate IDs, missing IDs, extra IDs, invalid IDs, and output-buffer
  overflow are all correctness failures.
- Any internal point/dimension reorder must be inverted before emitting IDs.

## G2A: 4,096-point admission contract

### Data

- Source: `data/cifar60k/cifar60k_base.fvecs`, SHA-256 recorded in the dataset
  receipt.
- Selection: NumPy `default_rng(20260903).choice(60000, 4096,
  replace=False)`, then sort the selected source-row IDs ascending.
- Numeric source: the stored float32 coordinates. Exact methods receive their
  exact float64 widening. FaSTED receives its declared half-precision
  conversion from the same textual values.
- Dimension: 512.

### Radius

Use SciPy `pdist(..., metric="sqeuclidean")`, which directly accumulates
float64 coordinate differences for the upper triangle. Select a strict mid-gap
radius whose directed count, including self-pairs, is 4,096 x 64 if there is no
tie. Record the two bracketing squared distances, their gap, `epsilon`,
`epsilon^2`, actual pair count, and any tie expansion. The effective rounded
`epsilon^2` must remain strictly between the brackets.

### Required artifacts

- selected source IDs;
- contiguous float32 NumPy input;
- exact-widened float64 raw binary input for MiSTIC;
- comma-separated exact-widened decimal input for GDS-Join/FaSTED;
- sorted canonical oracle pair IDs;
- SHA-256 and shape/dtype receipts for every artifact.

### Gate

G2A is an admission/correctness stage, not a performance claim. Each exact
external keeper and the candidate must satisfy:

- zero missing, extra, duplicate, or invalid pairs;
- output count equals the oracle count;
- all output storage completed without overflow/truncation;
- at least two consecutive isolated executions return the same canonical hash.

If an upstream exact implementation cannot expose canonical IDs through a
minimal auditable adapter, mark that comparator `unvalidated`; do not infer
correctness from its printed count.

## G2B: full Cifar60K contract

G2B is forbidden until G2A passes for the candidate and at least one exact
external keeper.

- Data: all 60,000 rows of `cifar60k_base.fvecs`, dimension 512.
- Radius: `epsilon = 0.62890625`, the first Cifar60K radius distributed with
  the pinned FaSTED experiment records.
- Canonical output: directed IDs including self-pairs, as above.
- Oracle: a separately validated exact FP64 implementation. FaSTED's published
  pair count is not an oracle.
- Expected scale for capacity planning only: approximately 3.9 million
  directed pairs. This is not a frozen exact count.

The candidate must use chunked/tiled production plus scalable compaction or an
equivalent output-sensitive design. A fixed 65K result allocation or a materialized
60,000 x 60,000 status matrix is inadmissible.

## Frozen measurement scopes

### Public end-to-end denominator

Start with the source array resident in pageable host memory. End when sorted
canonical pair IDs are resident in host memory. Include:

- method-specific quantization, reorder, index, or certificate construction;
- host-to-device transfer;
- all search, refinement, compaction, and sort work;
- device-to-host transfer and ID-remapping needed by the canonical output.

Exclude file I/O, process startup, compilation, disk serialization, oracle
construction, and post-run correctness comparison. Peak host/device memory and
all excluded setup are reported separately.

### Secondary resident denominator

Inputs and reusable method state may already be device-resident. Time all GPU
work needed to produce the complete device-resident pair set. This diagnostic
scope must never be described as end to end.

## Isolation and run order

- Host: `gpu-host-8`; physical GPU0 only.
- Campaign lock: `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`.
- Before every measured process: record `nvidia-smi`, active compute processes,
  clocks/power state, driver, toolkit, binary/source hashes, and hold a
  30-second empty-GPU quiescence check.
- Never terminate or modify a foreign process. Abort and retain the attempted
  observation if any appears before or during a process.
- Eight fresh-process rounds. Rotate and reverse the four-method order so each
  exact method and the candidate appear early and late equally often. Preserve
  raw order and all clean observations.

## G2B estimators and decision gate

For each exact keeper independently, compute candidate/keeper comparisons from
matched clean rounds. Report raw latencies, p10/median/p90, process wins,
geometric mean of paired speedup, order split, and a seeded process-bootstrap
95% interval. No post-hoc estimator substitution is allowed.

Candidate acceptance requires all of the following:

1. exact canonical output in every admitted process;
2. clean compute-sanitizer memcheck plus a 1,000-launch or equivalent sustained
   output-hash stability test on the scalable core;
3. candidate faster than **both** validated exact external keepers in at least
   7/8 matched processes;
4. paired geometric-mean speedup lower bound at least **1.50x against the
   faster exact keeper**, where the faster keeper is the one with the lower
   predeclared end-to-end marginal median;
5. no peak-memory or output-capacity failure.

FaSTED is reported separately with precision, recall, F1, pair count, and the
same timing scopes when technically possible. Because its quality differs and
the pinned source has no license file, it is context rather than a gate.

## Evidence status

- Source inspection/build success: `partial` only.
- One-process G2A compatibility: `unvalidated` until canonical IDs match twice.
- G2A pass: `measured` correctness at subset scale, never a speed claim.
- G2B pass: `measured` only for the frozen dataset, radius, output, hardware,
  software, and timing scopes.
