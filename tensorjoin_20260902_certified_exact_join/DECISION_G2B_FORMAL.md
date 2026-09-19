# G2B formal public-denominator decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_g2b_public_formal`

## Decision

**PASS. A scoped same-public-denominator performance claim is now allowed.**

On the frozen CIFAR-10-GIST `60000x512` self-join at epsilon `0.62890625`,
TensorJoin, GDS-Join FP64, and MiSTIC FP64 all returned the same sorted
3,926,078 directed pair IDs. Across eight direction- and position-balanced
fresh-process rounds, TensorJoin won every paired comparison against both exact
keepers.

| Method | p10 (s) | Median (s) | p90 (s) | Exact under frozen oracle |
|---|---:|---:|---:|:---:|
| GDS-Join FP64 | 10.820870 | 10.927797 | 10.990203 | yes |
| MiSTIC FP64 | 4.631841 | 4.732177 | 4.770933 | yes |
| TensorJoin exact mixed precision | 0.924369 | 0.926153 | 0.928841 | yes |
| FaSTED FP16/FP32 context | 0.200003 | 0.229410 | 0.231410 | no |

The paired TensorJoin speedup over MiSTIC, the faster exact keeper by the
predeclared marginal-median rule, is **5.083x** geometric mean with seeded
round-bootstrap 95% interval **[5.030x, 5.133x]** and 8/8 wins. Against
GDS-Join it is **11.783x** with interval **[11.716x, 11.846x]** and 8/8 wins.
This exceeds the frozen requirements of at least 7/8 wins against both exact
keepers and a confidence lower bound of at least 1.50x against the faster one.

## Public denominator

The timer starts with the frozen float32 source in pageable host memory and
ends with sorted canonical uint64 pair IDs resident in host memory. It includes
method-specific widening, quantization, reorder/index/schedule construction,
output allocation, host-to-device transfer, search/refinement/compaction,
device-to-host transfer, canonical remapping/symmetry expansion, and host sort.
It excludes file I/O, process startup, CUDA context initialization, compilation,
disk serialization, hashing, and post-timer correctness checks.

## Exactness and safety gates

- All 24 exact-method formal records pass the frozen count/hash, structure, and
  capacity contract; no formal process was contaminated or replaced.
- TensorJoin previously passed full memcheck with `ERROR SUMMARY: 0 errors`.
- Its sustained gate completed 1,000/1,000 iterations and 3,000 core kernel
  launches with invariant stage counts/output hash and zero overflow.
- GPU0 returned to 14 MiB, 0% utilization, and no compute process after the
  campaign.

## FaSTED quality context

The pinned FaSTED adapter is materially faster but is not exact. Every one of
its eight formal runs emitted the same 3,926,610-pair output: 3,925,480 pairs
intersect the exact oracle, 598 exact pairs are missing, and 1,130 emitted pairs
are false positives. Precision is `0.99971222`, recall is `0.99984769`, and F1
is `0.99977995`. It is therefore a quality--latency context point, not an exact
keeper, oracle, or acceptance comparator. The pinned source has no license file
and remains private-evaluation-only.

## Retained operational failure

The first FaSTED validation wrapper was terminated by the monitor after one
already-attributed child GPU PID disappeared from `/proc` and its stale
`nvidia-smi` row was incorrectly reclassified as foreign. The same PID appears
in both the target and foreign lists, the child printed completion, no wrapper
result existed, and GPU0 was empty postflight. The failed manifest, output,
logs, and diagnosis are retained under
`*attempt0_stale_pid_false_contamination*`.

The repair made target-PID ownership monotonic within one run; it did not
disable or weaken detection of never-attributed foreign PIDs. The clean
validation and all 32 formal processes then passed without contamination.

## Allowed and forbidden wording

Allowed:

> On the frozen CIFAR-10-GIST 60K self-join and complete-output public
> denominator, TensorJoin is 5.083x faster than the faster exact external
> keeper (95% paired round-bootstrap interval [5.030x, 5.133x]) and reproduces
> the same 3,926,078 directed pairs in all eight rounds.

Forbidden without further evidence:

- fastest exact similarity join in general;
- first exact mixed-precision or Tensor-Core join;
- superiority across datasets, dimensions, radii, output cardinalities, GPU
  architectures, or multi-GPU settings;
- a claim that FaSTED's errors are acceptable for an application;
- a memory-efficiency claim from the 0.25-second external occupancy sampler.

## Evidence

- Frozen contract: `DESIGN_G2_TIMING.md`, `PROTOCOL_G2_FORMAL.md`.
- Summary: `results/g2b_public_formal_summary.json` (SHA256
  `893e4b63a4df40af6bf3bb6854275e58b9a737287e7fec0cbb0a5c34a21b8c30`).
- Manifest: `results/g2b_public_formal_manifest.json` (SHA256
  `d1d85dad54250989d24dbf0492e125e068df28f9a1bb85e1d4adbabcb7ee3a61`).
- Safety: `DECISION_G2B_SAFETY.md` and
  `results/g2b_tensorjoin_safety_manifest.json`.
- Formal source/binary snapshot: `artifacts/g2b_public_formal/source_snapshot/`.
- Hash ledger: `receipts/g2b_public_formal_evidence_sha256.txt` (SHA256
  `cda1cfc50e3e73d0b3ee2a63ec0242c35931f4c9400324e8cf27734c4a5bd34f`);
  all 216 listed files were reverified after synchronization to the local
  workspace.

## Strategic consequence

G2B closes the external exact-comparison requirement for this one frozen cell.
It promotes the middle strategy from an in-house kernel result to a defensible
database/systems research direction. It does **not** repair the Gate-0 novelty
failure of the broad "first exact low-precision search" thesis. The next
necessary evidence is breadth and mechanism attribution: multiple standard
vector datasets/dimensions/selectivities, output-sensitive scaling, and a
router/cascade ablation or cost-model test. More modality ports alone remain
insufficient.
