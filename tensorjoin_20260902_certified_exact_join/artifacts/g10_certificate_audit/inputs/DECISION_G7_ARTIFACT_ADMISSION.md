# G7 Decision: Official RT-HiSS and COSS Artifacts Located

Date: 2026-09-04

## Conclusion

**RT-HiSS is publicly available and now builds and runs its small upstream
sample on the 8P SM120 host. COSS public source was also located and pinned.**
Neither is admitted as a same-contract exact comparator yet. No new speedup
claim is justified by G7.

This corrects a retrieval miss in G6, not a newly published code release. The
RT-HiSS commit is dated July 20, 2026; the COSS commit is dated July 12, 2020.
The earlier "no public artifact found" statement is no longer current. G6 raw
results and its 57-file evidence manifest remain unchanged and verify locally.

## Artifact inventory

| System | Source identity | License observed | Current gate |
|---|---|---|---|
| RT-HiSS | https://github.com/revanthmunugala/rt-hiss at a42fc69cc4b602dc83071b029d185a41e69a04bd | MIT | Build and upstream N1000/D2 smoke passed; pair-output exactness not tested |
| OWL | https://github.com/owl-project/owl at c7c3a3ea35b17b5c096a3802ba74b9d8b4e2772a | Apache-2.0 | Upstream RT-HiSS gitlink; compiled with installed OptiX 9.1 headers |
| COSS | https://github.com/bwd29/Coordinate-Oblivious-Similarity-Search at c9f0f4fcf2ac96c0f2f1aee547bce7040400d911 | No LICENSE or SPDX declaration found in its two tracked files | Source inspected only; no build or runtime test |

COSS is described here as **public source**, not as a confirmed open-source
licensed artifact. Do not redistribute it as part of an AE/open-source bundle
without resolving permission. It is a 2020 base-code upload, not a verified
artifact for the ICCS 2023 parameter-selection follow-up.

The discovery route was the official author page
https://jan.ucc.nau.edu/mg2745/people/ -> Revanth Munugala's GitHub -> complete
public repository inventory. Brian Donnelly's account was recovered from the
already cloned MiSTIC origin. The local API/profile requests for three other
authors repeatedly failed at transport; the same read-only API queries from
gpu-host-8 succeeded. Network failures must not be classified as repository
absence. Receipts retain both failed and successful paths.

## Novelty and contract consequences

RT-HiSS is direct algorithmic prior art: RT-core three-dimensional primitive
filtering, high-dimensional CUDA refinement, batching and compressed output.
Its public code uses FP32 epsilon/squared epsilon and sequential `fmaf` distance
accumulation (`utility.cu:338--364`); this does not by itself prove disagreement
with the frozen oracle. The numerical gate remains unknown until pair IDs are
compared. Neither an FP32 label nor a successful sample count is an exactness
verdict.

The 2023 paper *Optimization and Comparison of Coordinate- and Metric-Based
Indexes on GPUs for Distance Similarity Searches* adds intrinsic-dimensionality
based parameter tuning and selection between GDS-Join and COSS. Therefore,
generic dimension/ID-guided index selection cannot be claimed as new here. Its
paper evaluation uses FP64 and discusses stronger GDS-Join optimizations; the
paper's public-source footnote points to GDS-Join, not to a separate COSS2023
release. Source: https://jan.ucc.nau.edu/mg2745/publications/Gowanlock_ICCS2023.pdf
(DOI: https://doi.org/10.1007/978-3-031-36021-3_37).

The locked TensorJoin thesis remains certified, output-sensitive precision
routing, with its existing same-artifact and scope limitations. G7 neither
establishes new novelty nor removes G5's negative evidence.

## RT-HiSS build and runtime evidence

- Separate adapter: `adapters/rthiss_g7_a0/`. The only upstream source change is
  the CMake architecture assignment: a cache-respecting override replaces the
  unconditional architecture 75 value. No algorithm, epsilon, precision,
  pruning, compression or CUDA kernel was modified.
- Full OWL submodule clone failed twice with incomplete-transfer errors.
  The exact pinned commit archive was obtained instead; SHA-256
  `c0527c64505e9c33effba98643e07416abfb11be6b94290e5f4e5c8117dc8923`.
  The archive contains the bundled dependencies and no nested gitlink was
  required for this build.
- Initial CMake configure failed to infer the CUDA library root from the
  `/usr/local/bin/nvcc` symlink. A separate a1 build uses the resolved
  `/usr/local/cuda-13.1/bin/nvcc`. No global configuration was changed.
- CUDA 13.1.115; GCC13.3; CMake3.28; installed read-only OptiX9.1 headers at
  clean commit `f1f6dd803f3159992d248178f6e09421c6eb8b6d`.
- CUDA refinement: compute_120/sm_120. Embedded OptiX PTX: version9.1,
  target sm_75, as emitted by unmodified upstream OWL; driver JIT is distinct
  from the CUDA refinement architecture.
- Executable SHA-256:
  `543229bbb6e3dbb174cb10d5dd81bc2388592c53938252d042ec7293d4f2e64b`.
- Linked libowl SHA-256:
  `df75c1bf36524381a0b9869c03e3d82e0bff4e62362a64300c5ff0d4419b1ffa`.
- GPU1 a0 admission was skipped when a fresh check found a foreign process.
  No RT-HiSS child or GPU work was started on GPU1.
- Predeclared routing revision R1 used free physical GPU7, UUID
  `GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603`. The same binary ran once with
  the shipped 1000x2 sample, epsilon0.01, highest-variance reorder and default
  shared/shared refinement. PID927824 exited0 and reported1160 neighbors,
  matching the README's example count. No pair identities were exported.
- No CUDA/OptiX failure or out-of-bounds marker appeared in the smoke log.
  This is not a sanitizer result. GPU7 returned to14MiB and no compute process;
  a subsequent postflight showed0% utilization. No foreign process was touched.
- `results/g7_rthiss_smoke_a1.json` explicitly sets `exactness=not_tested` and
  `public_time_seconds=null`. Upstream internal timers and process wall time
  are retained only as operational evidence, not performance comparisons.

## COSS source-audit admission issues

The original source uses double-precision data, reference distances and
refinement, so it is a relevant candidate for the FP64 oracle. However, several
implementation contracts must be resolved before running it:

1. The makefile targets compute_60/sm_60; CUDA13/SM120 requires a recorded build
   port and C++17/CCCL compatibility check.
2. The raw-input reader requests `size*sizeof(double)` bytes after allocating
   only `size` bytes (`Search.cu:698--699`). For an unchanged regular file EOF
   limits the actual read, but the request and error handling are incorrect;
   do not label this an observed buffer overrun.
3. A `neighborTable` contains `std::vector` members but the table array is
   allocated with `malloc` and assigned into without constructing its elements
   (`Search.cu:1061--1069`). This is a source-level C++ object-lifetime defect,
   not a measured crash. Use explicit construction in a separately diffed
   compatibility adapter before runtime admission.
4. Pair buffers use a hardcoded15000-neighbor assumption and per-stream int
   offsets; device writes are not capacity-guarded. Freeze workload/buffer
   limits and audit offset arithmetic before any large run. The computed
   `available_mem` value is not the actual allocation rule.
5. The printed time starts after dimension reorder/reference-point setup and
   ends before canonical output materialization. It is not the G2B public
   denominator. Input FP32->FP64 conversion, preprocessing, allocation, output
   mapping and sorting must all be normalized and charged appropriately.
6. Source initializes self entries separately from the search kernel and
   reorders points. Pair export must validate original IDs, self inclusion,
   direction, duplicates and bounds, not just the printed count.

No COSS kernel performance, exactness failure or algorithm weakness is inferred
from these unexecuted source findings.

## Claim-evidence ledger

| ID | Exact claim | Status | Evidence | Allowed wording |
|---|---|---|---|---|
| G7-C1 | Official RT-HiSS and COSS public code is retrievable at pinned commits | measured | author repository responses, git provenance, source snapshots | Both public artifacts were located; G6 retrieval gap is corrected |
| G7-C2 | RT-HiSS runs on the 8P SM120 host | partial | sample a1 JSON/log; executable and libowl hashes | Pinned RT-HiSS passed one N1000/D2 count-only smoke with OptiX9.1 on GPU7 |
| G7-C3 | RT-HiSS matches the G2A FP64 pair oracle | unknown | no evidence | Not tested; G2A pair-output adapter pending |
| G7-C4 | RT-HiSS or COSS performance relative to TensorJoin | unknown | no same-contract campaign | No new comparative timing claim |
| G7-C5 | COSS runtime correctness on SM120 | unvalidated | static source audit only | Compatibility and safety work remains |

## Next gate

Design and implement a separate RT-HiSS pair-export adapter, preserving the
compressed-mask algorithm. Retain original point IDs through all reordering,
decode actual accepted bits into canonical host IDs, and validate the decoder
on tiny boundary/duplicate/identity cases before the frozen G2A4096x512 oracle.
All added metadata transfers, decoding and sorting belong in a future public
denominator. Do not choose the upstream key-value variant merely to avoid
normalizing the default compressed-output implementation.

COSS then needs its own compatibility/safety protocol before build/runtime.
No G2B run is admitted by this artifact gate.

## Paper synchronization boundary

The G6 source ZIP, six-page compiled PDF and manifest remain frozen historical
artifacts. The active Overleaf draft still contains the now-stale artifact-
availability statement. **No Overleaf edit, compile or sync was performed in
G7.** At the next paper revision, replace "artifact not found" with the pinned
public-source reference and the exact measurement/admission state. Do not call
RT-HiSS measured on G2B based on this sample smoke.
