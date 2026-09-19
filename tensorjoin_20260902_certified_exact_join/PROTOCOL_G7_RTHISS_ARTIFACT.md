# G7 RT-HiSS artifact admission protocol

Frozen: 2026-09-04, before adapter edits/build/runtime tests.

## Decision and scope

The RT-HiSS public artifact was located through the author's official People
page and GitHub repository inventory. The earlier G6 "not found" observation
is superseded for current availability; preserve G6 as historical evidence.
This gate obtains and builds the official implementation before considering any
performance campaign. It does not change the TensorJoin keeper or paper thesis.

## Source and Gate 0

- RT-HiSS: https://github.com/revanthmunugala/rt-hiss
- Commit: a42fc69cc4b602dc83071b029d185a41e69a04bd; MIT license.
- Pinned OWL: c7c3a3ea35b17b5c096a3802ba74b9d8b4e2772a.
- The mechanism is RT-core 3D primitive filtering followed by FP32 CUDA
  refinement and compressed result transfers. The source uses sequential
  `fmaf` refinement, FP32 epsilon and squared threshold. This is direct prior
  art for GPU exact-algorithm joins, but does not establish a numerical
  certificate for the FP64-oracle contract. Numerical compatibility is unknown
  until tested; FP32 alone is not evidence of a failed result.
- COSS 2023 follow-up adds intrinsic-dimension-guided parameter/algorithm
  selection. Generic dimension-based index selection is therefore not a new
  contribution. This does not change the locked precision-routing thesis.

## Host and safety card

- Host: gpu-host-8 (gpu-host-8), account root, user-owned workspace
  @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join.
- Target: physical GPU1, GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3,
  RTX PRO 6000 Blackwell Server Edition, driver 590.48.01, CUDA 13.1.115.
- Shared campaign lock: @TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock.
- Recheck GPU UUID, utilization, compute processes and memory before every run;
  require no compute process, zero utilization and <128 MiB on GPU1.
- Never touch GPU0 or GPUs4--7 foreign work. Do not kill any foreign process,
  restart services, change clocks, install global packages or change drivers.
- Builds use at most four CPU jobs. Dependencies and changes are project-local.
- Existing OptiX 9.1 headers at RT-TIDE/deps/optix-9.1 may be used read-only;
  verified clean commit f1f6dd803f3159992d248178f6e09421c6eb8b6d.
- Current storage is 98% occupied with 187 GiB available; do not download the
  1.76 GB upstream dataset archive in this gate. Use the included small sample
  and existing G2A public data only.
- Raw evidence: raw/g7_*; source receipts: artifacts/g7_artifact_audit/.
- Rollback: stop only this gate's recorded process group, leave keepers and
  upstream checkout untouched; archive the separate adapter and result state.

## Gate order and allowed modifications

1. Pin/check source, license, OWL and OptiX identities. Acquire only public
   source. Record download failures rather than equating them with absence.
2. Build a separate adapter using C++17 and CUDA compute_120/sm_120. Correct
   upstream CMake's unconditional architecture=75 assignment to respect a
   cache override. Dependency API-only compatibility patches are permitted
   with retained diff; do not change search arithmetic, primitive geometry,
   pruning or batching policy without a new protocol revision.
3. Upstream sample smoke: N=1000, D=2, epsilon=0.01, default variance reorder
   and default shared/shared batching. One process, 180-second runtime cap.
   A clean completion/count is only a build/runtime smoke, not exactness or
   public latency evidence. Abort on CUDA/OptiX errors or allocation faults.
4. Only after smoke, design the pair-export/G2A adapter in a separate revision.
   It must preserve compressed-mask behavior and decode into sorted directed
   uint64 IDs (i*N+j), include self, retain output IDs through all reordering,
   and validate no duplicates/invalid IDs plus exact hash.
5. G2A: N4096 D512, epsilon=0.7541135250198396, threshold_d2=
   0.5686872086178483, oracle count262144 and pair hash
   da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d.
   This is only a future admission target, not permission for timing promotion.
   Any disagreement stops exact admission and triggers numeric/export diagnosis.
6. No G2B or formal timing until exact output, same denominator and required
   safety gates are frozen and passed. Never compare the upstream internal
   time/count-only sample to TensorJoin end-to-end output latency.

## State transitions

Artifact located -> dependencies pinned -> built -> sample smoke -> output
adapter designed -> exactness -> safety -> separately frozen timing campaign.
Compilation, sample counts and FP64-oracle equality are distinct gates.
Missing dependencies yield unvalidated, not rejected-algorithm status.

## Paper/source state

The G6 evidence manifest and six-page Overleaf PDF are historical snapshots.
No G6 manifest-covered file is edited by this gate. The paper's statement that
no RT-HiSS artifact was found is now stale and must be corrected in the next
explicit source/compile/sync revision; this protocol does not claim it is synced.
