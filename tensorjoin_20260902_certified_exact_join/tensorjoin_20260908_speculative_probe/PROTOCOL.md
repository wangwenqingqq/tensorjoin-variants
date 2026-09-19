# Certified speculative execution: bounded probe

Frozen before implementation timing, 2026-09-08. Run on gpu-host-8 GPU2 under
the inherited exclusive guard (30 seconds idle, <4096 MiB, four CPU threads,
<=1800 seconds per process). Never modify prior experiments or foreign jobs.

The input is the frozen 60000x512 represented FP32 CIFAR input, with the six
inherited thresholds. Reuse the input-only PCA layout and 64-dimensional
projection metadata of candidate_execution. Keep its conservative numerical
cutoff, FP16 interval formulas, frozen FP32 refinement and FP64 terminal, ID
restoration and complete sorted directed CPU output, including self pairs.
The main scope is a reused CPU index, with fresh GPU upload and metadata on
every query; construction is measured separately. No GPU-resident-query claim.

Let U be the set of 64x64 tiles containing at least one conservatively undecided
projected pair. Draft D samples rows and columns 0,4,...,60 in each tile and uses
the same 64 projection coordinates and cutoff on those 16x16 point pairs.
This is a fixed heuristic proposal, not an independent rejection certificate.
Every tile outside D is still checked by the full reliable filter. Repair is
U minus D. D and repair are disjoint; their union covers U. All executed tiles
use the same complete numerical solver. No prediction is committed as an answer.
No model training, oracle masks, result-dependent thresholds or timing tuning.

Each work chunk consists of at most 4096 consecutive upper-triangular tiles.
GPU compaction creates tile queues and device counts for each chunk. A guarded
FP16 kernel launches the fixed chunk capacity and returns before distance
calculation for inactive queue positions. One shared output/ambiguous buffer
set per query; draft and repair append sequentially on the compute stream and
share a single refinement/terminal round per input chunk. Thus speculative
work never duplicates tile results. Filtering uses a separate stream only in
the two concurrent modes. Dependencies use CUDA events; retain all referenced
buffers until both streams finish.

Four primary arms, same numerical kernels and fixed input chunks:
- sequential: generate all reliable queues, then execute them.
- pipeline: ordinary streaming filter, at most two verification chunks issued
  ahead of current computation; wait for current reliable queue before scan.
- draft_serial: generate all draft and repair queues, then execute draft and
  repair with no overlap.
- speculative: generate draft queues; use the same ordinary verification
  lookahead as pipeline, but allow current draft scan before waiting for current
  verification, then scan repairs. Full verification still visits every tile.

Three inherited strong controls are measured in the same processes:
original_full FP16, project64 (globally compacted reliable tile queue), and
packed16 (verified active-row/column packing). These prevent a favorable
comparison caused only by the new chunking or queue implementation.

Admission compares every full output array to the inherited FP64 reference;
then compare all reliable tile flags/counts to the inherited filter, verify
queue uniqueness/completeness/disjointness, and independently check sampled
projection distances on fixed tail/diagonal/random tiles. Force empty, full
and alternating draft masks at the original threshold to test repair extremes.
Check output/ambiguous canaries around the fixed-capacity buffers. Preserve
failed attempts; functional fixes are permitted before timing freeze.

After admission and dependency/source freeze: three fresh processes, one
untimed warmup per cell/arm, four retained repetitions each, alternating and
rotating order. No sample deletion. Report complete times, per-process medians,
six-threshold equal-weight mean times, and observed process ratio ranges.
Primary passing criterion: speculative saves at least 5% versus pipeline AND
the strongest fixed inherited control on the six-threshold mixture in each
confirmation process. Also report comparison to draft_serial and every point.
If the criterion fails, stop this fixed prototype rather than tune a gain.

Separate diagnostics record draft coverage, unnecessary draft tiles, repair
counts, executed capacity, stage counts, GPU event intervals and a CUDA trace
when available. Concurrent stream intervals alone do not prove concurrent
kernel execution. Profiling and numerical verification are outside timing.
Measure one actual build-plus-query diagnostic for each primary arm; no
training or construction is silently excluded from a cold-query claim.

Only one previously explored data set and one GPU: a positive result warrants
new data and prior-art comparison; it is not a novelty or generalization claim.
