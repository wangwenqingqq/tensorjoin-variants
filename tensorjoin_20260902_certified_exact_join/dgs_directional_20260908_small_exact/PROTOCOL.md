# DGS-inspired directional-envelope structural screen

Experiment: tensorjoin_20260908_directional_envelopes_small_exact
State: designed; frozen before new implementation or observations.

## Gate 0 and boundaries

This is a cheap falsification test, not an original algorithm claim. PathWeaver
DGS (ATC 2025, section 3.3) motivates eliminating vector reads and distance
computations before refinement. Its direction top-n selection is approximate
and cannot replace the fixed FP64 predicate. VA-file (VLDB 1998, section 4)
already establishes approximation-bound filtering. GTS (SIGMOD 2024) already
establishes GPU pivot-tree pruning and batching. No novelty pass is presumed.
Prior local E0 rejected a fanout-8 pivot-distance tree on audio/video. Do not
retune it. This experiment changes the bound to coordinate/directional
intervals and tests the current 60K data, not those old caches.
Sources: https://www.usenix.org/system/files/atc25-kim.pdf ;
https://www.vldb.org/conf/1998/p194.pdf ; https://arxiv.org/abs/2404.00966 .

## Contract

- Original 60000 x 512 finite FP32 CIFAR-derived GIST input, abs coordinate <=1.
- Input NPY SHA256 95048090f7834759a0f0fffb41a663807f8ef1c2255a5412f045071fdbd6198c.
- Squared threshold 25921/65536, inclusive; original fixed FP64 direct-difference
  predicate, directed output including self. No recall relaxation.
- Block size 64; all 938*939/2=440391 upper-triangle blocks, including diagonal
  and last ragged block. No new block-size tuning in this screen.
- Orders: original; deterministic balanced axis tree in raw coordinates;
  deterministic balanced axis tree in unnormalized Walsh-Hadamard coordinates.
  At each node choose largest variance axis, stable-sort coordinate then ID,
  split at floor(ceil(size/64)/2)*64. Leaves align with physical 64-row tiles.
- Bounds: exact raw-coordinate interval envelope and fixed 512-D unnormalized
  Walsh-Hadamard interval envelope, both stored as outward FP64 intervals.
  Test 16/64/512 coordinates ordered by global variance within each transform;
  prefixes are nested and predetermined, not selected using join answers.
- For blocks A,B, g_j=max(0,lo_A-hi_B,lo_B-hi_A).
  L=sum(g_j^2)/scale with scale=1 (raw) or 512 (Hadamard).
  Orthogonality guarantees L <= every real squared pair distance. Combine raw
  and Hadamard by max, NOT sum. Extend intervals for transform roundoff and
  lower the computed sum for FP64 rounding. Keep a separate conservative
  original-oracle roundoff allowance. Never prune on direction match alone.
- Main outputs: bound-pruned whole tiles, fraction of scheduled tiles/padded
  cells saved, logical valid pairs eliminated, preprocessing/filter CPU wall
  time (diagnostic only), and false negatives against available admitted full
  output IDs. Save masks, order permutations, summaries and all counts.
- If full IDs are not retained, regenerate once using unchanged admitted F16
  full operator on GPU2 under the existing guard. Verify count 3926078 and
  SHA256 13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495.
- Also compute ideal empty-answer-tile ceiling from full output IDs. It uses
  ground truth and cannot be a deployable filter or a speedup claim.

## Resource and execution card

Host gpu-host-8, hostname gpu-host-8; root under existing research-user workspace.
CPU NumPy FP64, at most 4 BLAS/OpenMP threads; this is the first stage.
Only GPU2 if subsequent validation/cost stage is admitted: RTX PRO 6000
Blackwell Server, UUID GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245, driver590.48.01,
torch2.11.0+cu130, NumPy2.3.1, Python3.12.3. Both existing GPU2 locks, 30s idle,
4GiB GPU allocation ceiling, at most30 minutes/run. Never touch other processes,
clocks, power limits, old kernels or manifests. New artifacts only; no git repo
at project root. Record exact command/PID/environment/source hashes at launch.

## Decision and measurement limits

- The structural pass requires zero false negatives and at least 20% whole
  64x64 tiles pruned on at least one declared order/bound. This admits only a
  bounded GPU complete-cost prototype, NOT novelty or performance promotion.
- If all full-coordinate bounds miss 20%, stop this fixed directional-envelope
  frontend without CUDA implementation. Report nonzero evidence faithfully;
  do not reject every possible safe filter from this particular failure.
- If admitted, remeasure matched fused FP16 in the same campaign; charge all
  transform/index construction, summary screening, gathering, launch and output
  costs. Keep resident/reusable-index diagnostics separate from fresh input.
- No GPU filter exists yet. SASS/sanitizer/Graph/stress/performance gates for
  such a filter are NOT run and must not be implied by this CPU screen.
- Unit tests: constructed separable and boundary cases, transform-distance
  consistency, envelope membership and random pair lower-bound checks.
- CPU timings include preprocessing and filter in separate rows, not a GPU
  latency claim. A failure or missing oracle artifact stays in raw evidence.
- Negative-result reopen condition: a genuinely tighter safe representation or
  changed justified task predicate, with a new predeclared structural screen.
