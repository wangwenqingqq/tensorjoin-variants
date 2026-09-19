# Protocol: H1 Exact GPU Multi-Vector Operator Screen

## Frozen hypothesis

Object-first compaction makes exact multi-vector threshold join profitable on
the target GPU: an INT8 Tensor Core token certificate plus FP64 recomputation
of only compacted dense object panels will be exact and at least 1.25x faster
than exhaustive direct-FP64 token evaluation in all four frozen H0 cells.

## Immutable inputs

- H0 result: `results/h0_multivector_certificate.json`, expected SHA-256
  `7d8b8e68c3ecaa6bb38de133e83afefcab4ddefa4b375572ba09fe3e9a68af31`.
- Audio cache: `data/esc50_panns_cnn14_1s_2048d.npz`.
- Video cache: `data/ucf101_r3d18_512d.npz`.
- Object split, normalization, score, and thresholds are exactly H0's.
- Physical GPU: 1 only, exposed as the sole device with
  `CUDA_VISIBLE_DEVICES=1`.

The runner must verify all source and data hashes before execution and refuse
to overwrite an existing result.

## Frozen implementation scope

Candidate timing includes:

1. zeroing counters;
2. full INT8 token-certificate kernel;
3. interval materialization;
4. all 65,536 object interval reductions and compaction;
5. device-to-host ambiguity count synchronization;
6. direct-FP64 calculation of every valid cell in each ambiguous padded 8x8
   object panel;
7. exact object reduction and result compaction;
8. final count and canonical object IDs copied to the host.

Keeper timing includes counter reset, exhaustive direct-FP64 token distances,
all object reductions, and final count/canonical IDs copied to the host.

Both exclude cache ingest, deterministic split, normalization, quantization,
allocation, compilation, independent CPU oracle generation, and one-time
validation.  All GPU inputs and workspaces are resident before timing.

## Correctness and opportunity records

For every dataset/threshold cell record:

- CPU, keeper, and candidate canonical ID count and SHA-256;
- candidate direct-accept and ambiguous object counts;
- ambiguous object fraction and padded/valid FP64 refinement cells;
- token and object lower/upper containment violations;
- unsafe direct accepts/rejects;
- duplicate, overflow, keeper mismatch, and candidate mismatch counts;
- maximum keeper versus CPU object-score error.

Exactness requires every count above to be zero except the explicitly reported
opportunity counts/fractions and numeric score-error magnitude.  The keeper and
candidate must return identical canonical IDs to the CPU oracle.

## Safety gate

Run NVIDIA Compute Sanitizer `memcheck` separately for keeper and candidate on
bounded actual-data 4-query-by-8-base invocations from both datasets after full
correctness passes.  Together these cover the D2048 fixed-cardinality path and
the D512 ragged 4--7-token path without turning memcheck into an exhaustive
multi-billion-coordinate run. Compilation success is not a safety result.
Missing sanitizer availability blocks timing promotion but does not convert a
correctness result into a failure.

## Timing screen

- CUDA events delimit the GPU work, while required host synchronizations and ID
  copies are separately charged with an outer monotonic wall timer.  The public
  screen denominator is the outer operator wall time.
- 10 untimed warmups and 50 retained observations per variant/cell.
- Alternate order KC then CK across the two repeats; retain raw order.
- Report p10/median/p90 for each variant and median keeper/candidate ratio.
- Pass only if candidate median is lower and speedup is at least 1.25x in every
  cell, after correctness and memcheck pass.

No NCU result is required for H1.  If timing fails, preserve the exact workload,
keeper, rejected mechanism, added materialization/synchronization costs,
measured effect, diagnosis, and reopen condition.

## Shared-GPU execution

Before every run, inspect all GPUs and compute processes.  Never stop another
user's process.  Acquire `@TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock`
with `flock` for correctness, sanitizer, and timing.  The intended command is
run only when physical GPU1 is idle.  The runner independently resolves GPU1's
UUID and refuses to start if `nvidia-smi` reports any foreign compute PID on
that UUID; the advisory lock alone is not treated as ownership evidence.
