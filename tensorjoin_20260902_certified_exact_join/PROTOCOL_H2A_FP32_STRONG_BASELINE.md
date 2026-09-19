# Protocol: H2A Certified FP32 Strong-Baseline Screen

## Immutable evidence dependencies

- H1 correctness SHA-256 `06461064d1b237aaa29f4face0eb3c223f5487abf478ddcf5fccb353ed28dca4`.
- H1 safety SHA-256 `f022cc90db19b295dd481a5b7946f3690b0cd935ba398befb31dab8218305d09`.
- H1 timing SHA-256 `c98d0e33c95ff1707caee5a1b37cd345fbbb1dabe239644a1f82582b8c1f3173`.
- H1 runner/kernel hashes `11aaa724...0713e` / `c24d6e94...b0db`.

The H2A runner must reject missing or changed dependencies and refuse to share
physical GPU3 with another compute PID.

## Phases

1. Full four-cell correctness and token/object containment against the H1 CPU
   oracle contract.
2. Compute Sanitizer memcheck for the new FP32 baseline on 4x8 actual audio and
   video object subsets.
3. Immutable safety summary with source/log hashes.
4. Only after 1--3 pass, 10 warmups plus 50 alternating retained observations
   per variant and cell.
5. Generated-code/resource audit after timing; static code is mechanism
   evidence only.

Every phase refuses overwrite. Raw failures remain under their attempt name.

## Charged work

Both variants charge counter reset, full token-interval materialization, all
65,536 object reductions, compaction atomics, the host-visible dynamic count,
all padded direct-FP64 refinement cells, exact object reduction, final ID copy,
and canonical sort. Cache ingest, normalization, quantization, allocation, and
compilation are excluded and happen before timing.

## Stop rule

Any exactness or memcheck failure rejects H2A. Performance passes only when the
INT8 candidate reaches at least 1.25x median outer-wall speedup and 50/50 wins
against certified FP32 in every cell. No per-cell rescue, threshold retuning,
or post-measurement radius change is allowed.

