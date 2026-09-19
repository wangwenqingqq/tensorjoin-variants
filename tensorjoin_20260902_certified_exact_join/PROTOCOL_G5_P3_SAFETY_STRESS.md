# G5 P3: Same-Specialization Safety and Stress

Date: 2026-09-03

## Admission evidence

G5 P1 reproduced the complete exact output and selected these full-N cubins:

| Function | Cubin SHA-256 |
|---|---|
| `analytic_certificate_ragged_safe_i64` | `abe3e1403e839ef60f13af32d546593f451becf1488bb5b11b940e57b00e992d` |
| `certified_fp32_filter_i64` | `1278324244afd13961e75cbd0a554a9bb9502c8e61193d361c015b7856208e20` |
| `refine_ambiguous_fp64_i64` | `c2b3362119abfbeb3df85e96788705f6ff4c79414e3a74e2d5d5f3355eff00f8` |

P2 confirms 16 low-precision MMA instructions in stage 1, no low-precision or
FP64 arithmetic in stage 2, 27 FP64 arithmetic instructions in stage 3, and
zero stack/local/LDL/STL in all three functions.  P2 result SHA-256 is
`ffcfdf9c4949756ad1de8974f7e4453378dc26690abba3c292a551553c48482b`.

## Frozen safety tile set

Run these upper-triangle 64 x 64 tile coordinates against the full 60,000 x
512 source:

```text
(0,0), (0,790), (1,13), (1,561), (1,576),
(1,716), (1,864), (2,251), (2,560)
```

Eight cross tiles were selected because P1 observed a residual-FP64 pair in
each; `(0,0)` guarantees diagonal/self coverage.  Tile selection occurred
after P1 only for safety coverage and cannot enter public timing.

## Binary identity

Every safety launch retains exactly:

```text
N_=60000, K=512, CAPACITY=16777216
BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
FP32_BLOCK_K=256, FP64_BLOCK_K=256
num_warps=4, stage1_num_stages=3
```

The runner allocates two complete fixed-capacity buffer sets.  A smaller launch
grid is permitted, but any cubin or PTX hash change from P1 is a failure.

## Independent oracle and execution

Build an independent CPU direct-FP64 oracle for every valid upper pair in the
nine selected tiles using float32-to-float64 widening and squared differences.
For every pipeline invocation, compare the sorted GPU upper IDs directly with
this oracle.

Run on isolated physical GPU2 under `/tmp/tensorjoin_g5_gpu2.lock`:

1. two iterations under `compute-sanitizer --tool memcheck
   --target-processes all --error-exitcode 99 --leak-check full`;
2. 1,000 iterations in a fresh process, alternating two complete buffer sets.

## Gate

Require all of the following:

- `ERROR SUMMARY: 0 errors` and sanitizer return code zero;
- zero oracle-output, stage-count, duplicate, invalid, or overflow mismatch;
- one stable stage-count signature across every iteration;
- positive stage-1 direct, stage-1 ambiguous, FP32 direct-accept,
  FP32 direct-reject, and residual-FP64 counts;
- exact P1 cubin/PTX hashes in both fresh caches;
- zero PyTorch allocated/reserved bytes after explicit cleanup;
- no foreign GPU2 process and an empty postflight.

P3 is correctness/safety/stability evidence only.  It does not contribute a
performance number.

