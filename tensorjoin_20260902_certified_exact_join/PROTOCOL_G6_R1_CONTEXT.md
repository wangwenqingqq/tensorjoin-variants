# G6-R1 full-scale context-only diagnostic

Date frozen: 2026-09-04 Asia/Shanghai

Parent: `PROTOCOL_G6_LATEST_BASELINES.md`.

Status: frozen after G2A exactness rejection and before any G2B observation
from GTS or cuVS.

## Why this revision exists

The parent exact-admission gate correctly stopped both available baselines:

- upstream-semantics GTS emitted 262,146 rather than 262,144 G2A pairs.  Its
  sole unordered extra pair flips from reject in direct FP64 to accept in
  upstream-style sequential FP32 accumulation;
- cuVS 26.8.1 emitted 262,140 pairs.  Its two unordered missing pairs are
  accepted by direct FP64 and direct FP32 reductions but rejected by cuVS's
  256-query-batch squared-Euclidean result.  Single-query cuVS accepts them,
  localizing a shape-dependent numerical path.

Neither implementation is an exact keeper under the frozen float32-source to
float64-oracle contract.  However, a bounded full-scale run is still useful for
answering the separate question: what speed/accuracy point do these current
available implementations occupy on G2B?  R1 permits that context only.  It
does not weaken or replace the exactness gate.

## Frozen execution

1. Run cuVS 26.8.1 once on G2B with the unchanged parent runner, `k=4096`,
   256-row query batches, physical GPU1, and the unchanged public denominator.
2. Run GTS once only after the cuVS postflight returns GPU1 to an unoccupied
   state.  Preserve upstream FP32 search semantics and the G6 export adapter.
3. GTS receives a 30-minute outer-wall screen budget.  If it does not complete,
   terminate only the launched process, retain the timeout log, and report a
   bounded non-completion rather than latency.
4. Do not repeat, tune, or change query batch size based on the observed times.

## Required reporting

For every completed baseline, report:

- public elapsed seconds, with the explicit one-run diagnostic label;
- emitted count and canonical output hash;
- exact-only and baseline-only pair counts;
- pair precision, recall, and F1 against the frozen 3,926,078-pair oracle;
- duplicates, invalid IDs, maximum accepted row count, and `k` saturation when
  applicable;
- package/source, runner, input, output, log, GPU UUID, and pre/post receipts.

Any timing attached to a non-exact result must be displayed together with its
accuracy and must not enter an “exact speedup” column.  An unexpectedly exact
G2B result would still be diagnostic because this protocol has only one process
and no direction-balanced comparator.

## Safety

Hold `@TENSORJOIN_ROOT@/.tensorjoin_g6_gpu1.lock`, require physical GPU1 to
have no foreign compute process immediately before each run, and do not touch
any foreign process or another GPU.  Preserve every rejected and timed-out
attempt.  Disk input/output and post-return accuracy diagnostics remain outside
the public timed boundary exactly as in the parent.

