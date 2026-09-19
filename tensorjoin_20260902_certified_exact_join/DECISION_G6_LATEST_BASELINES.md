# G6 Decision: Latest Available Baselines

Date: 2026-09-04

## Decision

No newly evaluated baseline is admitted as a same-contract exact keeper.
TensorJoin's formal exact comparison therefore remains against MiSTIC and
GDS-Join.  GTS and cuVS are retained as measured numerical-context points, and
RT-HiSS remains the newest direct prior art but is not measured because no
public artifact was found.

This result narrows rather than expands the paper claim.  TensorJoin may claim
certified, output-sensitive precision routing under its explicit FP64-oracle
contract; it may not claim the first or universally fastest exact GPU
similarity join.

## Frozen contract

- Workload: CIFAR-GIST-512 self-join, `N=60,000`, `D=512`,
  `epsilon=0.62890625`.
- Oracle: 3,926,078 sorted directed IDs, SHA-256
  `13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495`.
- Public denominator: pageable-host FP32 input through sorted canonical host
  `uint64` IDs; disk I/O, context creation, compilation, file write, output
  hash, correctness comparison, and JSON serialization are excluded.
- Hardware: isolated physical GPU1,
  `GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3`, NVIDIA RTX PRO 6000 Blackwell
  Server Edition.  No foreign process was touched.
- Because both baselines failed the predeclared G2A exactness gate, G2B is one
  context-only run per baseline.  Its times are not formal exact speedups.

## Measured result

| Baseline | Frozen identity | G2A exactness | G2B output vs. oracle | G2B one-run public time | Admission |
|---|---|---:|---:|---:|---|
| cuVS brute-force squared Euclidean | `cuvs-cu13==26.8.1` | -4 pairs | 24 FP, 26 FN; precision 0.999993887; recall 0.999993378; F1 0.999993632 | 3.551750 s | rejected as exact; context only |
| GTS | upstream `3bac1b7`, G6 export adapter binary `8c887e2d...960e6` | +2 pairs | 4 FP, 0 FN; precision 0.999998981; recall 1.0; F1 0.999999491 | 1111.520 s | rejected as exact; context only |
| RT-HiSS | SC 2026 / arXiv 2609.01975 | not runnable | not measured | not measured | required direct prior art; artifact pending |

Both executable baselines emitted sorted valid IDs with no duplicates.  cuVS
had no `k=4096`-saturated row; its maximum accepted row count was 2,855.
GTS completed before the frozen 1,800-second cap and returned GPU1 to 14 MiB
and zero utilization.

For scale intuition only, the cuVS run is 3.835x the historical TensorJoin
G2B median and the GTS run is 1200.147x that median.  These are descriptive
ratios between a one-run non-exact context point and an older formal median;
they are forbidden from any exact-speedup column.

## Numerical diagnosis

### GTS

GTS preserves its upstream FP32 search arithmetic.  On G2A, the only unordered
extra pair is outside the FP64 radius by `1.5309e-7` in distance but inside by
`1.4888e-7` under upstream-style sequential FP32 accumulation.  On G2B, its
four directed false positives are two unordered boundary pairs: their FP64
squared-distance margins are `+1.6187e-7` and `+3.3069e-8`, whereas sequential
FP32 margins are negative.  The adapter produced no duplicate or invalid ID;
the failure is the arithmetic contract, not pair serialization.

### cuVS

cuVS G2A missed two unordered pairs.  Direct FP64 and direct FP32 reductions
accept both, and single-query cuVS accepts them, but the frozen 256-query-batch
squared-Euclidean path reports the same above-threshold value for all four
directions.  The mismatch is localized to a shape-dependent cuVS distance
path, not host filtering, top-k saturation, or sorting.  G2B confirms that the
difference is bidirectional rather than a simple conservative bound: 24 false
positives and 26 false negatives.

## Evidence status

Established by retained artifacts:

- GTS authoritative G2B result: `results/g6_gts_g2b_r1_a0.json`;
- GTS G2B boundary diagnosis:
  `results/g6_gts_g2b_r1_a0_mismatch_diagnosis.json`;
- cuVS G2B result: `results/g6_cuvs_g2b_r1_a0.json`;
- G2A diagnoses: `results/g6_gts_g2a_a1_mismatch_diagnosis.json` and
  `results/g6_cuvs_g2a_a1_threshold_diagnosis.json`;
- build/install/raw logs under `raw/g6_*` and receipts under `artifacts/g6/`.

`results/g6_gts_g2b_r1_a0_timeout.json` preserves a superseded operational
inference caused by a delayed outer PTY and zero-byte driver log.  The
runner-owned result, runner-owned log, timestamps, and return code prove that
GTS completed normally.  The superseded file must not support any claim.

## Claim policy

Allowed:

- TensorJoin exactly matches the frozen FP64 oracle, unlike the tested GTS and
  cuVS configurations on this workload.
- Current available baselines expose a material distinction between an
  algorithm described as exact/exhaustive and a reproducible numerical
  threshold contract.
- MiSTIC and GDS-Join remain the measured same-contract exact external
  baselines; GTS and cuVS are accuracy-latency context.

Forbidden:

- “TensorJoin is faster than RT-HiSS”; RT-HiSS was not run.
- “TensorJoin is 3.835x faster than cuVS”; cuVS was non-exact and the sampling
  contracts differ.
- “TensorJoin is the first exact GPU similarity join” or “the fastest exact
  join in general.”
- treating a four-pair or fifty-pair disagreement as exact merely because its
  aggregate F1 rounds near one.

## Next action

Request or locate runnable RT-HiSS and COSS artifacts, then port them to the
same input, FP64 oracle, canonical output, denominator, and SM120 host.  Until
then, update the paper with an explicit latest-baseline audit table and the
artifact-availability limitation, without changing the locked thesis or
contribution IDs.
