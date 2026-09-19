# Decision G4C: cross-dataset dynamic-router timing admitted

Date: 2026-09-03

## Decision

**Promote cross-dataset, shape-local dynamic-router attribution.** On the three
predeclared public `N=4096,k=64` anchors, the exact three-stage path beats the
same ragged-safe G3B plus direct-FP64 keeper in every one of 24 formal processes.
All three per-dataset confidence and sustained gates pass.

This closes the most important G4A limitation: the dynamic-count speedup is no
longer restricted to one CIFAR subset, dimension, or sample. It remains a
resident-GPU operator result rather than ingest-inclusive or full public-system
end-to-end evidence.

## Formal result

| Dataset | Marginal keeper median | Marginal candidate median | Paired process-median speedup | 95% process bootstrap | Sustained speedup | Wins |
|---|---:|---:|---:|---:|---:|---:|
| SIFT-128 | 264.449 us | 207.959 us | 1.270875x | [1.267100x, 1.274316x] | 1.269737x | 8/8 + 8/8 |
| CIFAR-GIST-512 | 826.680 us | 355.435 us | 2.328925x | [2.299469x, 2.358065x] | 2.340438x | 8/8 + 8/8 |
| Fashion-MNIST-784 | 487.019 us | 287.777 us | 1.709810x | [1.687169x, 1.735672x] | 1.708095x | 8/8 + 8/8 |

The first win count is for process medians; the second is for 1,000-call
sustained ratios. Every confidence lower bound and sustained geometric mean is
above the frozen 1.10x threshold.

## Correctness, dynamic work, and binary evidence

- Formal slots: 24/24 clean and admitted; no foreign or postflight GPU process.
- Each process validates exact upper-ID equality before and after timing.
- Dynamic counts are invariant in all 4,800 retained calls per variant and all
  24,000 sustained calls per variant.
- Keeper/candidate G3B ambiguity counts are 39,810 / 109,499 / 27,349 for
  SIFT/CIFAR/Fashion; the candidate reduces FP64 work to 83 / 117 / 63 pairs.
- Every fresh formal cache reproduces the exact per-dimension G3B, G3C, and
  FP64 cubins audited after the screen.
- Static audit confirms `sm_120a`, INT8 MMA only in G3B, FP64 arithmetic only in
  fallback, and zero stack/local bytes or static LDL/STL for every selected
  kernel.

## Evidence and claim boundary

- Formal summary: `results/g4c_formal_summary.json`, SHA-256
  `43ed0a578636a47bceff009b14570b8898df5fdfea200c4a853c5db93a309cf3`.
- Formal runtime audit: `results/g4c_formal_runtime_audit.json`, SHA-256
  `b1714742c962222443da1d9a7852ac9642040772b369632dfd19334ef0f1a106`.
- Formal protocol SHA-256:
  `211fee8ecd0d43e8c1143b709719438dab2450469fe78321bdffef9d3726e4e6`.
- Screen summary/runtime audit are retained separately and are not promoted as
  performance claims.

Allowed: on the three tested public 4096-point, target-64 anchors, charging
actual dynamic count reads, synchronizations, and host-dispatched stage extents,
the exact three-stage router is 1.271x--2.329x faster than refining all G3B
ambiguity in FP64.

Not allowed: full 60K breadth, ingest-inclusive/output-materializing end-to-end
speedup, arbitrary density/scale speedup, a dimension-only causal claim, or a
universal all-dataset claim.

## Next highest-value gate

The breadth matrix and timing anchors now support the middle strategy. The next
paper-critical question is not another modality or density port; it is whether
a certificate-aware index can remove dense G3B tile work without repeating the
already rejected conventional pivot-tree ceiling. That direction requires a
fresh novelty kill test and a structural pruning oracle before GPU work.
