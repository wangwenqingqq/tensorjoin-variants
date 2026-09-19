# G10 decision: retain the certificate; no standalone novelty pass

Date: 2026-09-04. Scope: existing-source audit, primary literature, and one
four-case local CPU semantic illustration. No GPU work or paper changes.

**The certificate is useful infrastructure, but the present evidence does not
justify promoting it into an independent database/systems contribution.**
The strongest problem is not G9's 1.13x result: approximation bounds,
uncertainty-driven precision escalation and candidate refinement already have
clear antecedents. A non-incremental interaction or a new work-elimination
mechanism has not been established by the current artifacts.

This rejects broad standalone novelty wording. It does NOT establish that one
paper contains the whole implementation, that a certified GPU join can never
be valuable, or that earlier positive measurements disappear.

## 1. What the implementation actually does

For a reconstruction z_i of original vector x_i, store an upper bound
e_i >= ||x_i-z_i||. For two rows, triangle inequalities give

```
max(0, ||z_i-z_j|| - e_i - e_j) <= ||x_i-x_j||
                              <= ||z_i-z_j|| + e_i + e_j.
```

G3B surrounds the reconstructed distance with an additional arithmetic-error
envelope. It accepts or rejects when the bound separates the threshold, and
compacts unresolved IDs. G3C adds an FP32 direct-distance interval before the
old FP64 terminal. G9 changes the reconstructed arithmetic to exact INT8-limb
dot products and uses an interval-guarded FP64 terminal in its restricted domain.

Live source anchors, all preserved unchanged:

- `src/run_g3b_r1_gpu_analytic_certificate.py:44–164`: integer dot, residual
  envelope, threshold decisions and compaction; `:203–234`: residual preparation.
- `src/run_g3c_b_r1_gpu_cascade.py:58–129`: middle precision filter and routing.
- `src/run_g2a_tensorjoin.py:194–232`: old terminal uses a rounded FP64 sum and
  an ordinary threshold comparison.
- `src/g9_tc_kernels.py:118–135`: G9 terminal retains an unresolved state.
- `src/run_g9_tc_refinement.py:200–207`: unresolved main workloads fail
  admission; fixture uncertainty is intentionally allowed.

The current mechanism escalates work according to **numerical ambiguity**, not
just the final output count. Let Q=N(N+1)/2, U8 and U32 be the unresolved counts
after the first and second filters, and M the output size. An attribution model
is preprocessing + dense TC work over Q + FP32 work over U8 + FP64 work over
U32 + queues/synchronizations + materialization of M. It is not an O(U32)
whole-join algorithm: the dense Q-by-D work still occurs. “Output-sensitive”
must not be used to imply an unproved output-sensitive complexity theorem.

## 2. Claim-by-claim novelty test

| Candidate claim | Closest established mechanism | Actual remaining difference | Decision |
|---|---|---|---|
| Quantize, use distance bounds, refine candidates | [VA-file, Section 4](https://www.vldb.org/conf/1998/p194.pdf) | GPU all-pairs execution and particular scalar encoding | Generic claim rejected; not a new filtering principle |
| Increase precision only for difficult decisions | [Adaptive robust predicates](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf), [general polynomial filters](https://link.springer.com/article/10.1007/s10543-023-00975-x) | TC-first batched high-dimensional execution | Generic insight rejected; a distinctive batch interaction is not yet shown |
| Derive deterministic mixed-precision error bounds | [QDOT](https://epubs.siam.org/doi/10.1137/21M1406994) | Threshold predicates and complete pair materialization rather than approximate dot tolerance | Contract distinction is real, but a standalone mechanism claim remains unproven |
| Use reconstructed vectors and residual triangle bounds | [TRIM, Sections 3.1–3.2](https://arxiv.org/html/2508.17828v1) | Both rows quantized, deterministic interval and GPU arithmetic envelope | Generic geometry not new; strict and relaxed bounds must not be conflated |
| TC distance self-join | [FaSTED](https://arxiv.org/html/2508.21230v1) | Sound filters and repair, rather than accepting low-precision threshold errors | A meaningful engineering difference, not by itself decisive novelty |
| Certify retrieval completeness | [Certified Cosine](https://arxiv.org/html/1910.02478v2) | Self-join output versus nearest-neighbor certificates | General certification claim rejected; numerical and search certificates are different |
| Use integer units for higher-precision products | [Integer DGEMM](https://arxiv.org/abs/2306.11975), [Ozaki Scheme II](https://arxiv.org/abs/2504.08009) | G9's specific admitted limb representation and certificate | Retain as implementation; not a new decomposition family |
| Three-stage pipeline plus output buffers constitutes a new system | Established ingredients above; modern batching/refinement in [RT-HiSS](https://arxiv.org/abs/2609.01975) | A particular integrated implementation | Unknown as a full-system thesis; no demonstrated non-incremental interaction or complete same-artifact evidence |

These are scoped comparisons, not a claim that NN, ANN, geometric predicates
and exact self-join have identical contracts. In particular, RaBitQ's
high-probability bound does NOT subsume a deterministic certificate, nor does
an ANN traversal prove exhaustive join completeness. Those distinctions remain
important, but they cannot erase the deterministic antecedents above.

## 3. Three meanings of exactness must be separated

| Guarantee | Meaning | Current evidence |
|---|---|---|
| Search completeness | No qualifying pair is omitted by candidate generation, relative to a defined predicate | Full-output oracle/structure checks on frozen workloads; not supplied by reranking ANN candidates alone |
| Reference equivalence | Output agrees with a specific FP64 computation and its threshold semantics | G2B/G3/G4 frozen-oracle equality; not arbitrary reduction-order or all-input equivalence |
| Exact-real predicate | Evaluate sum((x_i-y_i)^2) <= T exactly, treating stored FP32 coordinates and T as exact rationals | G9 has sound interval decisions and rejects unresolved main-workload admission; there is no general terminating exact fallback |

The distinction is reproducible with only two nonzero coordinates:

```
x = (1, 2^-27, 0, ..., 0), y = 0, D = 512, T = 1.
Exact squared distance = 1 + 2^-54 > 1: reject.
Rounded binary64 squared distance = 1: ordinary FP64 comparison accepts.
G9-style final interval crosses 1: unresolved, not a false acceptance.
```

All coordinates are exactly representable in FP32. The local same-process
CPU illustration also checks clearly inside, equality and clearly outside
controls, with exact rational interval enclosure. All four checks pass.

**Scope:** this is a measured CPU arithmetic illustration plus source-based
interpretation, NOT a newly executed GPU failure or an observed error rate in
the public datasets. Historical FP64-oracle comparisons remain intact. Adding
an exact accumulator or expansion terminal would close a semantic gap; by
itself that is established robust-arithmetic practice, not a fresh thesis.

Evidence: `PROTOCOL_G10_CERTIFICATE_AUDIT.md`,
`src/check_g10_exactness_contract_cpu.py`,
`results/g10_exactness_contract_cpu_a0.json`,
`raw/g10_exactness_contract_cpu_a0.log`.

## 4. Positive assets are not erased

| Asset | Verified historical result | Material limit |
|---|---|---|
| G2B external comparison | 5.083x over MiSTIC, eight paired rounds, one CIFAR-GIST-512 60K workload | Empirical FP64-oracle agreement; not the same artifact as the later proof-bounded router |
| G3C certified middle stage | Only 97 of 102079 first-stage uncertain pairs reach FP64 on the frozen subset | Specific input/toolchain domain; not a new principle or whole-system result |
| G4C router attribution | 1.271x–2.329x on three N4096 public anchors | Versus the internal TC-plus-direct-FP64 path, not a strong standalone adaptive-FP32-first scan |
| G5 unification | Correctness/code/safety pass; external ratios 4.092x and 0.951x | The frozen performance gate failed; the full-denominator variance is not localized |
| G9 captured refinement | 1.125x–1.138x across actual layouts | Optimistic prebuilt-queue component, below its 1.15x gate; eager loses |
| G7 baseline artifacts | RT-HiSS tiny count smoke; COSS source audit | New same-contract pair-output and performance admission remains pending |

The quoted result JSONs and inspected sources are hashed in the audit evidence
manifest. No public-data result was rerun, resampled or reclassified by G10.

## 5. Decision and next-action boundary

**Implementation:** retain the code, proofs, fixtures, adapters and raw evidence.
**Standalone mechanism:** do not claim new error-bounded filtering, generic
adaptive precision, or a universally exact GPU operator from this module.
**Current paper thesis:** no novelty pass; do not promote the existing bundle
as sufficient merely by adding datasets, a conventional tree, Graph replay,
proof details or a renamed “query planner.” Existing paper files are untouched.

The highest-value missing attribution is a strong **non-TC-first adaptive
distance-predicate control** with the same terminal semantics and complete
output. G4's TC-plus-FP64 keeper and G9's already-uncertain P16 component do not
answer whether a full scan with an ordinary FP32 filter and exact/defined
terminal already captures the benefit. Such a control is not measured here;
neither a win nor a loss is inferred. A future repaired-FaSTED control likewise
must certify rejected pairs, not just rerank the approximate hits.

This missing control is a necessary research check, not automatic permission
to restart GPU work. First identify a candidate mechanism with all of:

1. An explicit task and exactness contract, including equality and termination.
2. A specific work term removed beyond existing bounds and adaptive filters.
3. A reason the interaction is non-incremental, supported by closest prior art.
4. A cheap falsification test and same-contract external/adapted comparator.

No current artifact fills all four. Return to that problem/mechanism selection
stage; preserve the certificate as a reusable component rather than discard it
or force it into the lead contribution. A positive future mechanism may reuse
this work, but that possibility is not current evidence of publishability.

## Evidence classification

| ID | Claim | State | Boundary |
|---|---|---|---|
| G10-C1 | Generic certificate/precision-escalation novelty is not supportable | inferred | Primary-source overlap; no claim of a literal end-to-end duplicate |
| G10-C2 | FP64 threshold equivalence need not imply exact-real equivalence | partial | Exact rational argument and four measured CPU cases; no GPU execution or prevalence claim |
| G10-C3 | Existing positive and negative experimental artifacts remain | measured | Hash verification of the inspected files; no fresh timing |
| G10-C4 | The existing integrated system has a non-incremental contribution | unknown | No identified work-removal interaction with decisive same-contract evidence |
| G10-C5 | Full adaptive-FP32-first comparison favors either design | unknown | No experiment performed |

Detailed primary-source routes and retrieval limits:
`SOURCES_G10_CERTIFICATE_AUDIT.md`. Evidence:
`artifacts/g10_certificate_audit/evidence_manifest.sha256`.
