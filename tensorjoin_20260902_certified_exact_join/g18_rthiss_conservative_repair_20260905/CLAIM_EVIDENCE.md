# G18 claim-evidence ledger

Last verified2026-09-05 on gpu-host-8 / RTX PRO6000 GPU3. Scope is the complete
frozen FP64-reference pair contract unless stated otherwise, never exact-real
arithmetic or a public timing denominator. Full provenance is in closure/code
receipts and the evidence manifest. Native RT-HiSS commit remains
`a42fc69cc4b602dc83071b029d185a41e69a04bd`.

| ID | Exact claim | State | Evidence / quality gate | Counterevidence and strongest allowed wording |
|---|---|---|---|---|
| G18-C1 | The two-sided repaired default RT-HiSS operator reproduces all frozen pair IDs on nine D512 inputs. | measured |18 operator records in `campaign_gpu3_a3.json`; independent bitmap/ID oracle, actual permutations, candidate and counter checks | Bounded reference-output admission only; no general RT candidate certificate,60K, other layouts or exact-real claim. |
| G18-C2 | Real4096 needs184/16,777,216 FP64 terminals (96 accept,88 reject), with zero missing/extra IDs. | measured | Five real operator observations reproduce the same work and canonical hash; `closure_checks.json` |0.0010967255% is candidate fraction, not latency. Adversarial zero32 needs62/1024=6.0546875%; do not claim universally cheap correction. |
| G18-C3 | The retained real native FN and FP are both corrected by the numerical mechanism. | measured | Same-launch8192-pair A/B; first four stages2,2,3,3; full real4096 oracle equality | Original G17 failure remains, and no claim is made that RT-HiSS is generally incorrect under its native arithmetic contract. |
| G18-C4 | The error band encloses the stated FP32 prefix / FP64 reference under the admitted arithmetic assumptions. | theoretical | `NUMERICAL_DESIGN.md`; exact rational constants/outward cuts; 515 measured rational prefix checks and 365 cutoff tests | Proof assumptions include finite absolute coordinate values <= 1, D512, RN/gradual underflow and specified reference order; tests alone are not universal proof. |
| G18-C5 | New code passes four bounded memory/synchronization observations and1000 two-buffer predicate repeats. | measured | memcheck/synccheck on zero32 and real4096; `predicate_stress_g3_a3.json`; full record hash stable | Leak checking off, predicate-only stress, no sustained full join/Graph/arbitrary-stream admission. |
| G18-C6 | Runtime-selected repaired refinement changes while compression and upstream CUDA work ownership remain invariant. | measured | `code_audit.json`, frozen diffs and runtime names; refinement992 instructions/40 registers versus336/36, compression408 invariant | Static SASS/resources and counters are mechanism evidence, not speed. OptiX driver-JIT identity not established. |
| G18-C7 | Point-map bytes must be identical to an earlier run for correctness. | rejected | A2 zeros31 failure; two unchanged-native GPU3 controls; actual gathered values, bijection and complete ID output | Retain failed proxy gate. A3 explicitly validates semantic mappings and all original IDs; never waive a false mapping or output mismatch. |
| G18-C8 | Repaired RT-HiSS is a cheap or slow equal-quality performance baseline. | unknown | No admitted public timing; NEXT_COST_CONTRACT.md | Selective terminals do not bound complete cost. Strip diagnostics, resolve engine lifecycle, revalidate and measure all participants together. |
| G18-C9 | TensorJoin now has decisive novelty or modern external superiority. | unknown | G10/G16/G17 prior-art/readiness boundaries; PAPER_IMPACT.md | Comparator normalization is not TensorJoin's contribution. Preserve G16 positive and old negatives; no paper/publication promotion. |

Exceptions are explicit: two prelaunch GPU2 occupancy blocks, one retained
point-map proxy failure, one unadmitted native diagnostic isolation record and
a rejected D18 build. Twenty main GPU3 slots are admitted under the composite
A3 gate; earlier failed receipts remain failed. No timing sample was resampled.
