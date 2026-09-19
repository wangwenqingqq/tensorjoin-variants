# G17 claim-evidence ledger

2026-09-05, gpu-host-8 / RTX PRO 6000 physical GPU 2. No timing denominator is
admitted. All numerical statements concern frozen FP64-reference pair output,
not an exact-real predicate or general correctness judgment about RT-HiSS.

| ID | Exact claim / scope | State | Evidence and quality gate | Counterevidence / allowed wording |
|---|---|---|---|---|
| G17-C1 | Separate default RT-HiSS adapter reconstructs original IDs on the nine frozen D512 inputs. | measured | `decoder_unit.json`; all 16 `case_*.json`; bitwise input permutations, independent raw-mask decoding, counts/padding/capacity | Bounded correctness adapter, not performance, arbitrary inputs, 60K or exact numerical equality. |
| G17-C2 | Native CIFAR4096 output differs from the frozen oracle despite equal 262,144 counts: 2 missing and 2 extra directed IDs. | measured | `case_native_cifar4096_a0.json`; output/oracle hashes; full ID differences; memcheck and synccheck repetitions | Do not call RT-HiSS generally wrong or omit it as a comparator. This is a same-reference admission failure on one real input. |
| G17-C3 | All four real-data disputed decisions are reproduced by native FP32 refinement arithmetic; candidate omission and decoder error do not explain them. | measured | scalar C fmaf replay; same-launch `pair_replay_a0.json`; all-pair candidate coverage | The standalone probe is diagnostic, not a new complete join or a general proof of the only possible numerical failure. |
| G17-C4 | The added zero-endpoint fixture has 2 native extras, 106 versus 104 reference pairs. | measured | `case_none_boundary_zero32_a0.json`, its two safety repetitions, `ADDENDUM_BOUNDARY_ZERO32.md` | Original boundary31 had a coverage gap and remains retained. Do not pretend the original fixture exposed this witness. |
| G17-C5 | Six instrumented adapter runs report zero access/synchronization errors. | measured | three memcheck and three synccheck logs; isolation guards | Exact-output failures remain failures. Leak checking disabled; no Graph, sustained or general safety claim. |
| G17-C6 | Runtime-observed adapter refinement/compression symbols have the same normalized SASS as the native-source control build. | measured | `code_audit_a0.json`, NSYS trace, full SASS, binary/library hashes; 336/408 instructions | Binding is to adapter runtime and native static control. OptiX driver-JIT identity not established; instruction equality is not speed evidence. |
| G17-C7 | A two-sided conservative repair can cheaply make RT-HiSS satisfy our reference contract. | unknown | `NEXT_BASELINE_REPAIR.md`; no repair measurement | Hypothesis only. Preserve native paths and count all repair costs; positive-only reranking is insufficient. |
| G17-C8 | TensorJoin is faster than the modern RT-HiSS baseline under equal output. | unknown | No admitted paired performance evidence | G17 instrumented times must not be used. G16's distinct FP32-first early-screen positive remains separately scoped. |
| G17-C9 | G17 resolves TensorJoin's novelty or submission readiness. | unknown | `PROTOCOL.md` Gate 0; G10/G16 prior-art/readiness audits | No novelty pass, paper revision or submission recommendation. Modern adapter engineering is not the missing non-incremental mechanism. |
