# G8 A0 claim-evidence ledger

Experiment ID: `tensorjoin_20260904_g8_refinement_layout_g2a4096`.
Last verified: 2026-09-04, gpu-host-8, physical GPU1 / SM120.
Source and binary hashes are in the summary and immutable evidence manifest.

| Claim ID | Proposed claim | Scope / denominator | State | Evidence / oracle | Counterevidence / caveat | Allowed wording |
|---|---|---|---|---|---|---|
| G8-C1 | Ambiguity count alone cannot always select a refinement layout | Synthetic 102079 fixed logical pairs, two physical placements; resident-GPU component | partial | Both ordered timing processes, identical three-way states and CPU FP64 definitive decisions | Not real uncertainty; no online organization charged | A synthetic fixed-work control exhibits a layout-dependent crossover |
| G8-C2 | Actual uncertainty benefits from adaptive sparse/tiled routing | G2A CIFAR-GIST uncertainty, four fixed layouts; same component | rejected | Real-data gate false in both processes and sustained diagnostics | All actual layouts favor P16; no certified TC path tested | This four-layout direct-FP32 experiment does not support an actual-data routing crossover |
| G8-C3 | P16 improves the complete current pipeline | Complete join or original compacting G3C stage | unknown | No evidence at that denominator | Only the aligned-output P1 control was timed; first pass, compaction, repair, export excluded | P16 is a promising generic batching optimization requiring a new integration campaign |
| G8-C4 | A0 definitive decisions are consistent with the oracle | All six fixed workloads; output states only | measured | All definitive states agree with CPU FP64, all method/layout hashes match, guard bytes pass | 97 real residuals unresolved; no all-input theorem claimed | Three-way component decisions match on the tested workload |
| G8-C5 | Certified TC block refinement is novel and fast | Future certified Tensor Core path | unknown | No A0 TC instructions or timing; protocol cites nearest prior art | Existing adaptive predicates and sparse blocking preclude broad novelty claims | Not established |
| G8-C6 | Tile padding explains a particular hardware bottleneck | Direct-FP32 T4x4, real layouts | inferred | Time and active tile count covary; padding ratios 9.43–13.78 | No NCU dynamic memory/compute attribution | Results are consistent with padded work dominating this tiled implementation |

Primary evidence: `results/g8_layout_summary_a0.json`; decision and full limits:
`DECISION_G8_REFINEMENT_LAYOUT_A0.md`. No G8 claim has been promoted to paper prose.
