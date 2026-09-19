# G12 Claim-Evidence Ledger

Experiment: `tensorjoin_20260904_g12_streamed_factorized_semantics_cpu_a0`.
Verified: 2026-09-04. Host/software/source provenance is in the raw JSON.
Contract: `PROTOCOL_G12_FACTORIZED_BASELINE.md`.

| ID | Claim | Status | Evidence and denominator | Boundary / allowed wording |
|---|---|---|---|---|
| C1 | A streamed factorized plan can avoid global pre-predicate candidate materialization | theoretical | Elementary set identity and live-set derivation in DESIGN_G12 | Fixed endpoint-only deterministic predicate and existential/set output; not a GPU benchmark |
| C2 | CPU reference matches the exact direct-distance oracle | partial | 96/96 synthetic integer runs; all thresholds, tile sizes and orders retained | Exact on those fixtures, not arbitrary FP32 or real-data vector admission |
| C3 | Score panel size stays <=t^2 without a global eligible-pair buffer inside the strong-plan function | partial | Logical counters plus source inspection; maximum 256 slots | Not process RSS: the independent oracle in the same process materializes eligibility |
| C4 | Candidate materialization is a uniquely removable cost relative to this strong baseline | rejected | C1 and C2 | That buffer is already absent; do not credit its removal to our proposal |
| C5 | Factor means/sums and per-vector norms determine exact pair matches | rejected | Two-dimensional summary-collision negative control | Only the stated lossy-summary shortcut is refuted |
| C6 | FFX is an admitted exact GPU vector-join baseline | unvalidated | Source-only audit; no such execution measured | FFX is nearest prior art for factorized pipelined consumers, not a reported exact GPU comparator |
| C7 | A new mechanism beats a complete strongest baseline end to end | unknown | No evidence | No GPU speedup, performance rejection, or universal impossibility claim |

Keeper: independent materializing integer predicate oracle (correctness only).
Candidate under examination: standard streamed factor scan plus output-only
DISTINCT (a comparator, not claimed novelty). Data: four synthetic fixtures;
no real-vector performance data. Graph/CUDA-event/NCU/SASS/sanitizer/stress/
clock/algorithm-engine fields: N/A. No commit applies to the local project;
external source revisions and individual file hashes are pinned separately.
