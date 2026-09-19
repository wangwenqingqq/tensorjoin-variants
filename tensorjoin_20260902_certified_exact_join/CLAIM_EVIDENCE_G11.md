# G11 Claim-Evidence Ledger

Verified 2026-09-04 on local macOS arm64, Python 3.9.6, NumPy 2.0.2.
Experiment: `tensorjoin_20260904_g11_relational_eligibility_cpu_a0`.
Contract: `PROTOCOL_G11_RELATIONAL_CENSUS.md`.

| ID | Claim | Status | Evidence / denominator | Counterevidence and allowed wording |
|---|---|---|---|---|
| G11-C1 | Neither predicate meets all three structural thresholds in both seeds | measured | Four cells in A0 JSON, all 16384 rows independently verified | Reject this workload premise, not all graph/vector mechanisms |
| G11-C2 | Common-reference witness duplication is about 1.36 occurrences per unique pair | measured | W/E for two sampled 4096x16384 batches on real citations | This is not latency; deduplicated CSR already avoids repeated distance work |
| G11-C3 | Common-reference factor payload is about 8.6–8.9x smaller than unique CSR | measured | int32 IDs plus 24-byte factor metadata versus int32 CSR columns and int64 offsets | Representation-only; shared graph, construction and GPU allocation are not included |
| G11-C4 | Common-reference fat factors cover about 93% of eligible pairs | measured | Union coverage, not sum of overlapping witness sizes | No duplicate-free panel algorithm implemented; not a novelty pass |
| G11-C5 | Literal 16x16 common-citer panels require 157–163 slots per unique pair | measured | Integer physical pair-slot ledger with padding and witness repetition | Do not convert slot ratios into GPU performance claims |
| G11-C6 | New operator beats strongest complete same-contract baseline | unknown | No evidence: no vectors or GPU evaluated | No speedup claim allowed |
| G11-C7 | A differentiated factor-preserving operator exists beyond nearest prior art | unknown | No evidence; source ledger identifies overlapping mechanisms | State a research question, not an accepted contribution |
| G11-C8 | Generic filtered GPU vector search or interval tile skipping is new | rejected | Primary sources in the G11 source ledger | Different query consumer alone is insufficient |

Raw evidence: `raw/g11_relational_census_cpu_a0.log`,
`results/g11_relational_census_cpu_a0.json`. Exact software/source/data hashes
are in the result metadata. The archive CRC and source/protocol hashes were
checked again after the run. No git commit applies: this workspace is not
managed as a git repository. The remote mirror is storage, not a remote rerun.
