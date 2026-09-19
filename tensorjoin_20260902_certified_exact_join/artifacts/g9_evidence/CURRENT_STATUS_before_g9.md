# Current TensorJoin Continuation Route

Last verified: 2026-09-04 (G8 A0 refinement-layout diagnostic).

Read `DECISION_G8_REFINEMENT_LAYOUT_A0.md` for the latest experiment. On the
fixed G2A uncertainty set, all four layouts favor the P16 sparse batching
control: ordinary-FP32 T4x4 is 5.93–8.66x slower. A synthetic fixed-pair clique
control does cross over (T4x4 1.289x faster when clustered, 7.63x slower when
scattered), but this does not satisfy the real-data gate. Two independent
processes and sustained diagnostics agree; state hashes, CPU FP64 definitive
decisions, memcheck and synccheck pass. No Tensor Core block path was tested.

Do not implement the proposed online layout router or promote this story into
the manuscript from A0. A distinct certified-TC refinement mechanism would
need a new numerical contract and real-data opportunity gate. P16 is an
engineering signal only; original compacting G3C integration is unmeasured.
Evidence: `results/g8_layout_summary_a0.json`, `CLAIM_EVIDENCE_G8_A0.md`,
`artifacts/g8_evidence/evidence_manifest.sha256`. Final GPU1 was idle; recheck
before every new run. GPU0's foreign training process was left untouched.

## Baseline admission (G7, still pending)

Read `DECISION_G7_ARTIFACT_ADMISSION.md` first for current baseline availability.
G6's "RT-HiSS artifact not found" statement is superseded: both RT-HiSS and COSS
public source were located. RT-HiSS is built and passed one upstream N1000/D2
smoke on physical GPU7 of gpu-host-8. No high-dimensional pair-oracle or new
performance comparison has been run. COSS is source-audited only.

Pending baseline work: separate RT-HiSS compressed-mask pair export, tiny decoder validation,
then frozen G2A4096x512 oracle. Keep G2B and G3/G4 keepers untouched and do not
wash out the rejected G5 timing result. The earlier G7 GPU1 occupation was a
historical observation, not current live state; always recheck before running.

Paper: G6 Overleaf snapshot still has the old artifact-availability text. G7
has not changed/compiled/synchronized the paper. Correct availability at the
next explicit paper revision; exact/performance admission remains pending.

Protocols: `PROTOCOL_G7_RTHISS_ARTIFACT.md`, `PROTOCOL_G7_RTHISS_GPU7_R1.md`.
Evidence: `artifacts/g7_artifact_audit/`, `raw/g7_*`,
`results/g7_rthiss_smoke_a0.json` (GPU1 skip),
`results/g7_rthiss_smoke_a1.json` (GPU7 sample pass).
