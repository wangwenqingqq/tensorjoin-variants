# Current TensorJoin Continuation Route

Last verified: 2026-09-04 (G7 artifact admission).

Read `DECISION_G7_ARTIFACT_ADMISSION.md` first for current baseline availability.
G6's "RT-HiSS artifact not found" statement is superseded: both RT-HiSS and COSS
public source were located. RT-HiSS is built and passed one upstream N1000/D2
smoke on physical GPU7 of gpu-host-8. No high-dimensional pair-oracle or new
performance comparison has been run. COSS is source-audited only.

Next: separate RT-HiSS compressed-mask pair export, tiny decoder validation,
then frozen G2A4096x512 oracle. Keep G2B and G3/G4 keepers untouched and do not
wash out the rejected G5 timing result. GPU1 was occupied at final preflight;
recheck all live host state before any new run.

Paper: G6 Overleaf snapshot still has the old artifact-availability text. G7
has not changed/compiled/synchronized the paper. Correct availability at the
next explicit paper revision; exact/performance admission remains pending.

Protocols: `PROTOCOL_G7_RTHISS_ARTIFACT.md`, `PROTOCOL_G7_RTHISS_GPU7_R1.md`.
Evidence: `artifacts/g7_artifact_audit/`, `raw/g7_*`,
`results/g7_rthiss_smoke_a0.json` (GPU1 skip),
`results/g7_rthiss_smoke_a1.json` (GPU7 sample pass).
