# Current TensorJoin Continuation Route

Last verified: 2026-09-04 (G10 certificate novelty and semantic-contract audit).

## Current research decision (G10)

Read `DECISION_G10_CERTIFICATE_NOVELTY.md` and
`SOURCES_G10_CERTIFICATE_AUDIT.md` first. The existing certificate is retained
as useful infrastructure, but generic quantized filtering, adaptive precision,
error bounds and TC distance self-join have clear prior art. No standalone
novelty pass or non-incremental integrated mechanism has been established.
The audit does not claim that a single prior paper duplicates the whole system.

G10's four-case CPU illustration separates equality to a frozen FP64 oracle
from an exact-real predicate on stored FP32 coordinates. An exact distance
1+2^-54 can round to 1 and cross the inclusive T=1 decision; an enclosing
interval correctly leaves it unresolved. This is not a new GPU failure or a
public-data error rate. G9 already preserves uncertainty and rejects unresolved
main-workload admission; a general terminating exact fallback is not present.

Do not revive the paper merely by adding breadth, a conventional tree, Graph
replay, proof detail or new naming. Identify a non-incremental work-removal
mechanism and its novelty kill test first. A full adaptive-FP32-first scan with
the same terminal/output contract is a missing attribution control, not a
measured win or loss. No new GPU campaign is admitted by this audit.

G2B's scoped 5.083x external result, G3/G4 attribution, G5's failed unification,
G7's pending modern-baseline admission and G9's narrow captured win all remain.
The historical paper-spine recommendation is not a current novelty approval.
No kernel, manuscript or Overleaf edit was made. Evidence:
`results/g10_exactness_contract_cpu_a0.json`,
`artifacts/g10_certificate_audit/evidence_manifest.sha256`.

## Latest GPU experiment (G9)

Read `DECISION_G9_TC_REFINEMENT.md` for the latest GPU experiment. On the fixed
102079 actual uncertain pairs, both TC schedules lose under eager submission.
With the same kernels and prebuilt queues under CUDA Graph replay, diagonal
TC achieves a small verified 1.125–1.138x component speedup across real layouts
(real identity: P16 68.003/67.826 us versus TC 60.339/59.786 us in two processes).
No actual case passes the frozen >=1.15x gate in both processes. Physical TC
tiles still lose on real data; the synthetic clustered 2.998x gain is only a
positive control. Preserve both the eager negative and the captured positive.

Close these prototypes at the current scope. Do not lower the gate, add tuning,
or expand into the proposed online router/paper story. Encoding, packing,
intermediate queue construction/count discovery, capture and final pair export
are excluded: full-cost performance remains UNKNOWN, not a measured negative.
Retain the checked numerical component; novelty remains unestablished. Reopen
only for a separately justified mechanism/workload with a fresh novelty audit.

Exact requested integer dots, directed numerical bounds, boundary fixtures,
final main predicates, guard/reallocation checks, memcheck and synccheck pass.
Both execution modes have two fresh timing processes and sustained diagnostics;
the same 18 cubins are retained. Installed CUDA tools are 13.1.115, but the
actual Triton Blackwell assembler is 12.9.86; see the recorded binary hash.
This is not a complete online self-join or a production all-input certification.

Evidence: `results/g9_summary_a0.json`, `results/g9_graph_summary_a0.json`,
`CLAIM_EVIDENCE_G9.md`, `artifacts/g9_evidence/evidence_manifest.sha256`.
At the recorded final postflight (2026-09-04 11:17:52 UTC), GPU7 had no compute
process, 14 MiB used and 0% utilization. Foreign GPU0/GPU1 work was untouched.
Recheck live state before any future launch. No manuscript or Overleaf changes.

## Previous layout diagnostic (G8, unchanged)

Read `DECISION_G8_REFINEMENT_LAYOUT_A0.md` for the preceding experiment. On the
fixed G2A uncertainty set, all four layouts favor the P16 sparse batching
control: ordinary-FP32 T4x4 is 5.93–8.66x slower. A synthetic fixed-pair clique
control does cross over (T4x4 1.289x faster when clustered, 7.63x slower when
scattered), but this does not satisfy the real-data gate. Two independent
processes and sustained diagnostics agree; state hashes, CPU FP64 definitive
decisions, memcheck and synccheck pass. G8 itself tested no Tensor Core path;
G9 above is the separate certified-TC follow-up.

Do not implement the proposed online layout router or promote this story into
the manuscript from A0. G9 tested a distinct certified-TC mechanism under a new
numerical contract, but did not pass the real-data opportunity gate. P16 is an
engineering signal only; original compacting G3C integration is unmeasured.
Evidence: `results/g8_layout_summary_a0.json`, `CLAIM_EVIDENCE_G8_A0.md`,
`artifacts/g8_evidence/evidence_manifest.sha256`. G8's final GPU1 idle state
was historical; it became occupied during G9. Always recheck before a new run.

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
