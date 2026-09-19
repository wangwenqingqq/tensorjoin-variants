# Targeted paper-readiness audit after G15 and during G16

2026-09-05. This is a source-and-evidence audit, not a full rendered-paper mock
review or a submission recommendation. The manuscript is a frozen historical
artifact; no prose, figures, Overleaf project or remote credential was changed.
Line anchors refer to `overleaf_bundle/main.tex` as hashed in the final archive.

## Strongest asset and strongest material caveat

The project has reproducible full-pair semantics, selected-code/safety evidence,
real public input, and a now-complete independently prepared FP32-first control.
G16 is testing an attributable data-path change without altering original G5
stages or hiding the control. A positive result would be a useful engineering
asset. It would not establish standalone novelty or modern external dominance.

## Verified blockers in the current historical draft

| Priority / source | Verified finding | Required correction/evidence; not a blanket rejection |
|---|---|---|
| P0, main.tex:79–85, 120–126, 416–423 | A per-pair precision decision/three-stage certificate is presented as the key contribution. G10 shows deterministic filter/refine and adaptive-predicate antecedents. | State the specific *non-incremental* mechanism/cost removal beyond the nearest sources. An integrated list of established components, GPU preprocessing, a new label or a conventional tree is insufficient. The current broad novelty claim is not admitted. |
| P0, main.tex:93, 108, 214–219; equation at146 | "Exact FP64" can be read as an exact-real predicate, while the source uses a rounded FP64 terminal and ordinary comparison. G10 has an exactly representable FP32 counterexample. | Specify the reference algorithm, threshold representation and reduction semantics. Use "frozen FP64-reference output" where intended. An exact-rational terminal would be a distinct new implementation gate, not proof supplied by current hashes. |
| P0, main.tex:337–338, 444–446 | The draft says no public RT-HiSS artifact was found/is available. G7 located pinned official code and ran a tiny count smoke. | Replace with actual artifact availability and precise pending same-output adapter/safety/performance state. Do not imply a count smoke establishes full pair equivalence. |
| P0, main.tex:282–285 and headline comparison | G15's complete FP32-first control is missing. It decisively beats the CPU-prepared G5 paths under the same complete denominator. | Preserve G15 as adverse attribution evidence. Any G16 positive must use the also-GPU-prepared control, not revive MiSTIC-only necessity claims. Historical G2B numbers remain historical and must not be silently relabeled G16. |
| P1, main.tex:32, 125, 211, 258, 492 | "Output-sensitive" risks a complexity claim. All upper pairs still undergo dense front-end work; refinement scales with numerical ambiguity, not merely output size. | Give the cost decomposition Q, U8, U32 and output M explicitly. Limit sensitivity wording to uncertainty/refinement/materialization unless an actual complexity result is proved. |
| P1, main.tex:301–310, 391–410 | Old stress and failure evidence belongs to exact old sources/shape sets. G16 introduces new preparation code. | Attach distinct G16 metadata/stress/sanitizer/runtime-code and public-scope evidence; do not inherit a new full60K sustained claim from N129 or metadata-only loops. |
| P1, main.tex:133, 462 | MiSTIC was the strongest admitted external comparator in the historical campaign, not a demonstrated strongest current system. | Admit RT-HiSS/COSS same-output runs and a fair repaired/adaptive FaSTED route, or explicitly keep the modern-comparison gap. An unexecuted baseline is unknown, not slow or incorrect. |

## Evidence admission ladder for the next paper revision

1. Finish G16 fixed correctness/safety/code and two-block same-contract screen.
   Record narrow implementation, mechanism, and thesis decisions separately.
2. Freeze the one-sentence differentiated thesis and contribution IDs against
   G10's primary sources and modern direct prior art. If the new mechanism is
   subsumed, stop that thesis rather than bury the overlap under experiments.
3. Admit modern external output adapters, preserving upstream algorithms,
   original IDs, inclusive predicate, complete materialization, buffer safety
   and full preprocessing/export denominator. Start tiny, then4096 before60K.
4. Only after the cheap mechanism/novelty/comparator gates, freeze a formal
   multi-workload and sustained campaign. Cover density/selectivity, difficult
   near-threshold inputs, scale/footprint and end-to-end tails without selecting
   favorable cells or mixing old and new artifacts.
5. Lock the claim-evidence map, teaser/pipeline/main table, then revise text.
   Back up, compile and render every page plus high-DPI figure crops. Distinguish
   local edit, compile, visual QA and Overleaf synchronization. No upload before
   all intended files and credentials are verified.

This ladder does not require another per-step user approval. It is a scientific
admission boundary: evidence can admit the next bounded step, not publication
certainty. Missing evidence is uncertainty; failed fixed tests stay failed.
