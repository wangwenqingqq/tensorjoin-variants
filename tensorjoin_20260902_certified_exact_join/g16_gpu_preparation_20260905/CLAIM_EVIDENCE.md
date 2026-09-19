# G16 claim-evidence ledger

Provenance for measured rows:2026-09-05, gpu-host-8 physical GPU2, exact source,
input, selected-code and library hashes in `artifacts/frozen_screen_sources.json`
and the linked result receipts. Full scope/denominator: `PROTOCOL.md`,
`ADDENDUM_FAIR_GPU_NORMS.md`, `DECISION.md`. No historic claim is rewritten.

| ID | Exact claim / scope | Status | Evidence | Counterevidence / strongest allowed wording |
|---|---|---|---|---|
| G16-C1 | GPU construction preserves original real60K metadata and bounds actual reconstruction on fixtures | measured | metadata_a0; frozen sources; seven cases | Tiny subnormal codes/scales differ; byte identity applies to real60K, not every input |
| G16-C2 | FP32 control receives conservative GPU norms and keeps reference join output | measured | norm_metadata_a0; fp32_gpu_validation_a1; full compatibility | Finite tested domain and frozen FP64 reference, not universal exact-real |
| G16-C3 | Changed kernels pass bounded safety and pointer stress | measured | safety_gates_r1; full memcheck; synccheck logs; two1000-call tests | Metadata and N129 stress, not1000 full60K; no leak-free/Graph/arbitrary-stream claim |
| G16-C4 | Original downstream work/code and actual SIMT control dispatch are preserved | measured | precision_audit; NSYS; three NCU exports; closure_checks | Exact three shapes/toolchain; parent library code-object linkage unresolved; profiler time not performance |
| G16-C5 | GPU-prepared TC path survives the equally optimized FP32 control on complete host-to-host60K screen | measured | six fixed slots in public_screen; closure_checks | 1.499724/1.267190 same-block ratios; reverse sample slower; only narrow early screen, no stable1.5x/formal/tail claim |
| G16-C6 | Candidate improves total memory footprint | rejected | public allocator peaks593,544,704 vs469,237,760 bytes | Candidate uses26.5% more PyTorch peak allocation; total device/library footprint not measured |
| G16-C7 | Data-path change removes a measured implementation bottleneck | inferred | G15 CPU loss plus G16 fixed-stage/code/work and equally optimized comparison | Supports data-path attribution, not a new algorithm, asymptotic pruning or universal TC superiority |
| G16-C8 | Current proposal is non-incrementally novel / superior to newest external systems | unknown | G10 prior-art ledger; G7 adapter admission; PAPER_READINESS | No novelty pass or modern external-dominance wording |
| G16-C9 | Candidate generalizes to other data/shapes and sustained workloads | unknown | no new evidence | Do not compose old G4/G5 subset or stress results into a new all-shape claim |
