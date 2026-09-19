# G15 claim-evidence ledger

All measured claims: gpu-host-8 GPU2, 2026-09-05, frozen G15 files plus unchanged
G5/H2B dependencies; exact identifiers and SHA-256 values in each JSON receipt.
Full denominator and limitations: `PLAN_AND_GATE0.md`, `BASELINE_DESIGN.md`,
`DECISION.md`. Old rejected claims are not rewritten.

| ID | Claim / scope | Status | Evidence | Counterevidence / allowed wording |
|---|---|---|---|---|
| G15-C1 | Blockwise CPU prep preserves five arrays and reduces full static-join time | measured | cpu_campaign; public_screen; chunked compatibility; original stage identities | Only frozen CIFAR60K; 1.23–1.27x public early-screen gain, not general/novel |
| G15-C2 | Complete pedantic FP32-first control matches frozen FP64 terminal output | measured | seven fixtures; full compatibility; public output hashes | Reference equivalence, not universal exact-real; finite admitted domain |
| G15-C3 | Runtime-selected control uses FP32 SIMT rather than hidden Tensor Core math | measured | NSYS, three NCU selected functions, precision_audit | Three exact shapes/toolchain only; parent code-object linkage unresolved |
| G15-C4 | FP32 control passes bounded device safety and pointer stress | measured | correctness_gates; full memcheck; fixture synccheck; 1000 N129 calls | Not leak-free shutdown, Graph/arbitrary stream, or sustained60K |
| G15-C5 | CPU-prepared TC path outperforms complete FP32-first | rejected | six-process public_screen; FP32/chunked 0.4141/0.4067 | Strong control is 2.42–2.46x faster; no positive TC-specific claim |
| G15-C6 | GPU preparation can recover a material advantage | unknown | G16 predeclared protocol only at G15 closure | No evidence from G15; G16 must also optimize FP32 preparation |
| G15-C7 | Static TC certificates supply a novel DB/system thesis | unknown | Gate0 overlap ledger; latest primary-source refresh | G15 adds strong engineering evidence, not novelty |
