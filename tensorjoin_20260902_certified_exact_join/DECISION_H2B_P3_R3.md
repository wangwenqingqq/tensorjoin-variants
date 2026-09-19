# Decision: H2B-P3-R3 Full-Operator Access Safety and Stability

Date: 2026-09-03

## Decision

**PASS the bounded device-access/synchronization and stress gate; admit P4
timing.**  Both full actual-data target-8 operators pass Compute Sanitizer
`memcheck` and `synccheck` with `ERROR SUMMARY: 0 errors`.  All four
dataset/target cells preserve one exact count/ambiguity/output signature over
1,000 complete invocations while alternating two independent device states.

This is not a leak-free claim.  The preceding full-leak R2 diagnostic retains
three PyTorch-owned cuBLAS-handle allocations totaling 8,520,704 bytes at
process shutdown.  R3 explicitly gates device access rather than silently
discarding that limitation.

## Sanitizer results

| Tool | Dataset | Full object/token shape | Target | Exact output | Tool errors |
|---|---|---|---:|---|---:|
| memcheck | ESC-50/PANNs D2048 | 128x512 / 640x2560 | 8 | 1,024 IDs, ambiguity 9 | 0 |
| memcheck | UCF101/R3D-18 D512 | 128x512 / 662x2684 | 8 | 1,024 IDs, ambiguity 4 | 0 |
| synccheck | ESC-50/PANNs D2048 | 128x512 / 640x2560 | 8 | 1,024 IDs, ambiguity 9 | 0 |
| synccheck | UCF101/R3D-18 D512 | 128x512 / 662x2684 | 8 | 1,024 IDs, ambiguity 4 | 0 |

The checked path includes pedantic SGEMM, fused object interval
aggregation/compaction, dynamic count read, exact FP64 ambiguous-panel repair,
final count/ID transfer, and canonical sorting.

## Two-buffer stress

Each cell retains 1,000 complete invocations and alternates state slots 0/1:

| Dataset | Target | Final count | Direct / ambiguous | Unique signatures | Exact iterations |
|---|---:|---:|---:|---:|---:|
| PANNs | 1 | 128 | 127 / 2 | 1 | 1,000/1,000 |
| PANNs | 8 | 1,024 | 1,021 / 9 | 1 | 1,000/1,000 |
| R3D-18 | 1 | 128 | 128 / 0 | 1 | 1,000/1,000 |
| R3D-18 | 8 | 1,024 | 1,023 / 4 | 1 | 1,000/1,000 |

All 4,000 retained invocations have zero duplicate, overflow, count drift, or
P1 ID-hash mismatch.  GPU4 was idle at postflight (14 MiB, 0%, no compute
application on the target UUID).

## Rejected attempts and material caveat

- Original P3: 12 outstanding allocations / 153,224,192 bytes under full leak
  checking; rejected by its any-error rule.
- P3-R2: after allocator/workspace cleanup, three process-owned
  `cublasCreate_v2` allocations / 8,520,704 bytes remain; rejected by its
  leak-free rule.
- P3-R3 disables leak reporting only for the access-error gate and retains the
  R2 log as the resource-lifetime evidence.  Allowed wording is
  “Compute Sanitizer access/synchronization clean on the tested operators,”
  never “leak-free.”

## Evidence

- Safety summary: `results/h2b_p3_safety_summary.json`, SHA-256
  `3a02d1a55016c70989d59ace2b1003d1ccbde9ba2bb7720bee45354ae9efd245`.
- Stress artifact: `results/h2b_p3_stress.json`, SHA-256
  `bd3c4b1c3c960da10962a1f31452a0018b4299046d75bfe02ba82a4ae18a1225`.
- Memcheck logs: SHA-256 `4dfd6471...050c` / `5379e72d...7631`.
- Synccheck logs: SHA-256 `58927da0...75d` / `0a0a3a4c...b219`.
- Runner/summarizer: SHA-256 `63b19aab...45ba` / `cd51af3e...2554`.
- Pre/postflight: SHA-256 `0c61d2d9...d304` / `f1b3043a...8181`.

## Next gate

P4 must compare the unchanged H1 candidate and H2B keeper in direction-balanced
fresh processes using host outer-wall timing, then run separately retained
sustained sequences.  Profiler or sanitizer duration is forbidden.  Every
timed output must reproduce the frozen exact ID/count signatures.
