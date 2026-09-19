# A0 Decision

Date: 2026-09-02

## Conclusion

**PASS to B0.** The per-vector INT8 triangle certificate is geometrically tight
enough on the frozen ESC-50 workload to justify a GPU implementation screen.
This is not yet a performance result or a paper thesis acceptance.

## Strongest evidence

| Target results/query | Exact pairs | Ambiguous pairs | Ambiguous fraction | Final mismatch |
|---:|---:|---:|---:|---:|
| 1 | 512 | 1,605 | 0.0765% | 0 |
| 8 | 4,096 | 7,300 | 0.3481% | 0 |
| 64 | 32,768 | 26,185 | 1.2486% | 0 |

- Maximum ambiguity: 1.2486% <= 5%.
- Median ambiguity: 0.3481% <= 1%.
- Lower/upper containment violations: 0/0 at every radius.
- False accept/reject and final classification mismatch: 0 at every radius.
- Worst-case INT32 accumulation: 16,516,096 < 2,147,483,647.
- Hot-code plus metadata footprint: 25.293% of FP32 source bytes.

## Material caveat

The exact method must retain or fetch the original vectors for refinement, so
total retained storage in this prototype is 125.293% of FP32 rather than a
compression win. Also, 1,356 of 18,000 possible windows were exact-zero after
centering and were deterministically excluded before sampling. The result is a
numerical mechanism test on handcrafted audio features, not a semantic audio
retrieval result.

## Evidence

- `results/a0_summary.json`
- `results/a0_thresholds.csv`
- `raw/a0.log`
- `receipts/a0_environment.txt`
- `receipts/esc50_source.txt`

