# G11 Decision: Reject the Duplicate-Removal Workload Premise

Date: 2026-09-04. Status: completed CPU structural census; no GPU admission.

## Conclusion

**Neither predicate passes the frozen three-part opportunity screen in both
seeds. Do not build the proposed duplicate-removal GPU operator from G11.**
This is a scoped workload-budget decision, not a general impossibility result
for graph-restricted vector joins or factor-preserving execution.

The positive evidence is important: common-reference eligibility compresses
well and has broad dense-factor coverage. Its main opportunity is potentially
avoiding expanded candidate representation / preserving reuse, NOT removing
large numbers of repeated distance computations. That alternative does not
automatically pass novelty: standard factorized execution is already prior art.

## What was actually done

1. Audited the nearest mechanisms before the census; see
   `SOURCES_G11_RELATIONAL_GATE0.md`. Generic filtered vector search, range
   masks, relational Tensor-Core execution, and factorized joins are not new.
2. Froze `PROTOCOL_G11_RELATIONAL_CENSUS.md` before implementation and data
   inspection, SHA-256
   `97e18ef1af988fc93755c8d084733e6cf982fd221b97580d473c91cf337e4ad2`.
3. Downloaded official OGB arxiv version 1, checked archive CRC and exact source
   inventory: 169343 nodes, 1166243 directed edges, zero duplicate edges.
4. Ran four CPU cells: two graph predicates x two independent endpoint samples,
   each 4096 query by 16384 disjoint database nodes; all witnesses retained.
5. Verified all 16384 query rows by an independent adjacency-union oracle;
   witness multiplicity sums and all decision predicates agree.

No vector-distance predicate, model training, CUDA kernel, GPU timing,
manuscript edit, or Overleaf synchronization was performed. The source provides
128D features for a possible later test, but G11 uses citations only. These are
researcher-constructed query batches, not production logs or OGB's official task.

## Results (all four cells)

E is unique eligible pairs. W is witness-expanded occurrences. Panel/E includes
duplicate computations and tile padding. Representation sizes exclude the
shared original graph, vector storage, allocator overhead and construction;
all must be charged separately in a future end-to-end experiment.

| Predicate | Seed | E | W/E | Factor / unique-CSR bytes | Fat-factor unique coverage | Literal 16x16 panel / E | Screen |
|---|---:|---:|---:|---:|---:|---:|---|
| Common reference | 20260904 | 1,347,583 | 1.3585 | 0.1124 | 93.52% | 3.3894 | fail: W/E <1.5 |
| Common reference | 20260905 | 1,293,977 | 1.3607 | 0.1168 | 93.33% | 3.4768 | fail: W/E <1.5 |
| Common citer | 20260904 | 24,678 | 2.0993 | 4.5532 | 2.34% | 163.0006 | fail: size and coverage |
| Common citer | 20260905 | 23,135 | 1.9844 | 4.3135 | 0.00% | 157.0524 | fail: size and coverage |

Frozen screen: W/E >=1.5, factor bytes <=25% of unique CSR, and >=50% of E
covered by witnesses with at least 16 endpoints on both sides. Both seeds must
pass for a predicate. No seed, empty row, or failed statistic was discarded.

### Preserve the material positive and the stronger controls

For common-reference cases, factor payload is 608628–609516 bytes versus
5208684–5423108 bytes for unique CSR: roughly 8.6–8.9x smaller as a compact
representation. This is NOT a measured memory-bandwidth or latency speedup.
The simple random-layout 16x16 block-mask control executes 46.25–47.49 slots
per eligible pair; preserving witness groupings executes 3.39–3.48. Thus this
real structural signal must not be rewritten as “all TC layouts are hopeless.”

Conversely, ordinary equal-neighborhood row grouping is a stronger structural
control than literal witness panels for common-citer cells. Its 16x16 slot
counts are 709632/695808 versus witness-panel 4022528/3633408. Discovering
those equal neighborhoods is not free. These controls are CPU work ledgers,
not implementations or measured comparisons to cuVS, VecFlow, or FlashMask.

W/E bounds the distance-work gain from perfect witness deduplication alone
against an expanded evaluator. A fair deduplicated CSR keeper already computes
each eligible pair once. Neither W/E nor padding is a wall-time model, and
padding alone does not prove that a Tensor-Core implementation is slower.

## Three separate decisions

| Layer | Decision | Confidence / reopen condition |
|---|---|---|
| Implementation | CPU census implemented and verified; no candidate vector operator | High for recorded counts and sampled data |
| Mechanism premise | Duplicate-removal thesis fails this workload screen | High for four cells; reopen only with independently motivated real workloads and a fresh, non-favorable sampling contract |
| Paper thesis | Novelty remains unestablished | No algorithm or same-contract end-to-end advantage yet; no manuscript promotion |

## Next action, not an automatic experiment authorization

If continuing this branch, first specify a genuinely differentiated
factor-preserving vector operator relative to standard factorization plus a
grouped/CSR/masked scan. It must explain how overlapping witnesses are handled
without a hidden O(E) expansion and without replacing a simple operation with
an expensive ownership construction. Merely renaming group-by plus GEMM fails.

The common-reference case is retained as a concrete test fixture for that
design question, NOT as a passing G11 result. A changed thesis needs a new
Gate-0 card and frozen cost contract. Do not lower the 1.5 threshold, increase
batch size, resample for higher degrees, or start GPU tuning to turn G11 green.
If no such difference can be stated, leave this branch parked and change the
problem rather than repackage the existing TensorJoin paper.

## Evidence and reproduction

- `src/run_g11_relational_census_cpu.py` (defaults require no flags).
- `results/g11_relational_census_cpu_a0.json` and
  `raw/g11_relational_census_cpu_a0.log` contain all raw counts and controls.
- Official data: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
- Archive SHA-256:
  `49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276`.
- The first HTTP transfer ended early; an explicit byte-range resume completed
  the same archive. Retain the resume headers and transport note. No changed
  source, reduced data, or scientific rerun was used.
- Source/protocol/data/result hashes and state snapshots are in
  `artifacts/g11_relational_census/`.

```bash
cd @LOCAL_WORKSPACE@/paper/tensorjoin_20260902_certified_exact_join
/usr/bin/python3 src/run_g11_relational_census_cpu.py \
  --output results/g11_relational_census_cpu_independent.json
```

The script refuses an existing output. The independent output path avoids
overwriting A0; rerunning is not necessary for this completed census.
