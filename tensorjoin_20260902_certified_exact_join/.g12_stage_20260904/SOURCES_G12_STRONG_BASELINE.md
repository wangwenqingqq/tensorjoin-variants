# G12 Source and Attribution Ledger

Checked 2026-09-04. Primary sources only. These sources establish prior
mechanisms; none is claimed to duplicate every detail of an exact GPU join.

## FFX — the newly located, most relevant 2026 prior art

[Factorized and Vectorized Execution: Optimizing Analytical and Semantic
Queries over Relations](https://amine.io/papers/2026-sigmod-ffx.pdf),
SIGMOD/PACMMOD 2026, DOI https://doi.org/10.1145/3802055 .

Sections 4–6 preserve factorized intermediates in a packed, pipelined engine.
The semantic consumer is `llm_map`; Section 6.1 leaves other LLM primitives to
future work. Thus “preserve factorization into the consumer” is established,
but this is not evidence of an exact GPU vector-threshold implementation.
Do not confuse CPU vectorized execution with dense embedding search.

Public source: https://github.com/dais-polymtl/ffx , pinned commit
`463603793836279d7f6e74636821378b91595c5a`.
In `src/operator/join/packed_theta_join.cpp`, lines 216–256 apply a scalar
predicate over slices, update selectors, and invoke the next operator on
survivors. This is a source-level check, not runtime admission. The inspected
instantiation at line 286 is `uint64_t`, not a distance-vector kernel.

Paper and five source files are locally retained with MIT source license and
hashes in `artifacts/g12_factorized_baseline/ffx_provenance.json`. No FFX build,
benchmark, LLM service invocation, or new claim about its performance was made.

## Earlier foundations, not newly discovered contributions

- [FDB, PVLDB 2012](https://www.vldb.org/pvldb/vol5/p1232_nurzhanbakibayev_vldb2012.pdf):
  select-project-join evaluation over compact factorized representations.
- [Kuzu, CIDR 2023](https://mail.vldb.org/cidrdb/papers/2023/p48-jin.pdf), Section 3:
  factorized vector groups and join processing. It is an established comparator
  foundation, not evidence that an arbitrary embedding predicate is free.
- [TigerVector, SIGMOD Companion 2025](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD25_TigerVector.pdf),
  Section 5.4: graph-pattern vector similarity joins. G12 does not alter G11's
  admission boundary or claim a new problem merely by changing top-k to range.
- VecFlow, FlashMask, TCUDB and theta-join factorization remain covered by
  `SOURCES_G11_RELATIONAL_GATE0.md`; they have not been benchmarked in G12.

## What is our derivation, not a claim attributed to those systems

`DESIGN_G12_STRONG_BASELINE.md` gives the elementary selection/DISTINCT
identity and constructs a blocked factor scan with output-only deduplication.
Its live-set bound and CPU reference are our explicit comparator derivation.
We do not attribute this exact operator to FFX or pretend its GPU performance
has been measured. The constructive counterexample suffices to refute the
alleged necessity of global pre-predicate candidate materialization.

## Search boundary

The search covered factorized filtering, duplicate elimination, vectorized
factorized engines, graph-pattern similarity joins, and grouped matrix
evaluation. It does not prove that every remaining scheduling or packing
mechanism is known. It does establish that the current broad contribution
phrasing is inadequate and that the proposed saved cost is avoidable by a
standard comparator construction.
