# Candidate-generation scope: what the predicate repair does not prove

Source audit on the unchanged pinned G17/RT-HiSS files, 2026-09-05. This is a
scope check, not a newly measured counterexample or a general correctness verdict.

| Boundary | Live source anchor in G17 adapter | Observation and consequence |
|---|---|---|
| Input radius | `hostCode.cpp:40`, `utility_seq.cpp:1-39` | Epsilon is parsed to FP32, and projected points receive that radius. It need not exactly encode the radius implied by the frozen FP64 squared threshold. |
| Group enclosure | `utility_seq.cpp:576-706`, `constructBoundingSphere` | Centers, square roots, radius growth and containment tests use ordinary FP32 operations; no directed enclosure proof is present in this path. Do not infer all-input conservative enclosure from the function name. |
| RT AABB | `deviceCode.cu:10-20`, `SpheresBounds` | Bounds use ordinary center minus/plus radius. This is not an explicitly outward-rounded implementation. |
| RT query | `deviceCode.cu:54-67`, `rayGen` | A short x-direction ray begins at the projected query center; candidate accounting happens at the intersection callback. Driver/OptiX traversal semantics are a separate boundary from CUDA distance refinement. |
| Candidate accounting | `deviceCode.cu:23-47` | Count/write modes and capacity checks remain unchanged; G17 independently exports full candidate metadata. A successful allocation or count check is not completeness proof. |
| Final refinement | `utility.cu`, shared/shared comparison loop | G18 changes only the predicate and diagnostic counters; it cannot recover a pair that the RT candidate pipeline never presents. |

For G18, candidate-set identity is required separately from pair-output equality.
On each frozen input, the reconstructed unique candidate count must equal N*N,
as in G17. Together with verified candidate IDs and bounds, this establishes
complete candidate coverage on that finite matrix. It also means that a positive
refinement result on these inputs supplies no evidence of useful RT pruning.

A later sparse/selective or larger workload needs a new full candidate/oracle
check. If it misses a reference-positive pair, stop refinement-only admission.
The remedy must explicitly cover radius conversion, group enclosure, AABB
rounding and traversal—not an arbitrary epsilon inflation selected after a
failure. Preserve the native algorithm and report both contracts rather than
silently switching to all-pairs traversal. No such geometry repair or all-input
certificate is claimed by G18.
