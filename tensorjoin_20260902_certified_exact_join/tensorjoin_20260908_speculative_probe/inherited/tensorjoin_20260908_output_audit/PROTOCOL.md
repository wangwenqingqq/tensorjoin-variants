# Shared output audit, 2026-09-08

Purpose: improve the common complete-output executor and reassess fixed F8,
F16 and pedantic FP32 pipelines. This is engineering/baseline work, not a new
precision planner or novelty claim. The previous threshold-sweep conclusion
and its frozen measurements remain unchanged.

Input, six thresholds, row order, exact FP64 reference outputs and pair IDs
are inherited from precision_routing_20260908_threshold_sweep. Output remains
all directed uint64 pair IDs, self once, sorted ascending on the CPU. F8 and
F16 use the same retained arithmetic kernels, tile scheduling and precision
queues. FP32 uses the retained pedantic cuBLAS panel path, frozen metadata,
classifier and FP64 terminal, with threshold only as a runtime parameter.

Three common output implementations:

- legacy: download each upper-triangle batch, sort upper IDs, mirror, sort.
- cpu_single: same downloads, concatenate unsorted upper IDs, mirror, sort
  once. The final global sort already determines the full result ordering.
- gpu_radix: retain copies of each upper-triangle batch on GPU, concatenate,
  mirror integer IDs, CUB unsigned 64-bit radix sort, then download the sorted
  complete output. The extra self mirror is UINT64_MAX, sorted to the end;
  count self entries and omit exactly those sentinels from the CPU result.
  No distance arithmetic, result truncation, compressed output contract,
  caching of input metadata, or use of reference output sizes in allocation.

Library loading, compilation and reference/hash checks are excluded from
timing. The complete clock includes tile lists, allocations, metadata, H2D,
classification, queues, count synchronization, output collection, integer
mirror/sort, temporary allocation, and final D2H into CPU RAM. Any diagnostics
are separate instrumented calls, never substituted for complete timings.
Output allocation is based on observed queue counts. Record peak allocated,
reserved and device-used memory plus CUB scratch size and output transfer bytes.

Admission: all 6 x 3 x 3 full configurations must match reference arrays and
hashes, with identical arithmetic stage counts across output variants for
each method/cell. Also exercise GPU output on empty, all-self, ragged and
shuffled upper-ID inputs, high IDs above signed 32-bit range, and canaries.
Capture the selected F16 kernels and verify byte identity against the sweep.
Hash the CUB shared library, source, compiler and all included build headers.
Keep the original retained FP32 library and kernel identities verified.

Exploration: two fresh processes, all 54 configurations, one warmup and two
retained calls per configuration. Rotate/reverse cells; use rotations of
the nine method/output combinations and alternate cycle direction. Select
one common output implementation by the lowest mean of per-process medians
over all 18 method/cell combinations with equal weights, ties toward legacy,
then cpu_single. No per-method or per-query output selector is introduced.

Confirmation: freeze the selected common implementation and admission/code
evidence. Three fresh primary processes followed by three fresh repeat
processes; all methods and cells compare legacy with selected output, one
warmup and four retained paired repetitions per method/cell. Balance both
method order and output order. If legacy wins exploration, confirm legacy
against the best nonlegacy candidate and report the failed replacement.

Report output speedup legacy/selected, and fixed-path speed ratios under the
same selected output. Use arithmetic means of process medians, paired process
and within-process repetition bootstrap, 20000 draws and fixed seeds. Analyze
primary and repeat blocks independently. No sample is removed based on time.
An output replacement is recommended only if both blocks support >=5% average
time reduction over the equal 18-cell mixture and no reproducible >5% cell
regression; otherwise retain the existing implementation or label a tradeoff.
Pairwise planner headroom, if shown, is descriptive under the new executor;
it does not reopen the prior stopped planner claim or establish novelty.

Resource contract: same GPU2 UUID and unmodified exclusive guard, 30 seconds
idle before each guarded process, <4 GiB device use, <=1800 seconds/process.
No foreign process termination or clock changes. Any resource/correctness
failure stops that process and remains in the archive. Implementation repairs
require a documented amendment and fresh admission before retained timing.
