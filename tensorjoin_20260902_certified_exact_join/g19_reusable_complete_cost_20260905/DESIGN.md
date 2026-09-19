# G19 implementation and ownership design

## One bounded adaptation

Preserve the G18 repaired arithmetic and native RT/refinement/compression work.
Move data-independent OWL setup to an explicit singleton engine, reset all
per-input host/device state, and make diagnostic IO/counters compile-time
optional. This is a lifecycle/output adapter, not a new tree, pruning policy,
precision bound, tensor tile or work schedule. Kernel ownership and ready edges
do not change. The counter removal can reduce diagnostic overhead but is not
itself a TensorJoin speedup claim.

| Owner | Lifetime / release |
|---|---|
| Engine | OWL context/module, geom type/object, params stream, raygen/program/pipeline; destroyed by owlContextDestroy after synchronized last use. Single nonconcurrent engine only. |
| Operation | Input malloc clone -> dimension reorder -> free after preparation; reordered grouping allocation freed after synchronous OWL uploads. |
| Geometry | Fresh user group and instance BVH per operation; detach raygen group then release instance/group handles after all launches finish. |
| OWL input/work buffers | Operation-owned handles; detach geom/param references, then owlBufferRelease after synchronized final consumer. Do not use owlBufferDestroy alone: pinned OWL source leaves its handle tracked. |
| Raw CUDA buffers | Native refine locals already freed; add owner cleanup for main's primitive-order allocation and pinned mask staging. |
| State | THRESHOLD reset to 0.01*epsilon, MAX_KD_LEVELS recomputed from current N; dimension/point/output maps, counters and diagnostic timings reset every call. |
| Output | Decode original IDs, sort once, transfer vector ownership to C API result; Python makes independent host output and destroys C++ result inside timer. |
| FFI error | Fail current process closed. Do not resume a partially executed operation after an exception. Only pre-entry invalid-domain tests may be retried. |

Public OWL null-buffer/group setters are confirmed in the pinned local source:
impl.cpp owlVariableSetBuffer/Group accept null shared pointers. Persistent params
therefore do not retain stale data references. Geometry, acceleration, SBT and
parameter buffer publication remain ordered before their consumers.

## Ready graph / live state

CPU clone -> dimension map -> conservative cutoff upload -> point grouping and
full H2D -> new BLAS/IAS and SBT -> intersection-count launch/sync -> allocation
and workload prefix -> RT candidates/sync -> native CUDA refine/sync -> mask
compression/sync -> required metadata and compact bits D2H -> host ID decode/sort
-> detach/release call resources -> independent output return -> next input.
No previous-call buffer may be read after release, and no engine may overlap calls.

The existing 1024-thread shared/shared CTA still owns query/candidate staging,
FP32 prefix and conditional FP64 sum, result-mask atomics and count reduction.
G18 uses 40 registers, zero stack/local bytes and 47,104 bytes dynamic shared;
those are reference observations, not an asserted G19 resource result. Removing
diagnostic atomics must not introduce spills or alter the 408-instruction
compression stream. No MMA/TMA/cluster/descriptor/warp/barrier redesign.

Expected deltas: three diagnostic global counter atomics per candidate become
zero (third is terminal-only); diagnostic full-mask readback and evidence writes
become zero; complete result/candidate mapping traffic stays charged. Engine
program construction moves outside the warmed-operation denominator while every
input-dependent BVH and SBT stays inside. No wall-time benefit is assumed before
same-contract measurement. Runtime/SASS and numerical gates are separate.

## Comparator boundaries

TC reuses G16 GPU metadata and unchanged three stage kernels, 64x64x64 stage1,
four warps, three stages, capacity 4096*64*64 and the same tile batching cap.
FP32 reuses pedantic cuBLAS GEMM plus GPU norm metadata and guarded terminal
refinement, panel/capacity 4096. N specializes to <=4096; no shape tuning.
Only explicit preservation of the stored threshold scalar may need a separate
source copy. Every difference and generated-code change must be inventoried.

No Graph/arbitrary stream or public engine-startup comparison is in this gate.
