# GPU metadata and directed predicate design card

Target SM120; existing warp-synchronous f16/f32 GEMM is library-owned. No
tcgen05, cluster, TMEM, custom producer handoff or barrier protocol is introduced.

## Numerical contract

For original x and explicit normal-or-zero FP16 z, residual r=x-z is exactly
representable in FP32 (Sterbenz for close nonzero z; zero case is exact).
Compute the three squared-norm centers with FP64 reductions. With
G=512*2^-52/(1-512*2^-52), each norm center s has radius G*s+1024*2^-1022.
The existing norm methodology supplies upper roots; store z/r upper roots
rounded toward +infinity in FP32. Store original norm endpoints rounded down/up.

Let lo_i,hi_i enclose ||x_i||^2 and l_i,e_i upper-bound ||z_i||,||r_i||.
For actual FP16-input/FP32-output dot p, assume the separately audited dot model
|p-z_i.z_j| <= g*l_i*l_j+4D*2^-126. Define every positive product/sum below
with FP32 round-up, g rounded up, and c endpoints with directed arithmetic:

    b = g*l_i*l_j + e_i*l_j + l_i*e_j + e_i*e_j +4D*2^-126
    c_lo = down(down(lo_i+lo_j)-2*p)
    c_hi = up(up(hi_i+hi_j)-2*p)
    R = up(2*b + up(2^-38*up(hi_i+hi_j)) +1e-12)
    L = max(down(c_lo-R),0), U=up(c_hi+R)

Power-of-two scaling by2 is exact on the admitted dot range. The original
norm uncertainty is carried by endpoints, not added again. Directed operations
replace the previous CPU expression's heuristic FP64 evaluation padding; this
is a new classifier with its own containment checks, not identical intervals.
Accept U<=T; reject L>T; compact uncertainty and invoke the unchanged middle
filter/terminal. The inherited terminal allowance remains explicitly scoped
to the fixed terminal/domain; there is no exact-real claim.

## Ownership, live state and ready graph

| Phase | Owner | Live state / last use | Layout and bound |
|---|---|---|---|
| Metadata | one128-thread CTA per row | x512 lanes, z, r; three FP64 norm reductions then endpoints/roots | row-major FP32 input and FP16 output, four FP32 metadata arrays; no retained vectors between complete calls |
| GEMM | owned cuBLAS kernel | library-specific accumulator/operand tiles | row-major FP16 Mx512 and Nx512, FP32 score MxN; exact selected function must be traced |
| Classifier |256-thread CTA/8 warps,1024 scalar positions | dot and four metadata values per row; b then endpoints, then dead before prefixes | all pair arithmetic FP32, one upper-pair owner; two prefix compactions, maxone reservation/class/CTA |
| Refinement | unchanged per-pair CTA | original FP32 coordinates, same counters | original cubins, fixed N60000 IDs |
| Materialize | host | upper arrays, mirror, sort | same canonical uint64 output contract |

No new cross-warp data handoff beyond Triton's reduction/prefix lowering.
Inspect generated shared/barrier code and apply allfour sanitizer tools.
Ready graph: blocking H2D -> metadata -> GEMM -> classifier -> blocking count
read -> same-stream stage2 -> count -> terminal -> result D2H -> buffer reuse.
Graph tests retain fixed inputs and calibrated counts; not dynamic dispatch.

Work hypothesis: compared with the CPU census, remove score D2H and CPU
classification, fuse3 metadata passes and casts/residual preparation. Compared
with A, C reads/stores61.44MB of FP16 vectors versus A's two30.72MB INT8 layouts;
C uses fewer numerical refinements in the earlier census but different native
TC throughput, interval instructions and120 versus108 scheduling batches.
Estimated dense score traffic is about7.7GB written then reread per complete
call; this is an added materialized-panel cost, not eliminated by TC arithmetic.
No time claim or unique novelty is inferred from these lower-bound terms.

Dispatch boundary:512-square validation and4096x4096,4096x2656,2656x2656 public
panels. No arbitrary N/D/T, nondefault stream or graph-API admission. Static
target: no local spills and no FP64 classifier arithmetic. If not met, retain
the failed target and explain before any bounded diagnostic exception.
