# G15-B FP32-first design card

Frozen before kernel implementation and GPU measurement, 2026-09-05.

## Numerical and task contract

The operator computes the complete upper-triangle self-join, then mirrors and
sorts directed IDs. D=512, finite stored FP32 coordinates with abs(x)<=1,
nonnegative exactly-FP32-representable squared thresholds. The public threshold
and full-array hash are those in `PLAN_AND_GATE0.md`. The terminal is the
unchanged G5 direct-FP64 reduction (BLOCK_K=256). Reference equivalence is the
claim; arbitrary exact-real arithmetic and arbitrary dimensions are not.

H2B's existing `norm_metadata` computes FP64 squared-norm centers n, conservative
norm radii nr and L2 upper bounds l. These metadata are *inside* the public timer.
Let p be the actual pedantic-FP32 GEMM output, promoted exactly to FP64. With
u32=2^-24, k=2*512+2 and gamma=k*u32/(1-k*u32), the new classifier uses:

```
r_dot = gamma*l_i*l_j + 4*512*tiny32
c = n_i + n_j - 2*p
m = abs(n_i) + abs(n_j) + 2*abs(p) + nr_i + nr_j + 2*r_dot
r = nr_i + nr_j + 2*r_dot + 32*eps64*m + 1e-12
    + 2^-38*(abs(n_i)+abs(n_j)+nr_i+nr_j)
L = max(c-r, 0); U = c+r
```

The last term additionally covers the retained FP64 terminal's rounding on the
admitted domain; it is not a claim that FP64 equals exact-real arithmetic.
All scalar coefficients are explicit FP64 compile-time constants; do not
silently round the gamma coefficient through a runtime FP32 scalar. Accept
when U<=T, reject when L>T, otherwise emit a pair for the same FP64 terminal.
No tolerance is allowed in final pair equality. Containment gates use direct
FP64 oracle sums with the independent terminal allowance; direct-decision gates
use zero mismatch. Any observed enclosure or output failure stops this fixed
formula, rather than widening it after seeing the failed data.

## Hardware, geometry and work hypothesis

SM120 GPU 2 as in the plan. Dense GEMM is delegated to the actual queried
pedantic cuBLAS implementation; no TC or low-precision path is assumed absent
until selected-code inspection. A triangular 4096-row-panel schedule has 120
GEMMs for N=60000. Full/tail shapes are frozen in the plan. The control performs
extra discarded diagonal lower-triangle dot work; this is counted explicitly.

The classifier is a linear 1024-pair CTA with 8 warps, one thread owning four
FP64 interval lanes. It loads the FP32 dot and FP64 per-vector metadata,
computes the bound, prefix-compacts accepts and ambiguity, and updates at most
one counter per nonempty class per CTA. No pair-coordinate loop is present in
classification. A separate unchanged G5 FP64 kernel visits only unresolved
pair IDs. This avoids using a deliberately slow direct-coordinate full scan.

### Ownership and phase-live-set table

| Phase | Owner/state | Last use and expected limit |
|---|---|---|
| GEMM | cuBLAS-selected CTA/register tiling | Full container and selected function audited separately |
| Metadata load | classifier lane: dot, row/column, n/nr/l for both rows | l last used at r_dot; n/nr last used in terminal allowance |
| Interval | lane: center, radius, L/U and class masks | FP64 intermediates dead before compaction prefix |
| Compaction | CTA: masks, int32 prefixes, scalar reservations | One unique slot per accepted/ambiguous lane; <=1024 reservations per class/CTA |
| Refinement | unchanged per-pair FP64 CTA | Same BLOCK_K=256 reduction and G5 terminal |
| Canonicalization | host | Same upper-sort, mirror and final-sort as G5 |

No custom cluster, TMA, TMEM, async producer-consumer handoff, or persistent
state. Triton owns prefix-scan synchronization. Shared-memory/barrier resources
and spills must be inspected, not assumed zero. No register cap or warp sweep.

### Ready/overwrite graph

Host metadata/input -> blocking H2D -> same-stream GEMM -> classifier -> blocking
counter read -> terminal (if needed) -> blocking final count and accepted-ID
copy -> next panel may overwrite score/ID buffers. Capacity is always 4096^2,
independent of oracle output or estimated selectivity. Overflow is fatal.
All global IDs use N-stride encoding, not a panel-local encoding. Only the
current default stream is admitted; Graph and arbitrary stream claims are
excluded. The FFI wrapper asserts the handle's originally selected stream.

## Validation and dispatch boundaries

- Rectangular panels, odd sizes 1/31/129/257 and public full/tail shapes.
- Real first-257 public rows; fixed-seed signed random rows; exact duplicates;
  all-zero; sign/cancellation-heavy vectors; one-hot/equality and FP32-subnormal
  coordinates; G10's 1+2^-54 exact-real counterexample remains a reference-scope
  fixture rather than a false exact-real claim.
- Full 60K canonical output equality against the frozen independent artifact.
- No new sources overwrite old ones. Compiled-cache identities are collected
  before/after public timing, and any new timed compilation fails admission.
- Memcheck/synccheck, selected precision audit, and 1000 two-buffer subset
  invocations precede performance promotion. Diagnostic-only timing is allowed
  to localize a bottleneck but cannot close missing gates.

Reject on wrong layout, status/math mode, nonfinite input/metadata, invalid or
duplicated output, unsafe direct decision, unresolved capacity, source drift,
timed JIT, or foreign GPU occupancy. A slow but correct control is retained;
do not call it the fastest possible FP32 algorithm without further evidence.
