# Gate2 a1: unused diagnostic-pointer type repair, before timing

The a0 fixture, public selected tiles and ten-program static/runtime-ABI audits
passed. Full qualification reached the final artifact capture, then stopped:
S8's non-dump wrapper passed int32 score storage as the two unused dump pointers,
whereas the fixture supplied FP32 dump arrays. TTIR/TTGIR pointer types and the
Triton compilation-key metadata differed. The actual cubin, PTX and LLIR were
byte-identical. raw/g2_identity_diagnostic_a0.txt retains the exact diff.

The first failing boundary is artifact identity serialization, not observed
numeric output or measured speed. The finally block retried the same capture
and prevented the result JSON from being written; a0 stdout, guard and cache
remain authoritative evidence of that failed attempt. Do not count the lost
in-memory full result as admitted or use its timings.

r1 keeps every kernel source, predicate and code object unchanged. It supplies
FP32 views of unused dump pointers outside the full-call tile loop, matching
fixture specialization, and preserves final-capture exceptions in the result
writer. A pointer view does not allocate score storage or change numeric work.
The same typed-view correction is applied to the profiling wrapper. Source is
versioned, not overwritten. Existing a0 compiled artifacts must still match
byte-for-byte, including IR and metadata. No performance sample has been
observed or replaced. A fresh full test, stress and sanitizers precede timing.

Retain a0 fixture/public/ABI evidence; rerun full and all not-yet-run stages with
a1 labels. The Williams timing schedule remains unconsumed and unchanged.
