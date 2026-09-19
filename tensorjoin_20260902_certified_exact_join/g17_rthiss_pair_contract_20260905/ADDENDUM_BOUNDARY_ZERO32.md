# G17 boundary-coverage addendum

Declared during the original matrix, before a timing campaign (none exists).
A source review of the fixture generator found that `boundary31` assigns a
nonzero coordinate to every row. Consequently the intended distance-to-zero
witness `(1,2^-13,...)` versus zero is absent; that case cannot establish its
intended unit-threshold coverage. Preserve the original input, observations,
protocol and generator unchanged. This is a fixture-coverage gap, not a measured
kernel failure or license to reinterpret old tests.

Add exactly one independent `boundary_zero32` fixture: append the all-zero
endpoint to the frozen original31 rows, with the same D512, epsilon1, FP64
threshold1, reorder mode and unchanged adapter binary. No random resampling,
threshold tuning or replacement of unfavorable runs. This explicitly tests
1+2^-26>1 (FP64 rejects while FP32 can round to1) and1+2^-54 (frozen FP64 itself
can round to1, so it is not an exact-real predicate test). Run normal, memcheck
and synccheck as separate append-only records. It is a structural/numerical
probe, never a performance comparison. Include it in the final admission
statement even if the original matrix passes.
