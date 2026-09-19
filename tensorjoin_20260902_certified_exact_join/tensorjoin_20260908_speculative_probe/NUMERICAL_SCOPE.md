# Correctness scope and work-set invariant

The answer contract is the inherited specified FP64 terminal on the frozen
represented FP32 vectors, followed by complete sorted directed CPU output.
This is not an unconditional real-arithmetic exactness claim.

The full verifier is the previous 64-dimensional projection filter. Coordinate
conversion, orthogonality, dot/norm accumulation and terminal allowances are
inherited without alteration from candidate_execution/PROTOCOL.md. The actual
cutoff is nextafter_float32((T+2^-20)*scale2+margin,+infinity), with the recorded
scale2 approximately 1.0000009536743277 and margin 0.013663386966621067. Admission
compares every full verifier flag AND every candidate pair count to the previous
filter for all six thresholds. Complete answers are compared to the same FP64
reference. These checks supplement the inherited source-level error argument.

For any draft tile set D and reliable undecided tile set U, including adversarial
D, the execution set is D union (U minus D). These two queues are disjoint and
their union contains U. Consequently all possible output pairs are visited, as
long as the inherited verifier is safe. Tiles in D outside U can only add work;
their final outcomes still use the full solver. No probabilistic acceptance rule
or draft threshold participates in final numerical decisions.

The sampled draft uses 16 rows and 16 columns per tile, at fixed offsets
0,4,...,60, and the same 64-dimensional coordinates. It is treated as fallible
even if the sampled pair calculation often agrees with the full verifier.
The algorithm does not require D to be a subset or superset of U.

The new scan_queue body is the original scan16 MODE=2 expression and compactor,
wrapped in a device-count condition and using indirect tile indices. Tiling,
512 dimensions in eight 64-dimension accumulation steps, GAMMA, directed
rounding primitives, arithmetic fusion setting and error expressions match.
The frozen native FP32 refinement and FP64 terminal binaries remain unchanged.
Admission with a forced all-tiles draft exercises the full input, including
diagonal/self and the final 32-row tail. Guard canaries test output bounds.

Compaction allocates one queue position per selected tile with atomic_add.
Queue capacity per group is the number of input tiles (at most 4096), so the
count cannot exceed capacity. Verification checks each queue against the full
expected flag set, including uniqueness and exact group membership. Draft and
repair scans append on ONE compute stream into shared counters and buffers;
they cannot race with each other. Their combined tile count is at most 4096,
so at most 4096*4096=2^24 pair positions can be visited per chunk. Each valid
position enters either the direct-output list or ambiguous list at most once.
Refinement and terminal preserve those same per-chunk bounds.

All inputs, count initialization and draft flags precede the verifier through
an explicit CUDA event. Each repair or reliable scan waits for the corresponding
verification event. The speculative draft scan deliberately precedes that wait.
No buffer used by the verifier is overwritten or freed before query completion.
Output collection is on the compute stream and all work completes before sorting.

This experiment tests one admitted input, its six thresholds and the inherited
numerical preconditions. It does not establish a new universal error theorem.
