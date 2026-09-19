# Secondary-card functional admission (2026-09-08 CST)

GPU2 trace3_a2 passes under guardR1. The next fixture_a0 was terminated after
foreign PID558777 appeared. Later a different user PID561084 occupies GPU2.
All interrupted records remain excluded; foreign processes are untouched.

Use currently idle same-host GPU0, UUID GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1,
for functional and sanitizer qualification. GuardR2 only adds a separate target
manifest and GPU0 lock paths; all isolation and bounded tombstone rules remain.
This does not waive the original-card gate for timing admission. No timing
campaign can start on the basis of this transfer alone. A final same-binary
original-GPU2 confirmation remains pending if GPU0 gates pass.

The numeric GPU source and binding are unchanged. `control_r1.py` canonicalizes
compiler metadata tuples to JSON lists before identity comparison: static
inspection identified a potential second-capture false mismatch. No numeric or
execution change. `validate_control_r1.py` records physical GPU, uses new gate
labels, and clears held vector allocations before teardown. These are explicit
harness revisions, not retroactive repairs to failed data.
