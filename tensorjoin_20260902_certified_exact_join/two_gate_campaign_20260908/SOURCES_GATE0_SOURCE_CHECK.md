# Source-level cheap novelty kill test

Read-only source snapshot: artifacts/upstream/fasted_9af85ed8_source.tar.gz in
the parent workspace; three files extracted into this campaign's sources/.
Do not mislabel the snapshot as a newly fetched latest upstream checkout.
Primary paper: https://arxiv.org/html/2508.21230v1
Primary repository: https://github.com/bwcurless/FaSTED

`findPairs.cuh` lines 428-429 creates a register-owned WarpTile accumulator;
lines 460/498 accumulate K slices; lines 536-538 inspect those results inside
the same CUDA kernel. `warpMma.cuh` lines 230-242 accesses the D fragment and
forms squared Euclidean distance directly from its registers. Lines 257 onward
apply the threshold and emit result pairs, with no global dense-score matrix.
`ptxMma.cuh` supplies the FP16-input/FP32-accumulator MMA primitive.

Thus fused Tensor-Core distance -> threshold -> result emission in this exact
application family is prior art, not a TensorJoin contribution. FaSTED itself
is not admitted to our fixed-FP64-reference contract: its point representation
and reported quality are different. Its lack of our reference guarantee does
not make ordinary fusion new.

Closest filter family: https://arxiv.org/html/2208.00497v1 (static/semi-static/
dynamic floating-point filters followed by stronger arithmetic). This does not
prove that every TensorJoin implementation detail is formally subsumed in the
same setting; it does kill the broad claim that error intervals plus adaptive
higher precision are new by themselves.

The matched F16/S16 controls compose these conventional mechanisms under the
same declared reference, with no per-pair FP64 classifier handicap. The 2x2
screen is justified as a cheap decisive falsification of the remaining cost
interpretation, not as permission to package a subsumed thesis with more data.
Even a performance pass requires a specific non-incremental distinction beyond
native INT8 throughput and ordinary fusion. No such distinction is presumed.
