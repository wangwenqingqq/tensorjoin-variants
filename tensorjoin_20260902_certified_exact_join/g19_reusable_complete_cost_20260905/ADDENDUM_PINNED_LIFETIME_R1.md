# G19 R1: repair the observed OWL pinned-allocation lifetime

Added after A0's four complete-ID matrix slots passed and RT memcheck failed.
No public timing was collected. The A0 campaign/result/binaries remain unchanged.

## First failing boundary and source evidence

The exact RT-off wrapper returns all four sanitizer-case outputs correctly.
At g19Destroy -> owlContextDestroy -> LaunchParams::DeviceData destruction,
Compute Sanitizer reports cudaErrorInvalidValue at cudaFree and a leaked
72-byte allocation originally made by cudaHostAlloc for launch parameters.
A0 memcheck therefore fails with two errors despite the inner numerical pass.

Pinned OWL's `owl/DeviceMemory.h::PinnedHostMem` allocates with cudaMallocHost,
but both its destructor and resize release with cudaFree. This is an ownership
API mismatch, not candidate arithmetic, map correctness or tensor precision.
A tiny same-process allocation/wrong-free/correct-free probe will independently
record the API statuses; it is diagnostic localization, not a clean sanitizer
gate. The already observed full-operator sanitizer receipt is retained.

## Narrow new artifact

Create adapter_r1 from the complete G19 adapter_a0. Copy the pinned OWL source
into this new artifact, replace only the two PinnedHostMem release calls by
cudaFreeHost, and inventory before/after hashes and the full source tree.
Never alter the shared upstream/OWL directory or G17/G18, nor the A0 binaries.
No allocation, stream, device kernel, RT traversal, geometry, threshold or output
algorithm changes. Fresh build_off_r1/build_on_r1 libraries are independently
frozen. Reconfirm the actual selected CUDA functions and all output/work gates.

## Admission restart, without weakening the protocol

Rerun diagnostic-on and diagnostic-off nine-input forward/reverse RT matrices,
full RT memcheck and synccheck, and both engine cycles with 100 complete calls.
The earlier TC/FP32 nine-input matrix slots remain valid because their code and
binary dependencies are unchanged; run their still-pending safety/stress/profile
slots as initially declared. Preserve all A0 observations; do not relabel A0
as a complete pass. Continue to demand full leak checking and unchanged memory
growth bounds. No new latency sample or estimator choice is made in this addendum.

A distinct operators_r1 / run_slot_r1 / supervise_r1 / admit_r1 layer selects only
the new RT build paths; the original files remain frozen. Additive hashes bind
all R1 files and both new libraries. Single-engine and fail-closed exception
boundaries remain unchanged.
