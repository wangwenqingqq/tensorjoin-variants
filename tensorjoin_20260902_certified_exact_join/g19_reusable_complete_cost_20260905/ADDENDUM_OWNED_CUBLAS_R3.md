# R3: owned cuBLAS engine, before any timing

R2 retained all four exact FP32 outputs but failed full shutdown memcheck with
three allocations (1,024 + 131,072 + 8,388,608 bytes) whose allocation boundary
was `cublasCreate_v2` through PyTorch's borrowed current handle. Explicit Torch
workspace/device/pinned-cache cleanup reached zero tracked allocations yet did
not destroy that borrowed handle. We will not destroy another owner's handle,
suppress leak checks, or use device reset to hide ownership errors.

R3 creates and destroys an independently owned cuBLAS handle. The GEMM call,
math mode 2, compute type 69, host pointer mode 0, algorithm -1, layout, stream,
input preparation, classification and output contracts remain unchanged.
An explicitly owned device workspace has the capacity returned by installed
`at::cuda::getChosenWorkspaceSize()` (typed size_t(void), declared in installed
CUDAContextLight.h, resolved symbol `_ZN2at4cuda22getChosenWorkspaceSizeEv`).
Require 8,519,680 bytes, matching R2's stable PyTorch live workspace footprint.
Set the stream before setting the workspace. No borrowed cuBLAS handle is
created. At close, synchronize, destroy the owned handle, release its workspace,
and execute the already tested R2 device/pinned-cache shutdown. Initialization
and final engine shutdown remain outside all per-call timing for every method.

The owned handle is a new comparator artifact, not presumed instruction-identical.
Repeat FP32's nine-fixture matrix and full memcheck. All pending stress/profiling
slots use R3; unaffected RT/TC passes are retained with explicit lineage. Resolve
the actual selected library function again before timing. If the new binding
changes the selected function, report it rather than assuming old code identity.
R2 stays failed; no performance conclusion is drawn from any sanitizer run.
