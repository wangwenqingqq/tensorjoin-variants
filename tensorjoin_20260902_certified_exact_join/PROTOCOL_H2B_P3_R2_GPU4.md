# Protocol Revision: H2B-P3-R2 GPU4 Device-Only Revision

Date: 2026-09-03

## Reason for revision

The P3-R1 preflight found physical GPU1 occupied by external PID `4186371`
using approximately 45 GiB at 99% utilization.  No R1 sanitizer, stress, or
experimental kernel was launched, and the external process was not modified.
The preflight is retained at `raw/h2b_p3_r1_preflight.txt`, SHA-256
`88d0434a950eb52f2ecfe5a4102cbe4e870de78601a18b8e1b37f5c8c0060f7a`.

## Only change

P3-R2 selects physical GPU4 and lock
`@TENSORJOIN_ROOT@/.tensorjoin_gpu4_campaign.lock`.  GPU4 is the same
NVIDIA RTX PRO 6000 Blackwell Server Edition / SM120 model and was idle at
the device revision.  Its UUID is
`GPU-863c06a5-9f33-0265-b098-013fa840d5db`.

No data, shape, threshold, API, precision, certificate, output, sanitizer,
stress, or pass/stop rule changes.  P2 remains the exact selected-function
precision evidence on the same GPU model/toolchain.  P3-R2 must still record
the actual target UUID and does not turn this model-equivalence argument into
a cross-device performance claim.

The revised runner SHA-256 is
`63b19aab011f8a4e51f34c739011897ea5fc2a3afed99040622dd3c444d945ba`.
All instrumented logs use a new `*_r2.log` suffix.  The full R1 allocator
cleanup and leak-check contract remains in force.

## Current state

Rejected on its first full audio memcheck: exact output passed and all
experiment/cache workspaces were released, but the leak-free rule still found
three process-owned cuBLAS-handle allocations (8,520,704 bytes).  No later R2
subgate ran.  The separate access-safety contract is frozen in
`PROTOCOL_H2B_P3_R3_ACCESS_SAFETY.md`.
