# Decision: H2B-P2 Runtime cuBLAS Kernel and SASS Audit

Date: 2026-09-03

## Decision

**PASS P2 selected-function precision audit; admit P3 sanitizer and stress.**
Nsight Systems binds each frozen pedantic `cublasGemmEx` call to one actual
runtime launch inside its dataset-specific NVTX range.  Nsight Compute then
profiles that exact selected function and exports its SASS.  Both selected
functions use scalar/SIMT FP32 `FFMA`; neither contains a TF32 conversion nor
any MMA-family instruction.

This is generated-code evidence for the two exact P1 token-dot shapes.  It is
not a sanitizer, stress, latency, Tensor-Core-use, streaming, or system result.

## Runtime-selected functions

| Workload `(M,N,K)` | Runtime-selected function | Grid / block | Registers/thread | Dynamic shared/block | Static SASS | FP32 `FFMA` | TF32 / any MMA |
|---|---|---:|---:|---:|---:|---:|---:|
| PANNs `(640,2560,2048)` | `cutlass_80_simt_sgemm_256x128_8x4_tn_align1` | `(40,2,11)` / `(256,1,1)` | 212 | 49.15 KiB | 4,288 | 1,152 | 0 / 0 |
| R3D-18 `(662,2684,512)` | `cutlass_80_simt_sgemm_128x64_8x5_tn_align1` | `(88,3,3)` / `(128,1,1)` | 130 | 30.72 KiB | 2,496 | 576 | 0 / 0 |

The normalized selected-function SASS hashes are respectively:

- `4e7a7c2ebfcc31ffd3365a23729d34225149181312e26461930c071452c20753`;
- `fa5dd1a91b572f36e7049737a3a44dcc470c9576cec221d2683c60cfe8d26623`.

Explicit static counts for `F2FP.TF32`, `HMMA`, `IMMA`, `MMA`, `QMMA`,
`QGMMA`, and `WGMMA` are zero in both functions.  NCU reports zero local
memory spilling requests, zero static shared memory, and 1.02 KiB of
driver-managed shared memory per block in both cases.

## Attribution and provenance

- Target: `gpu-host-8`, physical GPU1, UUID
  `GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3`, SM120.
- Nsight Systems: `2025.5.2.266-255236693005v0`.
- Nsight Compute: `2025.4.1.0`, build `37053803`.
- cuBLAS runtime: `130100`; math mode `2`, compute type `69`, FP32 A/B/C,
  algorithm `-1`, and `NVIDIA_TF32_OVERRIDE=0`.
- `libcublas.so.13` SHA-256:
  `e70f38efabe986acd5eb683497c62f0f1730a6176ee291d9d24c6e339d1fbf86`.
- `libcublasLt.so.13` SHA-256:
  `656298c804f5adbb0df930545c17911b9584ab4e5101c0eeb65d1fe881d880f8`.

The NCU reports and exported SASS are tied to the profiled runtime launches.
The trace did not expose a parent static/JIT image identifier, so parent-image
linkage remains unresolved; this limitation is recorded rather than inferred
from a similarly named library symbol.

## Failed attempt retained

The first NSYS invocation supplied `env` as a relative executable.  NSYS
rejected it before CUDA launch; no measurement from that attempt is used.
`raw/h2b_p2_nsys_esc50_panns_attempt0.log` has SHA-256
`80b48f8ce0d445af929304aaaa13dc1ad0074f43df03ed514d41c89c435407a0`.

## Evidence

- Audit summary: `results/h2b_p2_cublas_sass_audit.json`, SHA-256
  `c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77`.
- Summarizer: `src/summarize_h2b_p2_cublas_sass.py`, SHA-256
  `e18abf6f18237b8f7cba2ba7ef89de2b2ffa9264010231af5c313960f218b49e`.
- Raw selected-function SASS CSVs: SHA-256
  `97cc8bb019d6c9085f33dfafcf53d991dcbccb56e1522cadecbfc3344d049e4a` /
  `47b7f65573c4faa42e98299e85e95b3ebb3a4b2c352d14829669e0589d27c5f5`.
- NCU reports: SHA-256
  `de5a2e317b6c2f381bd70c6687bd1ce9000fe62a0c6e247464435c2b9106998e` /
  `5c7dc3e5a9f340185c15d7a3f3d6b8b2fa3a726ae5b829fcbc9995b209fa48fc`.
- NSYS reports: SHA-256
  `cf4fb3b2b3818c7bc88e71f1faf246ffcb03187ddf7823f58758fd7673196849` /
  `08ce14215a4e0e64886a813111b1ddd79fb8118751ee6847c4c626d6ded2ea3f`.

## Next gate

P3 must run bounded actual-data Compute Sanitizer checks and isolated long-loop
output/count/hash stability for the full H2B operator, not merely the SGEMM.
Only a clean P3 may admit P4 outer-wall timing.
