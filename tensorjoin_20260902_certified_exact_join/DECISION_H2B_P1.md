# Decision: H2B-P1 Direct-cuBLAS Correctness Gate

Date: 2026-09-03

## Decision

**PASS P1 correctness and repeatability; admit P2 generated-code resolution.**
The direct `cublasGemmEx` keeper, configured with queried
`CUBLAS_PEDANTIC_MATH`, explicit `CUBLAS_COMPUTE_32F_PEDANTIC`, FP32
inputs/outputs, and `NVIDIA_TF32_OVERRIDE=0`, reproduces the independent
direct-FP64 object oracle on all four full cells and all 14 frozen
adversarial/dimension cases.

This is correctness evidence for the exact API/configuration and observed
workloads.  P1 does not yet prove which runtime cuBLAS kernel executed, does
not substitute API settings for a SASS precision audit, and provides no
sanitizer, stress, or performance result.

## Full actual-data results

| Dataset | Target/query | Ambiguous objects | Fraction | Max observed dot error | Frozen max dot radius | Final result |
|---|---:|---:|---:|---:|---:|---|
| ESC-50/PANNs D2048 | 1 | 2 / 65,536 | 0.00305% | 2.017e-7 | 2.443e-4 | exact |
| ESC-50/PANNs D2048 | 8 | 9 / 65,536 | 0.0137% | 2.017e-7 | 2.443e-4 | exact |
| UCF101/R3D-18 D512 | 1 | 0 / 65,536 | 0% | 4.083e-7 | 6.116e-5 | exact |
| UCF101/R3D-18 D512 | 8 | 4 / 65,536 | 0.00610% | 4.083e-7 | 6.116e-5 | exact |

Across the four cells there are zero dot-radius, token/object containment,
unsafe direct-decision, output, duplicate, or overflow violations.  The
accepted H1 candidate and new cuBLAS keeper also return identical canonical-ID
hashes in every cell.

## Adversarial and repeatability evidence

Both D512 and D2048 pass each of:

- all-zero inputs;
- identical prefixes;
- sign-flipped prefixes;
- alternating-sign cancellation;
- one-hot inputs;
- small normal-scale inputs;
- ragged 4--7-token high-entropy objects.

All 14 cases have zero dot/token/object/direct-decision/final-output failures.
Two additional isolated processes reproduce all four actual-data count/hash
records and the same ambiguity counts (2/9/0/4).

## Runtime API provenance

- cuBLAS runtime version: `130100`.
- Library SHA-256:
  `e70f38efabe986acd5eb683497c62f0f1730a6176ee291d9d24c6e339d1fbf86`.
- Math mode queried after set: `2` (`CUBLAS_PEDANTIC_MATH`).
- Compute type passed to every GEMM: `69`
  (`CUBLAS_COMPUTE_32F_PEDANTIC`).
- Algorithm: `-1` (`CUBLAS_GEMM_DEFAULT`).
- `NVIDIA_TF32_OVERRIDE=0`; PyTorch global and CUDA matmul precision set to
  IEEE before the handle is acquired.
- Target: `gpu-host-8`, physical GPU1, RTX PRO 6000 Blackwell Server Edition,
  SM120.  The pre-run-only device revision is in
  `PROTOCOL_H2B_P1_R1_GPU1.md`.

The first implementation attempt stopped before GEMM because cuBLAS 13 exports
the pointer-mode API with `_v2` suffixes.  The repair changes only the typed FFI
symbol names; the failure is retained and excluded.

## P2 gate

Use a profiler launch trace to bind both full token shapes to the actual
runtime-selected cuBLAS kernel.  Record launch geometry, cuBLAS/container
hashes, and a selected-function SASS/resource ledger.  P2 passes only if the
observed dot kernel has no TF32 or lower-precision MMA path under the frozen
pedantic contract.  Profiler duration must not be used as public timing.

## Evidence

- Correctness: `results/h2b_p1_pedantic_correctness.json`, SHA-256
  `68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99`.
- Repeat 0: `results/h2b_p1_pedantic_repeat_0.json`, SHA-256
  `6970b41c794ec787e880778f9a501de61331a7eb4cda5303d85130db4215fc27`.
- Repeat 1: `results/h2b_p1_pedantic_repeat_1.json`, SHA-256
  `75d22742a4a952aa19bfb8b2a53bd36508b45bc9211a71fe1e51cb84b4edda30`.
- Kernel/runner SHA-256:
  `33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43` /
  `d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1`.
- Rejected attempt 0: `raw/h2b_p1_correctness_attempt0.log`, SHA-256
  `52c577057597329d88dcd9772bfc80338c1b7dc3e6c0611d9a528095a7a8523a`.
