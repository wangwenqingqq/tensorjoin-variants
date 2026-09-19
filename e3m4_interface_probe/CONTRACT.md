# SM120 E3M4 callable-instruction survey

Status before execution: designed, not validated. Date: 2026-09-19.

## Question and prior art

Can an owned, reproducible probe call E3M4 on both operand sides of plain and
block-scaled SM120 dense m16n8k32 MMA with FP32 accumulation? This is independent
validation and interface construction, **not first discovery**. Public prior art:

- [blackwell-isa](https://github.com/kacper-daftcode/blackwell-isa/tree/8fe1478006d14bb52f6f644d4f7dc1993a85f25a)
  records E3M4 type code 2 and hardware-tested dense QMMA.SF combinations.
- [cubit](https://github.com/kacper-daftcode/cubit/tree/1eb8dc68d977c323fb6a24a60b8b76cc0813bc59)
  supplies an assembler/patcher, not a tuned E3M4 GEMM API. Both source snapshots
  have MIT licenses; no upstream implementation is vendored here.
- [PTX 9.1](https://docs.nvidia.com/cuda/archive/13.1.1/pdf/ptx_isa_9.1.pdf)
  documents the five other f8f6f4 types and the scaled instruction syntax.

## Frozen target and design

- RTX PRO 6000 Blackwell Server Edition, SM120, CUDA 13.1.115, driver 590.48.01.
- Physical GPU 0, supervised by the existing campaign guard. No clocks, power,
  driver, GPU reset, or foreign process changes. Stop on foreign occupancy,
  CUDA error or timeout; preserve diagnostics. Binaries are private copies.
- Two separately compiled families: plain kind::f8f6f4 and block-scaled
  kind::mxf8f6f4, both m16n8k32, row/column layout, FP32 zero accumulator.
- One CTA/warp per test record. A and B each repeat one raw code throughout
  their logical matrix; each warp writes all 128 FP32 output elements.
- All 25 documented format pairs are compiled first. The six operand bits
  must predict the whole compiler-generated matrix. Only then introduce the
  published E3M4 selector into a copied E4M3 baseline; no other bit may change.
- Formats: E4M3FN, E5M2, hidden E3M4, E3M2, E2M3, E2M1. Test all 36 pairs in
  each family. The 11 E3M4-containing pairs per family are the target, with
  documented-format controls remeasured in the same run.
- E3M4: sign/exponent/mantissa = 1/3/4, bias 3, min subnormal 1/64,
  maximum finite 30, magnitude code 0x7f is NaN. E4M3FN raw 0x10 is 1/32,
  **not** 1/8; E3M4 raw 0x10 is 1/4.
- Test each valid raw code on each operand axis with the peer set to one,
  deterministic finite mixed pairs, and explicit anti-E4M3 cases. For the
  scaled family add powers-of-two scale controls; all four packed scale bytes
  are identical and byte/thread selectors are zero. Uniform inputs do NOT
  establish nonuniform lane, scale-selector, sparse-metadata or GEMM layouts.
- CPU binary64 decode computes the uniform dot oracle, then casts to FP32.
  Finite results must match exactly, NaNs by classification, infinities by
  sign. Signed zero is not an admission requirement. Chosen values and scale
  exponents avoid finite-output FP32 underflow/overflow and accumulation loss.
- No producer/conversion instruction is inferred: Python supplies raw bytes;
  quantization, GEMM, TensorJoin certificates and throughput are out of scope.

## Resource / readiness contract

Each lane owns four persistent FP32 outputs, one repeated A word, one repeated
B word and two scale words. Input load -> synchronous warp MMA -> four stores.
There is no shared memory, TMEM, asynchronous stage, barrier, or inter-CTA
communication. Last use of inputs/scales is the MMA; outputs die after stores.
One useful MMA per CTA; this is a launch-bound correctness probe, not a
throughput measurement. No performance comparator, estimator or speed gate is
declared because no performance claim will be made.

## Gates and explicit exclusions

1. Compile and structurally inspect all 50 documented controls, exact one MMA.
2. Verify operand-field prediction and exact patch-only binary differences.
3. Each format/family runs in a fresh process with a hard timeout. Check all
   outputs, not just a sample or successful launch.
4. Validate anti-fallback and exhaustive raw-code axes; preserve failures.
5. Memcheck all 22 E3M4-containing probes, plus repeated-launch stress and
   synccheck/racecheck/initcheck on the two E3M4/E3M4 family representatives.
6. Keep raw output bytes, hashes, code/source/binary manifests and full results.

Reserved absolute type codes 6 and 7 are not launched on a shared host. Their
absence is NOT a measured illegal-instruction result. Sparse QMMA, F16 output,
other shapes, F2FP conversion, SM100/tcgen05, arbitrary 128-bit opcode sweeps,
full GEMM, performance and product admission remain untested here.

Rollback: stop only the owned process group; delete no existing source, cache,
or prior result. No production dispatch is changed. Publish the probe, curated
numeric evidence, source/binary hashes and exact limitations on the existing
private precision-format-routing branch, not large binaries or device config.
