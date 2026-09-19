# Gate 0: Analytic Soundness Before Breadth

Date frozen: 2026-09-03

## Decision

**CONTINUE with one cheap analytic-bound gate before any new dataset or formal
performance campaign.**

G2B establishes exact output empirically and exhaustively for one frozen
dataset, radius, and hardware/software state. It does not establish that the
current fixed `BOUND_PAD=1e-4` is a universal outward floating-point bound. A
database/systems paper may report the measured exact result, but it cannot call
the operator generally certified until this gap is closed.

## Novelty kill test

- QDOT already derives deterministic error bounds for approximate dot
  products. A generic "error-bounded mixed precision" contribution is prior
  art, not a new thesis.
- FaSTED already applies FP16/FP32 Tensor-Core distance computation to
  similarity self-join. Merely adding a conservative guard and FP64 fallback is
  too close to established filter/refine structure for an upper-strategy claim.
- Recent bit-accurate Tensor-Core modeling work reports architecture-dependent
  floating-point behavior. That strengthens the need for an explicit execution
  contract; it does not create novelty for us.
- The surviving contribution remains the joined system mechanism: exact
  complete-output materialization, output-sensitive precision routing, and its
  measured advantage over exact GPU indexes.

Primary sources:

- QDOT: https://doi.org/10.1137/21M1406994
- FaSTED: https://arxiv.org/abs/2508.21230
- Accurate Models of NVIDIA Tensor Cores, v4:
  https://arxiv.org/abs/2512.07004
- NVIDIA PTX ISA integer MMA and square-root semantics:
  https://docs.nvidia.com/cuda/parallel-thread-execution/

Novelty is an inference from the searched literature, not proof that no
unindexed or unpublished work exists.

## Why the current integer path is promising

The dense stage uses signed INT8 operands and INT32 accumulation. For D=512 and
codes restricted to `[-127,127]`, the largest absolute accumulated magnitude is
`512*127*127 = 8,258,048`, below both `2^24` and `2^31`. Thus the integer dot
product neither overflows INT32 nor loses information when converted to
float32. The unresolved soundness terms are:

1. float32 scale products and reconstructed squared-distance arithmetic;
2. the host-computed upper bound on per-vector reconstruction error;
3. square-root implementation error and final interval arithmetic;
4. the FP32 refinement stage's fixed `1e-3` decision guard.

G3A addresses only whether a conservative analytic bound for items 1--3 stays
selective enough to justify a GPU implementation. It does not retroactively
prove the existing kernel or the FP32 refinement stage.

## Stop and promotion rules

Run `PROTOCOL_G3A.md` on the frozen G2A `4096x512` source before editing the G2B
keeper.

- **Stop the current proof route** if the analytic reconstructed-distance
  interval has any observed containment/decision violation or increases INT8
  ambiguity above 1.25x the frozen G2A2 count.
- **Promote only to a separate GPU proof candidate** if G3A passes. The GPU
  candidate must then expose the intended PTX/SASS math path, pass adversarial
  and frozen-oracle correctness, memcheck, and sustained tests before any
  performance measurement.
- Keep the accepted G2B source immutable. Do not silently replace its empirical
  certificate or rewrite the G2B result as analytically proved.

Breadth resumes only after the proof candidate either passes or is retained as
a diagnosed negative result. More modality ports cannot override a soundness
failure.

