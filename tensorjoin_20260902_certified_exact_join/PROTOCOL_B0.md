# Protocol B0: Locked PyTorch GPU Exact-Join Screen

Experiment ID: `tensorjoin_20260902_pytorch_gpu_exact_join_b0`

## Purpose and status boundary

B0 is a diagnostic implementation screen, not a promotable CUDA result. It asks
whether the A0 geometry translates into enough hardware and operator headroom
to justify a fused CUDA kernel. PyTorch library dispatch is used deliberately
to avoid writing a custom kernel before the direction passes this gate.

## Frozen hardware and isolation

- Host: `gpu-host-8`; live hostname `gpu-host-8`.
- Physical GPU: index 0, UUID
  `GPU-9bd364a9-9d88-0b78-ef55-caac6f73d6b1`.
- Expected architecture: compute capability 12.0 (`sm_120`).
- Driver observed before design: 590.48.01.
- Environment: `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`;
  PyTorch 2.11.0+cu130; CUDA runtime 13.0.
- Lock: `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`.
- Abort a process before launch if GPU 0 has an unrelated compute process.
  Do not stop or alter any unrelated process. Clocks and power limits remain
  unmanaged and are recorded for every process.

## Frozen workload and exactness

- Same feature cache, seed, disjoint-clip sample, and shape as A0:
  `M=512`, `N=4096`, `D=1024`.
- For each target selectivity 1, 8, and 64, use the midpoint between consecutive
  FP64 squared-distance order statistics. Midpoints avoid placing the predicate
  boundary exactly on a measured pair while preserving the target output count.
- Oracle pair IDs are computed with NumPy FP64 from the stored FP32 vectors.
- Correctness requires exact equality of emitted linear pair-ID sets for both
  keeper and candidate at all three radii.

## Keeper and candidate

All representations are resident on the GPU before timing. Feature extraction,
quantization, index construction, disk I/O, and host-to-device transfer are
excluded.

**Keeper: exhaustive FP64 exact join**

1. PyTorch FP64 matrix multiplication for every query/base dot product.
2. FP64 norm epilogue and threshold predicate.
3. Materialize every emitted pair ID with `nonzero`.

**Candidate: certified INT8 join**

1. `torch._int_mm` over per-vector INT8 codes with INT32 accumulation.
2. FP64 dequantization/norm epilogue and triangle bounds for every pair.
   Add `1e-12` to the residual-radius sum as a conservative B0 numerical pad.
3. Materialize ambiguous pair IDs.
4. Recompute ambiguous squared distances directly in FP64 from original stored
   FP32 vectors promoted to FP64.
5. Materialize direct accepts plus refined accepts as emitted pair IDs.

The candidate includes all dynamic materialization and refinement overhead.
It does not claim bit-level outward-rounded GPU certification; exact equality
to the oracle is only a B0 validation gate.

## Diagnostic dense-compute controls

- INT8: `torch._int_mm`, INT8 inputs and INT32 output.
- FP32: `torch.mm` with TF32 disabled and float32 matmul precision `highest`.
- FP64: `torch.mm` with FP64 inputs and output.
- Shape and row-major logical operation are identical. These are kernel-stage
  controls and cannot be reported as end-to-end results.

## Timing and estimator

- CUDA-event timing; default stream; eager PyTorch; no CUDA Graph.
- 20 warmups and 100 retained observations per operation and radius.
- Eight fresh Python processes in order schedule:
  `A/B, B/A, A/B, B/A, A/B, B/A, A/B, B/A`, where A is keeper and B candidate.
- GPU-idle check immediately before each process under the one campaign lock.
- Preserve every raw latency.
- Primary estimator: geometric mean of the eight per-process median speedup
  ratios.
- Confidence: deterministic process bootstrap (seed 20260902, 20,000 resamples)
  over mean log speedup; report percentile 95% interval.
- Also report marginal p10/median/p90, all per-process ratios, process wins, and
  order split.

## Gate and next action

B0 passes directly to a fused CUDA implementation only if all conditions hold:

1. Keeper and candidate emit exactly the oracle pair IDs at every radius in
   every process.
2. Dense INT8 versus non-TF32 FP32 has a 95% lower speedup bound >=1.5x and at
   least 7/8 process wins.
3. Candidate versus exhaustive FP64 keeper has a 95% lower end-to-end speedup
   bound >=1.5x at every radius and at least 7/8 wins at every radius.

If the dense gate passes but the end-to-end gate fails, one fused-kernel design
is admissible only when the retained component timings attribute the miss to
framework launch/materialization overhead and an explicit latency floor still
permits >=1.5x. Otherwise stop this mechanism. No threshold may be weakened
after measurement.
