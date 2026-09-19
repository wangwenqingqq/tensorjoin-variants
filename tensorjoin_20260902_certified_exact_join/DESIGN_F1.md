# F1 Design Card: High-Output Precision Router

Experiment ID: `tensorjoin_20260903_gpu_cascade_f1`

## Target and hypothesis

- Hardware: physical GPU 0 on `gpu-host-8`, RTX PRO 6000 Blackwell Server
  Edition, compute capability 12.0. The existing Triton path lowers signed INT8
  dot products to warp-level IMMA; SM120 has no `tcgen05`/TMEM dependency.
- Hypothesis: at the frozen 64-result/query points, an FP32 filter over only
  INT8-ambiguous IDs removes enough scalar FP64 work to improve the existing
  two-stage candidate despite one extra kernel and host-visible count.
- This is a high-output dispatch candidate, not a universal replacement. F0
  already rejected the fixed cascade at the tie-expanded audio low radius.

## Numerical and output contract

- Inputs are resident FP32 vectors; INT8 codes, per-vector scales, reconstructed
  norms, and residual-norm bounds are prebuilt and excluded from timing exactly
  as in D1/D2B/D3B.
- Stage 1 is the unchanged INT8 certificate and `1e-4` pad.
- Stage 2 recomputes direct-difference FP32 squared distance for each ambiguous
  pair. Bitwise-identical FP32 vectors are accepted as exact zero; otherwise the
  unchanged `1e-3` guard makes a direct accept/reject or emits an FP64 ID.
- Stage 3 recomputes direct-difference FP64 distance only for emitted IDs.
- Output is the complete pair-ID set under the same tie-aware FP64 midpoint
  radius. Zero false direct decisions, mismatches, duplicates, and overflows are
  required.

## Shape and dispatch boundary

| Modality | Q x N x D | Radius | Expected INT8 ambiguity | Expected FP64 IDs |
|---|---:|---:|---:|---:|
| PANNs audio | 512 x 4096 x 2048 | nominal 64, actual 64.020/query | 7,007 | 366 |
| R3D-18 video | 512 x 4096 x 512 | 64/query | 27,283 | 1,061 |
| Indian Pines HSI | 512 x 4096 x 1984 | 64/query | 11,565 | 928 |

No low-radius cell is admitted in F1. No shape, guard, tile, or capacity is
tuned by modality.

## Ownership and phase-live set

| Phase | Owner | Live state | Last use |
|---|---|---|---|
| INT8 scan | 64x64 Triton program, 4 warps | INT32 accumulator, scales, norms, bounds | status classification |
| FP32 filter | one Triton program per ambiguous pair, 4 warps | one FP32 reduction and equality count over 256-D slices | guarded decision/FP64-ID append |
| FP64 verify | one Triton program per residual pair, 4 warps | one FP64 reduction over 256-D slices | final accept append |
| output | atomic appenders | result, ambiguity, overflow, and FP64 counters | host validation |

There is no inter-CTA ready edge or shared-memory handoff. Kernel completion and
the existing host-visible count delimit each phase. Large live objects are the
per-program reduction slice only; no accumulator persists across kernels.

## Work and movement hypothesis

- Invariant: one dense INT8 scan over 2,097,152 pairs and the same exact output.
- Added: one FP32 direct-distance evaluation for every INT8-ambiguous pair, one
  FP64-ID buffer, one kernel launch, and one count synchronization.
- Removed: 91.98%--96.11% of FP64 vector-distance evaluations at the three F0
  target-64 cells.
- The predicted floor is dominated by the dense scan plus FP32 gather traffic;
  the test is falsified if irregular gather and the extra synchronization erase
  the FP64 reduction.

## Keeper, comparators, and verification ladder

- Primary mechanism keeper: unchanged INT8 scan followed by FP64 verification
  of every ambiguous pair, remeasured in the same process.
- External performance comparator inside this prototype: the existing guarded
  FP32 exact join, also remeasured in the same process.
- Ladder: compile -> direct-decision/final-ID correctness for all three cells ->
  one-process CUDA-event smoke -> only if admitted, memcheck/stress/SASS and
  eight-process direction-balanced timing.
- No graph, end-to-end ingest, multi-GPU, or external specialized join claim is
  included.
