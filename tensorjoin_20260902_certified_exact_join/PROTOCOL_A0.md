# Protocol A0: ESC-50 Certificate Geometry

Experiment ID: `tensorjoin_20260902_int8_certificate_esc50_a0`

## Hypothesis

Per-vector symmetric INT8 quantization gives sufficiently tight deterministic
distance intervals that no more than 5% of real audio-window pairs require
original-precision refinement at useful radius-join selectivities.

## Frozen workload

- Dataset: ESC-50 at pinned commit
  `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
- Audio: native 44.1 kHz mono WAV.
- Window: 1.0 seconds; hop: 0.5 seconds.
- Feature: STFT with `nperseg=512`, `noverlap=256`, no boundary padding;
  `log1p(abs(STFT))`; deterministic mean pooling to `32 x 32`; flatten to
  `D=1024`; subtract per-window mean; L2 normalize.
- Data-quality rule: deterministically exclude a window before the seeded split
  if its centered feature has exactly zero L2 norm. Record every excluded clip
  and start sample. A non-finite feature remains a hard error.
- Split: RNG seed `20260902`; query and base windows come from disjoint source
  clips; `M=512`, `N=4096`.
- Distance oracle: squared Euclidean distance over the stored FP32 features,
  evaluated through FP64 arithmetic.
- Radius calibration: choose global squared-distance order statistics that
  target 1, 8, and 64 output pairs per query. This only calibrates selectivity;
  all ambiguity statistics use all `M*N` pairs.

## Frozen candidate

- Symmetric per-vector INT8 quantization:
  `scale=max(abs(v))/127`, `code=round(v/scale)`.
- Reconstructed coordinates are explicitly stored/treated as FP32 for A0.
- Residual norm is evaluated from the stored FP32 source and reconstruction.
- INT8 dot products accumulate in INT32. With `D=1024`, the largest possible
  magnitude is `1024*127*127=16,516,096`, below signed INT32 capacity.
- Pair intervals use the triangle-inequality formula in `GATE0.md`.
- A0 uses FP64 interval arithmetic plus a declared `1e-12` absolute containment
  guard. Bit-level outward rounding is deferred to the GPU correctness gate and
  is not claimed by A0.

## Metrics and gates

For each target selectivity, record:

- actual exact result count and results/query;
- direct-accept, direct-reject, and ambiguous pair counts;
- ambiguous fraction of all pairs;
- ambiguous-to-output amplification;
- lower/upper containment violations under the A0 guard;
- false accept, false reject, and final exact classification mismatch.

Pass only if:

- containment violations = 0;
- final classification mismatch = 0;
- ambiguous fraction <= 5% for every selectivity;
- median ambiguous fraction <= 1%; and
- the accumulator range gate passes.

The gate is immutable after the first valid full run. Crashes, corrupted input,
or contract mismatches are invalid runs and may be repaired without changing
the decision rule.

## Resource and safety boundary

- A0 is CPU-only; set `CUDA_VISIBLE_DEVICES` to the empty string.
- Do not start, stop, or modify any unrelated process.
- Do not download UCF101 or implement CUDA before A0 passes.
- Raw logs and source receipts must be retained.
