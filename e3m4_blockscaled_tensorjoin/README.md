# Native block32 E3M4 in TensorJoin Stage 1

**The native scaled interface works in a nonuniform TensorJoin Gram matrix, but
this implementation does not replace INT8 or FP16.** On the frozen real fixture,
block32 E3M4 removes only 10 FP32-refinement pairs relative to per-row E3M4. It
beats block32 E4M3, but fails the predeclared full-cost and sustained gates against
both conventional controls. No production dispatch or paper novelty is claimed.

## Same-campaign real-data result

CIFAR fixture: **4096 x 512**, finite FP32 inputs, squared-distance threshold
**0.5686872086178483**, **262,144 identical directed canonical IDs** for every
method. Final measurements and gates use physical GPU 7, RTX PRO 6000 Blackwell
Server Edition / SM120, driver 590.48.01, CUDA 13.1.115, PyTorch 2.11.0+cu130 and
Triton 3.6.0. GPU ownership is supervised; shared-host CPU activity is not isolated.

Four fresh processes alternate forward/reverse method order. Each method has two
warmups and five retained observations per process. Compilation is excluded.
Wall time includes pageable H2D, allocation, quantization, block scales, FP64
metadata, Stage 1, host queue handling, FP32/FP64 refinement, D2H and sorted host
IDs. The following times are marginal medians over 20 observations; speed ratios
below use paired process medians, not a ratio of these marginal medians.

| Method | FP32 pairs | FP64 pairs | End-to-end median (ms) |
|---|---:|---:|---:|
| INT8, per row | 176,991 | 97 | 6.212 |
| FP16 control | 5,305 | 97 | 5.956 |
| E4M3, per row | 926,446 | 97 | 7.206 |
| E3M4, per row | 314,939 | 97 | 6.889 |
| E4M3, block32 | 926,446 | 97 | 7.799 |
| E3M4, block32 | 314,929 | 97 | 6.954 |

| Comparator / block32 E3M4 | Paired geometric ratio | 95% log-t interval | E3M4 process wins |
|---|---:|---:|---:|
| INT8 | 0.925 | [0.789, 1.083] | 0/4 |
| FP16 | 0.888 | [0.760, 1.037] | 0/4 |
| Block32 E4M3 | 1.129 | [1.103, 1.155] | 4/4 |
| Per-row E3M4 | 0.996 | [0.977, 1.016] | 1/4 |

A ratio above one favors the candidate. Intervals use four process log ratios
and Student-t with three degrees of freedom. Variability is material: INT8 /
block32 E3M4 is 0.978 for forward-order processes and 0.874 for reverse-order
processes. All observations are retained. The short-run intervals against INT8
and FP16 cross one; do not claim a statistically resolved slowdown from that
screen alone. It nevertheless fails the frozen >1.10x-in-every-process gate.

### Does a smaller refinement queue pay off?

On the synthetic outlier fixture (1024 x 512, threshold 0.1024), FP32 pairs fall
from **32,483 (INT8)** to **27,714 (per-row E3M4)** to **26,980 (block32 E3M4)**.
The final FP64 count remains **25**. Full times are **1.303 / 1.310 / 1.319 ms**,
respectively; paired INT8 / block32 E3M4 is **0.981 [0.949, 1.013]**. Thus the
16.9% reduction versus INT8 is a positive queue-size result, not a speedup.

On the clustered control, block32 E3M4 increases the INT8 FP32 queue from
12,079 to 19,980, with 15 FP64 pairs. Paired INT8 / candidate is
0.986 [0.933, 1.041]. Neither synthetic control rescues the real-data decision.

Power-of-two block scales can help range/underflow without adding mantissa bits.
That is a plausible explanation for the small extra queue benefit over per-row
E3M4, not a measured causal decomposition of each residual source. The observed
block32 E3M4-over-E4M3 gain is a format comparison at the same granularity, not
proof that block scaling itself helps. No profiler-based bottleneck claim is made.

### Bounded consecutive-call check

Each fresh process also runs 32 consecutive complete CIFAR calls per method:
**768 calls** across six methods and four processes. Individual-call medians are
5.650 ms (INT8), 5.445 ms (FP16), 5.841 ms (per-row E3M4) and 5.870 ms (block32
E3M4). Paired INT8 / block32 E3M4 is 0.996 [0.881, 1.127]; FP16 / block32 E3M4
is 0.926 [0.922, 0.930]. Non-regression fails. This is a bounded screen, not
production uptime qualification.

The additional enclosing batch denominator includes outside-run compiled-code
registry hashing and exact-ID assertions; individual-call times exclude those
checks. For block32 E3M4 the enclosing mean is 15.678–16.232 ms per call. This
instrumented denominator also fails the declared non-regression gate and must
not be confused with operator-only service latency. Phase CUDA events include
host submission gaps; they are attribution, not isolated kernel throughput.

## Interface and numerical contract

This extends the [previous instruction survey](../e3m4_interface_probe/README.md)
from uniform instruction inputs to a nonuniform tiled Gram matrix:

- One UE8M0 scale per row/block of 32; Q is uint8 N x 512, S is uint8 N x 16.
  The B value operand is transposed; its scale array is not transposed.
- Triton `dot_scaled` first compiles the documented E4M3/E4M3
  `QMMA.SF.16832.F32` family with E8 scales. Checked private cubin copies change
  only absolute instruction bits 82 and 84 to select E3M4/E3M4. No FP16 dot
  emulation and no guessed plain-to-scaled enable toggle are used.
- Tiles are 32 x 32 x 64 with four warps and two pipeline stages. Final native
  scaled-dot specializations report zero spills; resource records include the
  actual shapes and constexprs. No kernel tuning or new dynamic router is added.
- Reconstruct z using each element's actual block scale. Compute outward original
  norm intervals and upper bounds on ||z|| and ||x-z||. The retained certificate
  depends on these reconstruction bounds, not on scale granularity, so its
  classifier and FP32/FP64 cascade are reused unchanged. See the
  [conditional certificate](../precision_format_routing/docs/CERTIFICATE.md)
  and [frozen scale-floor/rounding contract](CONTRACT.md).
- The floating-dot envelope remains conditional, gamma=0.00012232370499987155.
  Empirical interval containment and finite-FP64 ID equality are not a universal
  hardware guarantee or proof of exact-real arithmetic.

E3M4 is an undocumented SM120 execution path, not an official CUDA dtype or a
portable Blackwell API. The public blackwell-isa/cubit work is prior art; see
[pinned provenance](../e3m4_interface_probe/README.md#prior-art-and-tensorjoin-boundary).
Neither selector discovery nor generic precision routing is a new contribution.

## Validation and preserved evidence

- Both FP8 codebooks: 256 codes on each operand axis, 65,536 pair outputs per
  format; raw 0x10 self-dot at K64 is 4.0 for E3M4 versus 0.0625 for E4M3.
- Fresh E4M3/E5M2 A/B compiler controls, plus 20 nonuniform tests over
  N=31/32/33/97/129 and K=64/512 with mixed scales and signed codes.
- Original/reconstructed/residual metadata bounds and complete distance interval
  containment at N=97; 1,010 midpoint/neighbor/sign values per format;
  eight dot Graph replays and two non-default streams per format.
- All six methods match complete sorted IDs on all 14 fixtures; small fixtures
  additionally match independent CPU FP64, including adjacent binary64 thresholds.
- Eight final sanitizer runs: memcheck, racecheck, initcheck and synccheck on
  CIFAR prefix512 and boundary32. Zero errors; racecheck has zero hazards/warnings.
  `readelf` emits a vendor-section info-field warning, separate from sanitizer errors.
- Stress: 1,344 complete candidate cascades plus 14 references, at most N=512,
  with 32 distinct GPU input addresses. Full cascades are not Graph captured.
- GPU 0/1 diagnostic runs and the blocked GPU 0 preflight are preserved. Other
  users' processes were never stopped. All final gates were repeated on GPU 7
  after the diagnostic compiled-registry fix; final source hashes match timing.

[SUMMARY.json](results/SUMMARY.json) retains quantiles, process medians, ratios,
intervals, queue counts and phase attribution. Raw `.jsonl` files retain order,
warmups, compiler preparation, main samples and consecutive-call samples.
[EVIDENCE_MANIFEST.json](results/EVIDENCE_MANIFEST.json) hashes the evidence;
[GPU_SUPERVISION.json](results/GPU_SUPERVISION.json) retains numeric telemetry
without process identities. Compiled manifests retain observed Triton variants,
not an inventory of PyTorch/library helper kernels.

Full cubins contain private build paths in debug sections and stay on the
source machine/private evidence copy. Fourteen patch pairs passed an independent
full-container difference check before publication; the
[report](results/BINARY_CONTAINER_AUDIT.json) preserves container hashes. Published
`.text` sections, offsets, sizes, hashes, baseline SASS and instruction patch
ledgers permit CPU-only rechecking of code-section differences. That offline
check does not independently reconstruct the withheld full containers.

## Reproduction and decision

These are matched research prototypes, not the optimized legacy INT8/cuBLAS FP16
operators. No full 60K dataset, strongest external baseline, universal error proof
or product admission is claimed. Keep INT8 and FP16 baselines; do not promote
this block32 E3M4 implementation. Reopen only for a specific residual-reduction
mechanism or a tighter defensible certificate that survives a fully charged
comparison against both baselines. More format choices alone are not enough.

From the repository root with the stated environment and an explicitly idle GPU:

```sh
export TENSORJOIN_SOURCE=/path/to/authorized/certified-exact-join
RUN="$PWD/e3m4_blockscaled_tensorjoin/src/run.py"
GUARD=precision_format_routing/src/guard.py
python "$GUARD" --gpu 7 --label mx_replay_probe -- python "$RUN" --mode probe --label replay_probe
python "$GUARD" --gpu 7 --label mx_replay_validate -- python "$RUN" --mode validate --label replay_validate
# Labels are immutable; use new names instead of overwriting evidence.
for tool in memcheck racecheck initcheck synccheck; do
  for case in cifar4096 boundary_zero32; do
    label="replay_${tool}_${case}"
    python "$GUARD" --gpu 7 --label "mx_$label" -- \
      compute-sanitizer --tool "$tool" --error-exitcode 3 \
      python "$RUN" --mode validate --label "$label" --case "$case" --limit-n 512
  done
done
python "$GUARD" --gpu 7 --label mx_replay_stress -- python "$RUN" --mode stress --label replay_stress
for order in 0 1 2 3; do
  python "$GUARD" --gpu 7 --label "mx_replay_bench_$order" -- \
    python "$RUN" --mode bench --label "replay_bench_$order" --order "$order"
done
# These commands check the bundled campaign, not the new replay labels:
python e3m4_blockscaled_tensorjoin/summarize.py
python e3m4_blockscaled_tensorjoin/verify_evidence.py
python -m unittest discover -s e3m4_blockscaled_tensorjoin -p test_summary.py
```

The fixture loader checks source-file hashes; restricted raw input data is not
redistributed. The last three commands require only Python and NumPy and replay
the published statistical/evidence checks, not GPU execution.
