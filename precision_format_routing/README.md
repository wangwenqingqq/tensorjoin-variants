# Native precision-format routing: bounded 8P screen

**Conclusion: native E3M4 works, but replacing INT8 did not produce an end-to-end win in this screen.** On the real CIFAR fixture E3M4 increased the FP32 queue. A synthetic outlier fixture reduced that queue, without a statistically resolved latency benefit. The fully charged global selector was much slower than a fixed format. No production path is promoted.

## Main real-data result

4096 x 512 FP32 rows; G17 frozen CIFAR fixture, squared-distance threshold **0.5686872086178483**. This is not the 60K campaign's threshold. All methods returned **262,144 identical directed canonical IDs**. Four fresh processes, forward/reverse method order, two untimed warmups and five measured repeats per method per process.

The following wall times are medians over 20 measurements, from pageable host input through transfer, allocation, preparation, full cascade, D2H and host canonicalization. These are matched prototype implementations sharing a certificate framework, not a comparison of the legacy optimized INT8 and cuBLAS FP16 operators.

| Method | FP32-refinement pairs | FP64 pairs | End-to-end median (ms) | Process-paired INT8 / method, 95% log-t interval |
|---|---:|---:|---:|---:|
| INT8 | 176,991 | 97 | 5.597 | 1.000 [1.000, 1.000] |
| E3M4 | 314,939 | 97 | 5.788 | 0.967 [0.963, 0.971] |
| E4M3 | 926,446 | 97 | 6.674 | 0.840 [0.835, 0.845] |
| E5M2 | 3,239,462 | 97 | 9.880 | 0.567 [0.562, 0.572] |
| FP16 (subnormal flush) | 5,305 | 97 | 5.418 | 1.030 [1.021, 1.038] |
| FP32 IEEE dot (input subnormal flush) | 1,174 | 97 | 6.713 | 0.837 [0.834, 0.841] |
| Direct FP64 | 0 | 8390656 (all upper pairs) | 57.666 | 0.097 [0.096, 0.098] |
| Charged global router (selected FP16) | 5,305 | 97 | 41.382 | 0.135 [0.135, 0.136] |

- E3M4 increased the FP32 queue by **77.9%**. Its terminal FP64 queue stayed at **97**, as it did for every fixed-format cascade.
- FP16 greatly reduced the FP32 queue, but its approximately 3% full-cost advantage did not meet the predeclared >1.10x-in-every-process screen gate.
- The cold global router selected FP16 in all 20 real-fixture observations. Charging sample conversion, calibration, graph setup, selection and the chosen full run removed any benefit. This rejects this cold selector implementation, not all amortized or per-tile routing policies.

## Positive mechanism / negative end-to-end control

On the explicitly synthetic outlier fixture (1024 x 512, threshold 0.1024), E3M4 reduced FP32 candidates from **32,483 to 27,714 (14.7%)**. Both retained **25** terminal FP64 pairs. INT8/E3M4 median full times were **1.265/1.271 ms**. The paired ratio was **0.994**, with 95% interval **[0.975, 1.012]**: no resolved speedup.

The clustered synthetic control also did not favor E3M4. Small N=31/32 correctness fixtures are launch-bound: direct FP64 passed the narrow speed screen there, and FP16 did so on real31. These small-shape wins are retained in the full summary, not generalized to 1024/4096 or hidden by an aggregate.

## Verification and scope

- GPU: RTX PRO 6000 Blackwell Server Edition, SM120, driver 590.48.01; selected physical GPU 1. Other GPUs were left untouched. No clock/power changes. Triton 3.6.0 / PyTorch 2.11.0+cu130; actual ptxas pinned and asserted to 13.1.
- E3M4 is executed by a checked private cubin patch, not decoded into FP16/FP32 for GEMM. All 256 codes on both operands, documented E4M3/E5M2 controls, anti-fallback values, ties-even producer boundaries, and high-entropy D512 interval checks passed.
- All eight methods passed exact full-ID equality on 14 fixtures; small cases additionally passed independent CPU FP64 checks. Adversarial binary64 threshold below/equal/above cases returned 481/961/961 IDs.
- Nine final sanitizer runs: memcheck/racecheck/initcheck/synccheck on CIFAR prefix512 and boundary32, plus memcheck on repeated pointer-churn boundary32. Zero errors; racecheck also reported zero warnings/hazards.
- Final stress: 16 full cascades per method per fixture (1,792 tested calls plus 14 direct references), with larger fixtures limited to 512 rows. 440 distinct GPU input addresses were observed. Fixed calibration components were Graph replayed; the complete CPU-directed cascade was not Graph captured.
- Native dot specializations have zero reported spills. Per-run compiled manifests retain all dot shapes, but only the last classifier/refinement specialization for each dictionary key; they are NOT a complete shape/threshold execution-to-binary map. The supplemental cache inventory includes exploratory and final kernels and is not such a map either.
- The certificate still assumes a stated conservative floating-dot envelope. Empirical finite-FP64 ID agreement is not a proof of exact-real correctness or a universal hardware guarantee.
- No full 60K run, tuned external-baseline comparison, sustained production workload qualification or production promotion is claimed. The largest tested shape is 4096 x 512. The full-cost measurements also include shared-host CPU behavior; GPU ownership was supervised, not system-wide CPU isolation.

## Decision and reopening condition

Keep INT8 and the conventional FP16 control as distinct baselines. Do not promote this per-row E3M4 path or this cold global router. Reopen only with a specific distribution/layout that measurably reduces certificate ambiguity, a tighter defensible format-specific certificate or an amortized low-cost selector, and a full-cost win over the strongest fixed format—not just INT8. Generic format switching and E3M4 discovery are prior art; this screen establishes no paper novelty.

## Evidence and reproduction

- [Follow-up: callable E3M4 instruction survey](../e3m4_interface_probe/README.md) independently tests plain and block-scaled format combinations. It is not a new TensorJoin performance result.
- [Frozen contract and pre-timing repairs](docs/CONTRACT.md)
- [Certificate and format identity](docs/CERTIFICATE.md)
- [Full 14-fixture summary](results/SUMMARY.json): p10/p50/p90, all process medians, paired ratios/intervals and per-shape screen gates.
- [Validation manifest](results/VALIDATION_MANIFEST.json), [selected-GPU telemetry](results/GPU_OCCUPANCY.json), and [raw final logs](results/logs/).
- `results/v1b_bench_0.json` through `v1b_bench_3.json` contain every warmup, compiler-preparation and measured observation. Compiler preparation and warmups are excluded from summaries.
- `results/diagnostic/` retains earlier experiments, including the bundled-ptxas 12.9 mistake and the pre-repair scalar-threshold ABI. Their timings/old passes do not qualify the final implementation.

Run from this directory with a compatible CUDA Python environment and an explicitly idle GPU. The source dataset is not redistributed: set `TENSORJOIN_SOURCE` to the authorized certified-exact-join source tree containing its G17 fixtures. The loader verifies each frozen file hash. Synthetic fixtures are generated by the checked-in fixed seed.

```sh
export TENSORJOIN_SOURCE=/path/to/authorized/certified-exact-join
python src/guard.py --gpu 1 --label replay_probe -- python src/experiment.py --mode probe --label replay_probe
python src/guard.py --gpu 1 --label replay_validate -- python src/experiment.py --mode validate --label replay_validate
# Repeat sanitizer/stress gates before any new timing campaign.
python src/guard.py --gpu 1 --label replay_stress -- python src/experiment.py --mode stress --label replay_stress
python src/summarize.py  # rebuild the bundled v1b four-process summary
```

The guard requires Linux / nvidia-smi / CUDA 13.1 at its declared standard path. A replay label must be new. Sanitizers, compilation and disk I/O are not public latency samples. Original TensorJoin directories are never modified. Existing rights/notices remain in force; this private experiment does not relicense dependencies or authorize source-data redistribution.
