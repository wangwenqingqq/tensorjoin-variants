# F1 Run Card

Host: `gpu-host-8` (verified SSH route)

Path: `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`

Branch/version: no Git repository; immutable source hashes recorded in each
result. The exact smoke source is frozen as
`artifacts/f1_smoke_source_058cc566.py` with SHA-256
`058cc566f5a95f4ec915b4755952891f398e87c07beea8199309de55d770b95a`.
The promotion source differs only by validation/stress/process-ID
orchestration and is frozen as `artifacts/f1_promotion_source_19fd8341.py`,
SHA-256
`19fd8341a61a72ede6c7f3f036f3ac7889a3c598b9c94dd78ec1457c4f32a8e6`.

Environment: `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`; PyTorch 2.11.0
with CUDA 13.0 and Triton 3.6.0 as verified by preceding D1--D3 runs.

GPU/port: physical GPU 0 only; no network service or port. Serialize with
`@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`.

Datasets:

- `data/esc50_panns_cnn14_1s_2048d.npz`
- `data/ucf101_r3d18_512d.npz`
- `data/indian_pines_patches_1984d.npz`

Model/checkpoint: feature extraction is complete and outside F1 timing.

Dimension: audio 2,048; video 512; HSI 1,984. Each uses 512 queries, 4,096
base vectors, and only the frozen nominal-64 midpoint radius.

Command: invoke `src/run_f1_gpu_cascade.py` once per modality with five
warmups, twenty observations, and predeclared orders `GTC`, `TCG`, `CGT`.
After a smoke pass, use validation-only memcheck, 1,000-launch stress, then the
eight formal orders and 20/100 timing contract in `PROTOCOL_F1.md`.

Log: `raw/f1_smoke.log`; results:
`results/f1_{audio,video,hsi}_smoke.json`. Promotion logs/results use
`raw/f1_{modality}_{memcheck,stress,process_*}.log` and matching result files.

PID: owned by the locked shell and recorded by the remote execution session.

Do-not-touch: all non-F1 processes and GPUs 1--7; clocks, power limits, and
shared environments.

Validation: exact pair IDs for all three variants, expected F0 FP64 counts,
then frozen smoke ratios in `PROTOCOL_F1.md`.

Rollback: stop only the owned F1 shell; preserve logs/results; source changes
are additive and do not alter D1/D2/D3 keepers.
