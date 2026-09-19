# Protocol: H2B-P3 Full-Operator Safety and Stability

Date: 2026-09-03

## Admission dependencies

P3 may run only if these immutable artifacts match:

- P1 correctness: `68f7e58713e4552ab43743b4a1b089a0b3df046d298f20b01f1ca6e889387e99`;
- P1 repeats: `6970b41c794ec787e880778f9a501de61331a7eb4cda5303d85130db4215fc27` /
  `75d22742a4a952aa19bfb8b2a53bd36508b45bc9211a71fe1e51cb84b4edda30`;
- P2 runtime/SASS audit:
  `c4145a0ea0c2f2f3e86bbe6e0d49d106c0ea8a6ac50401fce4c098d2eab94d77`;
- P1 runner/kernel: `d747b7c28e7fb1f196a45ce45ce027710cae338f876c973c8f98be4ca966eef1` /
  `33103f092b50241944f9bfcee0851fdab2a9578067ecc217ebaa706d132dfb43`;
- P3 runner: `414222764ce1e6bebc3fecc35c0e498ec95bfb759bb3a3c3e3547b435e979f86`.

The device remains physical GPU1 on `gpu-host-8`, protected by
`@TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock`.  Every process retains
the runner's foreign-PID refusal.  No external process may be stopped.

## Safety matrix

Run the complete H2B operator at target 8 on both full actual-data matrices:

- ESC-50/PANNs: 128x512 objects, 640x2560 tokens, D2048;
- UCF101/R3D-18: 128x512 objects, 662x2684 tokens, D512.

The instrumented path is:

```text
pedantic cublasGemmEx
-> fused FP64 interval reconstruction/object aggregation/compaction
-> charged device-count read
-> exact FP64 ambiguous-panel repair
-> final device-count read and sorted IDs
```

For each dataset retain one Compute Sanitizer `memcheck` and one `synccheck`
log with nonzero error exit codes.  A tool-reported error, process failure,
wrong output count/hash, ambiguity-count drift, duplicate, or overflow rejects
P3.  Sanitizer duration is diagnostic-only and may not enter P4 timing.

## Stability matrix

In one isolated uninstrumented process, allocate two independent device-state
buffers for each dataset.  After warming both buffers, retain 1,000 complete
operator invocations per dataset/target cell while alternating buffers.  Thus
the campaign retains 4,000 invocations across both target-1/8 cells and both
datasets.

Every invocation must reproduce the P1 exact ID hash and count, the P1
ambiguity count, zero duplicates/overflow, and one invariant full signature
per cell.  Preserve every iteration record in
`results/h2b_p3_stress.json`; do not replace it with aggregate-only evidence.

## Commands

All commands run under `flock -n` on the GPU1 campaign lock with
`CUDA_VISIBLE_DEVICES=1` and `NVIDIA_TF32_OVERRIDE=0`:

```text
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 86 \
  python src/run_h2b_p3_safety_stress.py --phase safety --dataset DATASET --target 8

compute-sanitizer --tool synccheck --error-exitcode 87 \
  python src/run_h2b_p3_safety_stress.py --phase safety --dataset DATASET --target 8

python src/run_h2b_p3_safety_stress.py --phase stress
```

Logs are append-only under `raw/h2b_p3_*`.  A deterministic post-run parser
must form `results/h2b_p3_safety_summary.json` from the four sanitizer logs
and frozen stress artifact.

## Evidence boundary and next gate

P3 can establish bounded tested safety and repeat stability only.  It cannot
establish absence of all GPU defects and supplies no latency evidence.  P4 is
blocked until all six P3 subgates pass and the postflight confirms the target
GPU is idle.

## Current state

Rejected on the first audio memcheck because full leak checking reported 12
outstanding framework/workspace allocations (153,224,192 bytes) at process
exit.  Exact output was produced and no out-of-bounds error was reported, but
the frozen any-error rule applies.  The remaining original subgates were not
launched.  A separate harness-only revision is frozen in
`PROTOCOL_H2B_P3_R1_SAFETY_STRESS.md`.
