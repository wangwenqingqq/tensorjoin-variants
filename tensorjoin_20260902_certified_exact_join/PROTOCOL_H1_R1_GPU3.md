# H1-R1 Revision: Physical GPU3

H1's original physical GPU1 target became unavailable before any admitted
correctness run because another user's eight-GPU process occupied it.  At
2026-09-03 09:08:46 GMT, physical GPUs 3 and 4 on the same `gpu-host-8` host
were empty (14 MiB, 0%, P8), while all other cards had live compute workloads.

H1-R1 changes only:

- physical device from GPU1 to GPU3;
- advisory lock to `@TENSORJOIN_ROOT@/.tensorjoin_gpu3_campaign.lock`;
- experiment/result/log names gain the `h1_r1` suffix.

Hardware model, host, software, data, score, thresholds, kernels, correctness
rules, memcheck scope, timing denominator, warmups, observations, and 1.25x
screen gate remain identical to `DESIGN_H1_MULTIVECTOR_GPU_SCREEN.md` and
`PROTOCOL_H1_MULTIVECTOR_GPU_SCREEN.md`.  The runner resolves GPU3's UUID and
refuses if a foreign compute PID appears before launch.  GPU3 must be rechecked
before every campaign phase; no process on another card may be stopped.

