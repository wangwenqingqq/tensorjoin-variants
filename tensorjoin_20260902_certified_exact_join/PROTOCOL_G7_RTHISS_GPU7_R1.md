# G7 runtime-smoke routing revision R1

Frozen: 2026-09-04 after the GPU1 admission check, before any GPU work.

The G7 a0 GPU1 runtime attempt did not launch a process: the final preflight
found foreign PID919685 using 23,364 MiB on the selected GPU. Preserve
results/g7_rthiss_smoke_a0.json as an admission skip, not an algorithm failure.
The subsequent full snapshot shows GPU0--4 occupied by foreign work and GPU5--7
without compute processes. No foreign process was stopped or modified.

R1 changes only the smoke-test physical device to GPU7, UUID
GPU-e5a0e7f5-9917-c2a1-ee8e-7282cbba2603, RTX PRO6000 Blackwell Server Edition.
Use @TENSORJOIN_ROOT@/.tensorjoin_gpu7_campaign.lock and the same strict
pre/postflight criteria. This replaces the original GPU4--7 do-not-touch list
only for freshly verified free GPU7; all other devices are excluded.

The binary remains 543229bbb6e3dbb174cb10d5dd81bc2388592c53938252d042ec7293d4f2e64b
and libowl remains df75c1bf36524381a0b9869c03e3d82e0bff4e62362a64300c5ff0d4419b1ffa.
Do not rebuild, retune, alter epsilon or output mode. Dataset, command options,
OMP_NUM_THREADS=8 and the 180-second cap remain unchanged. Upstream CUDA
refinement is compiled for compute_120/sm_120; upstream OWL emits embedded
PTX .version9.1/.target sm_75 for driver JIT. This distinction is retained.

This revision admits one count-only upstream sample smoke, not a high-D
correctness test or a timing campaign. If GPU7 is occupied at final preflight,
stop without launching and retain the skip; do not keep searching for cards.
