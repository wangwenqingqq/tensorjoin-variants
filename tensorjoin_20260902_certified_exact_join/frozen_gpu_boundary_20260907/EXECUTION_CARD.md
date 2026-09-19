# Resolved execution and design card

- Host: local Mac -> configured tiaoban -> gpu-host-8, live hostname gpu-host-8.
- Workspace/account: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join, root.
- Version: no Git repository; frozen_manifest.json binds 4 original code objects and associated IR/metadata; original_inputs.json protects 278 local files.
- Device: physical GPU3, GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2, RTX PRO 6000 Blackwell Server Edition, SM120.
- GPU2 was occupied by another user's SGLang. GPU1/2/7 workloads remain untouched. GPU3 selected from idle inventory; require 30 s quiescence and fail closed on new foreign occupancy.
- Locks: /tmp/tensorjoin_gpu3_campaign.lock and /tmp/tensorjoin_g5_gpu3.lock; nonblocking acquisition.
- Environment: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python, Python 3.12.3, torch 2.11.0+cu130, Triton 3.6.0, NumPy 2.3.1, driver 590.48.01; installed nvcc 13.1.115 is NOT used to rebuild binaries. Sanitizer 2025.4.1.
- Observed GPU settings: P0, SM2370 MHz, memory12481 MHz, power limit600 W, ~90 W idle. No settings changed.
- Workload: N60000 D512 FP32 |x|<=1, T=25921/65536, epsilon=161/256. Synthetic selected tiles then retained public full input.
- New launcher: ctypes CUDA Driver API; torch owns context, buffers and stream. Modules loaded from .cubin files only. ABI verified against PTX and runtime parameter offset/size query. Scratch pointers zero only because both scratch sizes are zero. Official API reference: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html (checked 2026-09-07).
- CUDA work unchanged: block128 / 4 warps / one CTA; dynamic shared16384,16,32,32 bytes for stage1,stage2,terminal,metadata. Existing MMA and reduction ownership/lifetimes/barriers retained, not redesigned. Extra host orchestration, direct-terminal comparisons and CPU rational audits are diagnostic costs, not optimizations.
- Live sets: vectors+metadata persist; three capacity16777216 queues persist; stage counters reset per batch; direct oracle and graph reuse are serialized. Stage1 completes before host reads count; stage2 completes before host reads fallback count; terminal completes before CPU canonicalization. No buffer overwritten before its consumer completes.
- Pointer stress keeps an old allocation alive while creating the next, then varies queue order. Graph replay has immutable inputs and calibrated counts; it is not arbitrary-input graph dispatch validation.
- Resource bound: target <4 GiB process GPU memory; fail before intentionally invalid tile/queue/row workloads. Metadata row2^22 wrap is host-rejection-only.
- Commands/PID: guard creates exclusive raw/<label>.log, raw/<label>_occupancy.jsonl and results/<label>_guard.json, recording exact child command/PID and source hashes. Timeout30 min; terminate only this campaign's child process group.
- No latency denominator: this is correctness validation. No baseline or performance result promoted.
- Preflight bookkeeping: SSH a0 completed and its output was retained; local zsh failed afterward assigning reserved variable `status`. A second shell attempt failed at `echo ====` before executing reads. Both are local harness-command failures, not GPU tests. a1 confirms live environment.
