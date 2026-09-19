# A2: full bounded gate on idle GPU 3 after repeated GPU 2 occupancy

Declared before any GPU 3 launch. A0 and A1 both stopped at signed31 preflight:
foreign Physion PIDs1788163 and1791960 occupied physical GPU 2 before any child
could start. Later live inspection found PID1794635 continuing that workload.
Keep both interrupted receipts/logs and the two earlier GPU 2 passes. Do not
kill or move foreign work, waive quiescence, or reinterpret either block as a
numerical failure. No performance observations exist to resample.

Resource-only change: physical GPU 3,
UUID GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2, same RTX PRO 6000 model, host,
driver590.48.01, SM120 binary, toolkit, shape and all mathematical contracts.
Fresh live preflight observed GPU 3 at14 MiB,0% utilization with no compute
process. Recheck through the unchanged guard; no guarantee of future idleness.
Use /tmp/tensorjoin_gpu3_campaign.lock and the G5 guard's physical-GPU3 lock.
Do not touch other GPUs, especially existing GPU0/1/2/7 tasks.

Run all20 planned correctness/safety/stress/profile slots afresh on GPU3 with
`_g3_a2` IDs, including the initial8192-pair probe and real31. This ensures one
complete same-device gate; retain rather than combine the earlier GPU2 passes
into its result. Original ordering, nine inputs, thresholds, conservative bounds,
1000 two-buffer predicate iterations,4 full safety runs and4 operator repeats
are unchanged. Exactly the same binary/library bytes are required.

Separate GPU3 wrappers change only the expected CUDA_VISIBLE_DEVICES assertion
and verify the additive freeze. Original wrappers and all original frozen files
remain unchanged. `frozen_gpu3_a2.json` binds new wrappers, orchestration,
addendum, old blocked logs/results and the original freeze before launch.
A further occupancy or correctness/safety failure stops this campaign; no
indefinite launch retry or silent hardware substitution is admitted.

Paper scope: 20 GPU3 observations, two separate historical GPU2 passes, and two
retained prelaunch blocks. Neither a cross-GPU timing result nor a universal
hardware/correctness guarantee. Native G17 sources/results remain immutable.
