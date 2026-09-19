# Preserved failed/excluded attempts

No failed run is relabeled as passed. Numerical source/coefficients were not
changed after numerical observations.

| Target/run | Observation | Diagnosis / decision | Reopen condition |
|---|---|---|---|
| GPU2 ncu_half_0_a0 | Foreign PID548489 during quiescence; no child launched | Genuine shared-card occupancy; excluded | Fresh isolated admission |
| GPU2 ncu_half_3_a1 | Numerical trace passed; guard rejected PID552285 as `[No data]` after exit | Same PID in trace/occupancy; memory already14MiB; bounded NVML tombstone race supported | R1 verified-start-ticks guard; separate3_a2 passes |
| GPU2 gpu_fixture_a0 | Foreign PID558777 appeared after our child started | Real foreign process, not the tombstone exception; only own child group stopped | Separate target/isolated window; no numerical result from interrupted run promoted |
| GPU0 fixture_a1 | First panel output/interval passed; artifact serialization raised GPUTarget TypeError | Harness metadata serialization failure; source-level FP32 predicate unchanged | Explicit ControlR2 serializer; separate fixture_a2 passes twice |
| Offline NCU normalizer A0 | Cross-launch normalized hash mismatch | Only unrelocated BSSY reconvergence target differs | R1 normalizes that address class; old normalized files retained |
| Offline disassembly invocation | `cuobjdump` absent from shell PATH; zero-byte output files | Tool-path error, not a kernel failure | Explicit installed13.1 path writes separate `.r1.txt` outputs |
| GPU2 original_confirmation_a0 | Foreign PID571421 in quiescence; child PIDnull | Original-card confirmation not started, not a numerical or mechanism failure | Available original-card window; same frozen binaries and all unchanged gates |

Guard R1/R2 do not relax memory, duration, locks, or live foreign-process checks.
R1 accepts only a bounded missing-/proc tombstone for a previously ancestry-
verified PID, with original start ticks and an event log. Unknown or reused
PIDs remain rejected. Eight mocked identity cases are retained separately from
live isolation evidence. No other user's process was killed or reconfigured.
