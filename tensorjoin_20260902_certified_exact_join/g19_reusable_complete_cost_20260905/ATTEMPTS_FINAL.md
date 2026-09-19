# G19 retained attempts and recovery boundaries

All artifacts below remain separate. No prior receipt was rewritten into a pass.

| Attempt | Observation | Disposition |
|---|---|---|
| Initial generation A0 | Exact replacement assertion rejected the wrong spelling of the G18 counter initializer; partial source only. | Preserved under artifacts/rejected_generation_a0; corrected before building or GPU work. |
| SSH setup | Two jump-host closures and one shell without usable input preceded the explicit PTY/multiplex session. | Transport-only recovery; no SSH or server configuration changes. |
| Admission A0 | Four matrices pass (18 + 18 + 9 + 9 exact calls), then RT full memcheck fails: invalid cudaFree on pinned-host memory and a 72-byte leak. | A0 stays failed. R1 copies OWL privately and changes exactly two PinnedHostMem release calls to cudaFreeHost. |
| Allocator diagnostic R1 | Same-process 72-byte pinned allocation: wrong cudaFree returns 1; correct cudaFreeHost returns 0. | Diagnostic A/B only; does not substitute for full operator memcheck. |
| Admission R1 | Repeated RT on/off matrices, full RT memcheck and synccheck pass. TC outputs pass but full shutdown memcheck retains five device-cache allocations plus one 4-byte pinned-cache allocation (425,721,860 bytes). | R1 stays failed. R2 introduces explicit post-engine device/pinned allocator cleanup, not timed-call cache clearing. |
| Admission R2 | TC full memcheck becomes zero-error/zero-leak. FP32 outputs pass, all Torch tracked allocations reach zero, but three borrowed-cuBLAS-handle allocations remain (8,520,704 bytes). | R2 stays failed. R3 creates/destroys an owned cuBLAS handle with the installed chosen workspace capacity. Never destroy a borrowed handle or reset the device to hide ownership. |
| R3 prelaunch source sync | A broad upload removed one trailing LF from two lagging local source copies. Immutable R1 hash gate stopped before any GPU job. | Restored exact recorded hashes without changing manifests; see SYNC_RECOVERY_R3.md. Subsequent uploads used exact files. |
| Admission R3 composite | Nine-fixture FP32 matrix repeats with new handle ownership, full FP32 memcheck passes, all 164 full stress calls and three NSYS profiles pass. | Fourteen composite required slots pass with explicit predecessor lineage. Unchanged RT/TC passes retained; old borrowed-handle FP32 matrix not used to admit R3. |
| Actual-code binding | Two diagnostic captures bind real TC/FP32 launches to compiled objects; fresh NCU exports the selected cuBLAS function. | Code identity is a separate gate, not profiler latency. Timing results live only in results/screen.json. |

The stress live-allocation check uses each engine's after-prepare baseline,
not an unconditional zero that would reject an intentional live cuBLAS workspace.
This change was declared in ADDENDUM_TORCH_SHUTDOWN_R2.md before any G19 stress.
After closing TC/FP32 engines, tracked device allocated/reserved bytes must be
zero. Full process-shutdown memcheck is also required and was not waived.

No sanitizer times, debug exports or occupancy-wrapper wall times enter the
public performance denominator. All latency observations and order are retained.
