# Gate1: frozen INT8 versus owned FP16 complete execution

Experiment tensorjoin_20260908_two_gate_g1_d512. Frozen before timed observations.
A is the original G16 four-cubin INT8/FP32/FP64 path. C is the admitted owned
FP16 cuBLAS plus GPU directed-FP32 classifier and same stage2/terminal. Neither
numeric source nor launch choices are changed here. Original-card nine-step
FP16 qualification and G16 qualification are prerequisites, not replaced by
this timing run. Both remain conditional fixed-FP64-reference implementations,
not universal exact-real claims. Legacy speed denominators are prohibited.

Original gpu-host-8 GPU2 UUID16f27f5a-dfcd-48e0-bb39-bebbe4009245, driver590.48.01,
SM120 server SKU, same installed libraries/source/cubins. N60000,D512,
T25921/65536, full frozen FP32 file and3,926,078-ID canonical output hash.
Eager default stream. Host-pageable vectors to sorted canonical host IDs:
all input-dependent allocation, preparation, H2D, distance/filter/queue work,
FP32/FP64 refinement, count/result D2H, scheduling and sorting included.
Exclude data-file IO, context/module/handle initialization, JIT warmup,
correctness hashing and JSON. No retained prepared vectors between calls;
allocator reuse allowed identically. Use the unchanged internal complete timer.
C's diagnostic-only key is read only after this independent admission/freeze.

Eight fresh processes, AC/CA alternating4times. In each process, for the first
method2full warm calls then9retained calls; repeat for the second method.
Then16consecutive full calls per method in the opposite method-block order.
Validate every output outside its timer. No timed progress prints. Retain all
432calls, both ordering groups and every failure. Hashing gaps mean repeat16
is repeated-call stress, not continuous device saturation or multi-hour proof.

Primary estimator: geometric mean of8process paired median ratios C/A.
95% order-stratified percentile bootstrap,10,000resamples,seed20260908.
Repeated-call estimator uses per-process sums of16latencies, same analysis.
Report all raw orders, process wins, arithmetic/geometric paired means,
marginal medians, p10/median/p90 and first/last4 repeat-call ratios.
Pass: all8processes valid; primary CI lower>1; >=7/8wins; both order groups>1;
repeat16 CI lower>1 and both groups>1. Otherwise inconclusive or negative
at this scope. A mean advantage<10% is labeled small even if statistical pass.
No replacements or outlier removal. First failed process stops the stage and
is retained; any repair needs a versioned new admission, never silent promotion.

GuardR2, two historical per-GPU locks,30second idle window,4GiB device-used cap,
30minute process timeout, verified child ancestry, bounded logged NVML exit
tombstones only. Never touch foreign users or clock/power settings. Live initial
foreign jobs occupy GPU3-6. No manuscript, Overleaf or publication changes.

Gate1 can establish speed only against this panel-materializing library control;
it cannot establish a novel INT8 mechanism. Gate2 must equalize fusion and
schedule and assess the strongest admitted fused FP16 control separately.
