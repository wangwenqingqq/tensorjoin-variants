# G18 decision: bounded repaired RT-HiSS reference contract admitted

Verified 2026-09-05 on gpu-host-8, RTX PRO 6000 physical GPU3. This closes a
correctness/correction-work gate, not a public performance or novelty campaign.

## Conclusion, strongest evidence, caveat and next action

The separately implemented two-sided RT-HiSS repair returns the complete frozen
FP64-reference output on all nine D512 inputs. On CIFAR4096 it restores all
262,144 directed IDs with zero missing or extra pairs, correcting both native
error directions retained by G17. Only **184 of16,777,216 candidates** require
FP64 terminal evaluation: **0.0010967255%**. This passes the predeclared real-data
fallback-work screen (<1%), without changing RT candidates or converting all
refinement to FP64.

The strongest caveat is that this is not yet a speed result: counters, raw-mask
capture, host PID-attribution delay and evidence IO remain enabled. The synthetic
boundary_zero32 case needs62/1024 terminals (6.0546875%), so low fallback is not
universal. RT candidate coverage is proven only by exhaustive checks on this
finite matrix; no general conservative RT traversal or exact-real predicate is
established. Novelty remains unapproved.

Next build and validate a separately frozen diagnostic-free *complete-cost*
comparison artifact. Resolve reusable-engine versus fresh-invocation lifecycle
fairness first, then remeasure RT-HiSS, the G16 path and the complete FP32-first
control on the same shape/reference. Do not use these instrumented times or
compare4096 to G16's old60K observation. See NEXT_COST_CONTRACT.md.

## Complete observed work, not latency

| Frozen input | Correct directed IDs | Safe FP32 reject | Safe FP32 accept | FP64 accept / reject | Terminal fraction |
|---|---:|---:|---:|---:|---:|
| real31 |31|930|31|0 /0|0%|
| signed31 |31|930|31|0 /0|0%|
| shuffled_onehot31 |97|864|97|0 /0|0%|
| zeros31 |961|0|961|0 /0|0%|
| boundary31 |43|918|43|0 /0|0%|
| cancellation31 |961|0|961|0 /0|0%|
| subnormal31 |961|0|961|0 /0|0%|
| cifar4096 |262,144|16,514,984|262,048|96 /88|0.0010967255%|
| boundary_zero32 |104|918|44|60 /2|6.0546875%|

Every row has zero missing/extra IDs and complete candidate coverage. The real
anchor's output hash is unchanged from the original G2A oracle:
`da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`.
All five real4096 operator observations (correctness, two sanitizers and two
fresh-process repeats) reproduce this hash and every recorded work count.

The real operator executes2,659,208,992 FP32 coordinate updates and94,208 FP64
coordinate updates. The latter equals184*512. Prefix rejection is still active;
these counts do not measure the incremental cost versus native RT-HiSS, a
throughput advantage, or a whole-system speedup. All16,777,216 real pairs remain
RT candidates here: the experiment demonstrates no RT pruning benefit on this
anchor and does not generalize that observation to all workloads.

## Mechanism and numerical boundary

The native default shared/shared pipeline, data ordering algorithms, RT geometry,
mask/compression kernels and original-ID decoder remain unchanged. Only the
per-candidate predicate changes, with read-only inverse-dimension/cutoff state
and diagnostic counters. G17's native source/binary/output remain preserved.

NUMERICAL_DESIGN.md derives FP32 prefix and FP64-reference error envelopes from
the fixed finite domain and explicit RN arithmetic with gradual underflow.
Cutoffs are evaluated as exact rational expressions and rounded outward to FP32.
Reject from a prefix only above the upper cutoff; accept only a completed sum
below the lower cutoff; otherwise evaluate original-order sequential FP64 RN
subtraction, separate multiplication and addition. Positive-only reranking would
not recover the native false negatives and is not used.

The same-launch8192-pair probe agrees bitwise with the separately evaluated CPU
FP64 scores and agrees on every final decision. Its four retained real disputes
route to terminal accept, accept, reject, reject as required. Exact-rational
checks enclose515 sampled observed prefixes;365 CPU cutoff tests pass. The
1000-invocation, alternating-buffer/stride1-and3 predicate stress reproduces the
entire output record. This is predicate stress, not sustained full-join evidence.

## Admission gates and retained exceptions

The main GPU3 gate has20 admitted slots:2 predicate observations (A/B and stress)
and18 complete operator observations spanning9 inputs. Four operator sanitizer
runs (memcheck and synccheck on boundary_zero32 and CIFAR4096) report zero errors.
The two additional operator repeat orders allocate fresh process state. Leak
checking was disabled; no Graph, arbitrary stream or leak-free shutdown claim.

Selected-code audit observes the default repaired CUDA function at runtime:
40 registers/thread versus native36,1024-thread blocks and47,104 bytes dynamic
shared memory. Static stack/local resources and selected LDL/STL counts are zero.
Refinement changes from336 to992 static instructions and contains explicit
F32 and separate F64 arithmetic; predicate FADD/FFMA have no FTZ modifier. The
408-instruction compression stream remains byte-identical after normalization.
No latency or occupancy improvement follows from those observations. OptiX
runtime-generated code identity remains outside the audit.

Failed and interrupted observations were NOT erased:

1. The first build used the wrong CMake cache key and compiled D18. Inspection
   caught it before GPU launch. `build_a0` and its logs remain rejected;
   `build_r1` is explicitly D512, separately frozen and used throughout.
2. Two GPU2 signed31 attempts stopped during preflight due to foreign Physion
   occupancy. Two earlier GPU2 passes are retained separately, not combined
   into the complete GPU3 gate. No foreign process was killed or moved.
3. A2 zeros31 produced all961 correct IDs but failed an overstrict cross-run
   point-map byte-identity proxy. Its failed combined gate remains failed.
   Two unchanged-native controls confirm variable valid point permutations;
   all zeros still have identical reordered value bytes. A3 explicitly replaces
   the invalid proxy with bitwise actual input/map validation, bijection,
   unchanged dimension order, independent original-ID decoding, full oracle
   equality and candidate/work identity. No numerical/output requirement changes.
   Cancellation/subnormal inputs also have valid point permutations with different
   per-position value bytes, showing why public IDs, not a snapshot permutation,
   are the contract. See ADDENDUM_MAP_EQUIVALENCE_A3.md.
4. Native diagnostic d0 returned correct data but had an unattributed retiring
   PID in its guard. Isolation was not admitted. Two new controls keep their
   exact completed child unreaped briefly for PID attribution and pass the
   unchanged guard; d0 stays unadmitted. No safety waiver or timing result.
5. Jump-host deployment/read interruptions are retained as transport events,
   not failed numeric or performance samples. A task-owned foreground SSH
   multiplex connection was used for final transport; no global SSH changes.

Thus there are20 main admitted GPU3 observations,2 separate historical GPU2
passes,2 admitted native map diagnostics,2 retained failed guard observations,
and2 prelaunch occupancy blocks. A3 retains the first four already valid GPU3
slots and runs only the pending slots under the corrected validator. It does
not relabel A0/A1/A2 campaign receipts as complete successes.

## Three decisions and allowed paper impact

| Level | Decision | Scope / reopen boundary |
|---|---|---|
| Implementation | measured, bounded repaired reference contract admitted | Nine inputs, default shared/shared pipeline, frozen binary and declared FP64 algorithm; full same-output performance and reusable lifecycle remain pending. |
| Mechanism | measured selective correction work on the real anchor |184 terminals, both error directions fixed, <1% real-data screen passed; boundary fixture6.0546875%, whole-cost advantage unknown. |
| Thesis | unknown, not promoted | Stronger comparator fairness is an experimental asset, not a new adaptive-filter principle. G10 prior-art overlap and G16's narrow engineering scope remain. |

G16's1.499724/1.267190 complete-cost early-screen ratios are preserved but not
rerun or generalized. Do not claim TensorJoin beats repaired RT-HiSS yet. Do
not discard the baseline if it later wins, replace it by an FP64-all-candidate
strawman, or rescue a subsumed thesis with packaging/data breadth.

## Evidence and preservation

Start from `results/closure_checks.json`, `campaign_gpu3_a3.json`, every
`case_*.json`, both predicate reports and `code_audit.json`. Original stopped
campaign receipts, native diagnostics, frozen source/input maps, raw masks,
canonical pairs, guard/preflight/occupancy records and selected binaries are
retained. `artifacts/raw_evidence_manifest.json` is the final inventory.

Closure verified all420 G17 evidence files plus42 original core/manuscript files
against their prior hashes. GPU3 had no compute process at recorded postflight.
Other jobs were untouched. No manuscript was edited, compiled, visually checked,
synchronized, published or submitted by G18; G13 remains LOCAL ONLY.
