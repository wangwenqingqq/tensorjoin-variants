# A3: correct a byte-identity proxy without weakening pair correctness

Declared after A2 stopped at zeros31 and before a corrected-validator run.
The program returned0, all961 original-ID pairs matched the frozen oracle,
independent bitmap/permutation validation passed, and all961 candidates were
covered. The extra cross-run `point_map.u32` byte-equality assertion failed.
Keep that failed combined gate and its raw maps/results unchanged.

A map describes a run's original-ID permutation; it is not itself the public
answer. In zeros31 all stored vector values are identical, so a different valid
point permutation yields exactly the same reordered value bytes. CPU inspection
and two separately recorded unchanged-native GPU3 controls audit this boundary.
Thrust sort_by_key, used by the unchanged upstream grouping/Morton path, does
not specify stable ordering for equal keys. We must not turn an unjustified
cross-run tie-order requirement into a numerical rejection or silently waive it.

The failed *byte-invariant permutation* claim is retired explicitly. The new
semantic gate requires all of the following instead:
- unchanged upstream ordering/group-construction source, not a new sort method;
- a bijective actual original-ID map and unchanged dimension-map bytes;
- the existing C++ bitwise check that actual reordered values match original
  input gathered by that actual map and dimension order;
- independent raw bitmap -> actual original IDs, full canonical oracle equality,
  no dropped/duplicated original IDs and complete candidate-set identity;
- complete safe/fallback count and dimension accounting, as before.
Record snapshot map equality and per-position reordered-value equality as
attribution diagnostics, not public correctness. A false map or pair mismatch
still fails closed. No threshold, input, output policy, numerical envelope,
CUDA binary, RT algorithm or performance denominator is changed.

The first four A2 GPU3 slots remain valid; they passed the stronger old check.
Retain the failed zeros31 slot. Under a separately frozen validator, run zeros31
with a new `_g3_a3` ID and then the remaining original slots in their exact order.
Do not rerun already valid observations to select favorable data. The composite
receipt has20 admitted slots on GPU3 plus one retained validator-proxy failure;
it must never reclassify the original A2 receipt as complete or passing.

Original A0/A1 occupancy blocks and GPU2 historical passes also remain. Native
controls are diagnostics, not repaired-baseline observations. Additional failures
stop the continuation; no acceptance criterion is lowered to admit wrong IDs.

Source reference: [NVIDIA Thrust sort_by_key contract](https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga667333ee2e067bb7da3fb1b8ab6d348c.html).
The initial native diagnostic's isolation was not admitted because a retiring
PID was unattributed. Retain it; use two new identity-held native controls before
continuing. The corrected wrapper may hold its owned completed child unreaped
for1 second (with a bounded child-only watchdog) to preserve PID attribution.
That host diagnostic delay is outside all performance admission; it is not a
change to the unchanged guard or a waiver of foreign occupancy.
