# Row-layout causal test: admission failed, numerical cause localized

The predeclared experiment required exactly the same uncertainty-ID sets after
row permutation. That requirement failed; no formal latency processes or
sanitizer-admitted timing were run. This is an invalid intervention for that
strict causal question, not evidence against the existence of plan switching.

| Layout | F8 U1 count | Missing vs original | Extra vs original | F16 U1 count | FP64 U2 |
|---|---:|---:|---:|---:|---:|
| Original | 1827007 | 0 | 0 | 86029 | 1734 |
| Raw tree | 1827008 | 1 | 2 | 86029 | 1734 |
| Hadamard tree | 1827007 | 1 | 1 | 86029 | 1734 |

All six separately diagnosed full logical outputs match3926078 IDs and the
original fixed-reference hash. Both plans' terminal U2 sets remain identical;
F16 U1 sets also remain identical. Only count checking would miss the Hadamard
set change. The first census stopped before checking the failed cell's output;
the separate diagnostic, not the failed run, supplies its full output check.

Minimal actual-GPU experiment: place each affected pair at physical rows0,64,
then swap their vectors while keeping every other value and kernel unchanged.
The same INT8 dot is obtained in both orientations, but the staged FP32 product
fl(fl(integer_dot*scale_i)*scale_j) differs by1 ULP. The original F8 classifier
changes between direct reject and uncertain; the common later stages restore
the same final answer. No accepted answer was shown to change.

Affected original unordered pairs: (24938,58195), (42182,45621), (46219,53563).
See results/diagnosis_a0.json for actual scales, integer dots, FP32 bit patterns
and per-orientation GPU classifications. This localizes the intervention issue;
it does not assert a failure of the existing conservative bounds or a new
universal correctness guarantee. No numerical kernel was repaired or changed.

Census PID655396: failed invariant, raw/census_a0.log retained.
Diagnosis PID656873: passed, raw/diagnosis_a0.log retained.
Host gpu-host-8, original GPU2, guards verified, peak1264MiB.
The planned runner/admit/analyze sources are unexecuted, not accepted artifacts.

A separate experiment preserves original row positions and pair orientation,
and varies ONLY the tile visit order/batch grouping:
../precision_routing_20260908_schedule_existence/PROTOCOL.md.
The original protocol and source freeze remain untouched. Its acceptance
criteria have not been relaxed or retroactively reclassified.
