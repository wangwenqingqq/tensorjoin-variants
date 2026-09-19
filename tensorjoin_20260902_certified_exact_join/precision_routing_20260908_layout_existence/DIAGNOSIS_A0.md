# A0 admission failure: preserve, diagnose, do not relax

The original-order F8/F16 census matched the prior output and uncertainty counts.
The first raw_tree F8 run produced U1=1827008 instead of1827007; U2 remained1734.
Its full output was not checked before common.one's invariant assertion raised.
No formal timing or sanitizer admission was started. Raw failure is retained.

A separate read-only diagnostic will capture full logical outputs and U1/U2 sets
for all layouts without using count equality as an early-return condition. It
cannot admit this failed causal experiment. Changed pairs will be minimized to
a two-vector orientation test with unchanged F8 programs. No old kernel changes.
If row permutation changes scalar operand order and hence interval decisions,
use a NEW frozen scheduling-only intervention with original row IDs/orientation;
never change the old protocol to tolerate count differences.
