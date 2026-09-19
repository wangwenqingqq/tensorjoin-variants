# Gate1: small complete-cost advantage, not a novelty result

The eight direction-balanced fresh processes completed all 432 calls with the
same canonical output hash. Primary C/A geometric ratio is 1.0897755254,
95% order-stratified bootstrap CI [1.0389620821, 1.1402110329], 7/8 wins.
Repeat16 is 1.0710711550 [1.0531156900, 1.0861210621], 8/8 wins. Both order
subgroups exceed 1. The predeclared narrow statistical criterion passes; the
mean advantage is below 10% and must be described as small.

The C control has 86,029 ambiguous stage1 pairs versus A's 1,827,007. Both send
1,734 pairs to the terminal and produce 3,926,078 directed IDs. Queue size alone
therefore does not predict the complete denominator. Neither the old 5.08x nor
old 1.52x FP32 comparison is imported into this result.

Retain the material caveat: primary AC and CA subgroup ratios are 1.0410248129
and 1.1408092112. One admitted primary process loses. One A repeat block's
last4/first4 ratio is 1.4495270673; its cause is not established by the guard.
The no-foreign-GPU-occupancy guard is not proof against CPU/thermal/system noise.
No sample was dropped, repeated or replaced. This is warmed repeated complete
execution with correctness-hash gaps, not continuous saturation or broad
hardware/dataset evidence.

Next: the frozen 2x2 experiment tests matched fused FP16, and separately the
precision-by-fusion interaction. Gate1 does not show that INT8 is intrinsically
better than an equally fused FP16 pipeline.

Evidence: results/g1_analysis.json; results/g1_samples.csv; per-process JSON
and guards; artifacts/g1_source_freeze.json; PROTOCOL_G1.md. The two CPU-only
preflight repairs are retained in raw/ and the two admission notes.
