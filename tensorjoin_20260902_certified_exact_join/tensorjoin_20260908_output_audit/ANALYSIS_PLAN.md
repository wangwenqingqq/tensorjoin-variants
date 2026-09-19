# Analysis details, fixed during exploration and before confirmation

For each process/cell/method/output use the median retained wall time. A
block estimate is the arithmetic mean of its three process medians. For
confirmation, jointly resample process indices, then independently sample
repetition indices within each selected process/cell/method, preserving the
legacy/new-output pairing. Recompute medians and means in every bootstrap
draw. Use 20000 draws, NumPy PCG64 seed 2026090811 for primary, 2026090812 for
repeat, and 2026090813 for pooled descriptive results. Percentile 95% CIs.

Output speedup = legacy/new. Mixture time saving = 1 - sum(new)/sum(legacy),
over all 18 cell/method combinations with equal invocation probability.
Recommend replacement if saving CI lower bound >= 0.05 in both blocks and
no cell/method has a reproducible regression >5% (speedup CI upper bound
<1/1.05 in both blocks, with every process direction consistent). All such
cells remain visible even when the aggregate result is positive.

Under the shared selected output, report F16/F8 time ratio, F32/F8 and
F32/F16 speed ratios. Within-process method measurements are in different
blocks; the bootstrap pairs the process clusters, not unrelated individual
method calls. Report method order and all per-process estimates.

If whole-query F8/F16 oracle headroom is tabulated, hold the six queries at
equal probability and compute 1 - sum(min(F8,F16))/min(sum(F8),sum(F16)).
It is descriptive, includes no selector overhead and does not reopen the
stopped prior planner claim. Do not report pointwise fastest choices as an
implemented planner. Strong FP32 here means the retained pedantic cuBLAS
control with frozen filter/terminal, not a claim to have exhausted all FP32
algorithms or matched every published system's numerical/output contract.

No latency-based exclusions. Three process clusters per independent block
limit the scope of the confidence intervals. Admission timings include cold
startup/JIT and are never used as retained performance measurements.
