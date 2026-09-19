# Statistical implementation details (before retained timing)

Each process has six paired retained repetitions per cell (exploration has
three). For each process/cell/path use its retained median. The reported
path latency is the arithmetic mean of these process medians within a block;
R is the ratio of those path latencies. Also show all process ratios.

Use a paired hierarchical bootstrap: sample process indices with replacement,
then sample paired repetition indices with replacement within each selected
process/cell, preserving the F8/F16 pairing. Recompute medians, path means,
ratios and oracle headroom in each draw. The six cells are fixed and have
equal query probability, so they are all retained when estimating mixture
headroom. 20000 draws, NumPy PCG64 seed 2026090802 (explore), 2026090803
(primary), 2026090804 (repeat), 2026090805 (pooled, descriptive only).
Percentile 95% intervals; three process clusters per independent block.
The bootstrap expresses measured variation on this machine/workload; it
does not establish broad hardware or data-distribution generalization.

Oracle cost is sum(min(mean_process_median_F8, mean_process_median_F16))
over the six cells. Best static cost is min(sum(F8),sum(F16)). Headroom
means fractional wall-time saved: 1 - oracle_cost / best_static_cost.
The conservative gate requires lower 95% headroom bound >= 0.05 in both
blocks, plus the same pair of opposite winning cells, their ratio CIs
outside [1/1.05,1.05], and the matching direction in every process.

Paired order effects are reported separately as a sensitivity check (path
latencies when F8 runs first versus when F16 runs first). No sample is
discarded based on latency, and no warmup is counted as retained timing.
Exploration is excluded from all confirmation estimates and gates.
