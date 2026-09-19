# Fixed-weight ODE resolution follow-up

Motivation observed after the main model/solver selection: the selected Heun-2
endpoint has relative RMS discrepancy 0.682 versus Heun-16 on the preregistered
512 full-data diagnostic tiles. The main experiment and all its frozen inputs
remain unchanged. This follow-up separates inadequate ODE resolution from the
quality of the learned conditional field. It is adaptive and supplementary.

Use exactly the selected FM checkpoint, without training or model reselection.
On 512 validation block pairs of the original threshold, sampled with seed
2026090931, compute endpoints at 2/4/8/16/32/64/128/256/512 Heun steps. The 512
endpoint is a numerical reference, not an exact ODE solution. Select the first
of 16/32/64/128/256 steps with relative RMS discrepancy <=1% and binary-sign
disagreement <=1% against that reference. If none passes, use 512 and explicitly
report that no finer convergence check is available. This choice uses no test
labels, full-query timings or full-data pruning outcomes.

After fixing this resolution, evaluate all six full-data raw candidate predictions
and false-negative reference counts. Re-run real complete queries with pca,
heuristic25, direct25 and the fixed-weight FM25 at the chosen resolution. One
fresh exclusive GPU4 process, one full-array-checked warmup and three retained
repeats per configuration x six thresholds x F8/F16 = 192 complete calls.
Compare times only within this supplementary process; report them descriptively,
without claiming the main three-process confidence gate. Keep all repeats.
Every output count and SHA256 must match; every newly rejected tile must be
reference-empty. The timed engine recomputes all prediction and certificate
work, including all additional ODE calls. No cached prediction or truth route.

Freeze this protocol and worker script with the parent timing-dependency hashes
before the resolution check. Parent selected weights, source, protocol and
results are read-only. Save numerical selection, full raw metrics and all query
records separately. This follow-up is required to qualify conclusions about
whether 'denoising obtains the exact solution'; the coarse Heun result alone
must not be treated as a converged flow evaluation.
