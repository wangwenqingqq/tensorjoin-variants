# G9 R2: submission-path attribution, not an A0 retune

Declared after A0 completion but BEFORE any Graph measurement, 2026-09-04.
A0 is REJECTED under its no-Graph contract and remains unchanged. No candidate
passed its1.15x actual-data opportunity gate.

## Specific reopen evidence

For real_identity process0, separately submitted stage diagnostics for diagonal
TC were approximately42.61+24.90+20.26+17.99 microseconds, while the complete
staged path median was82.24 microseconds. P16's standalone FP64 stage was18.28
microseconds, but its complete path median was72.91 versus65.53 for standalone
FP32. Therefore individual stage samples include substantial submission effects;
their sum is not a GPU critical-path decomposition. This does not yet prove
the whole candidate is submission-bound.

User intent was an optimistic compute-plus-repair feasibility check. Before
interpreting A0 as an arithmetic-level limit, perform one bounded attribution
experiment: remove per-kernel Python submissions using CUDA Graph. No kernel,
tile, precision, input, queue, comparator, or method selection changes. This is
NOT a favorable-order retry, an A0 pass, or permission to tune rejected kernels.

## Frozen delta

Copy the A0 runner to `src/run_g9_graph_diagnostic.py`; preserve the original.
Capture exactly one complete staged path per method/workload after compilation
and warm execution. Time Graph replay. Initial encoding and all intermediate
queue construction remain free as in A0; ALL TC, certificate, FP32 and FP64
computation kernels and aligned output writes remain inside the Graph.

The strong P16+FP64 comparator is also captured and measured in the same mode.
Capture/setup is excluded for both. Use the same physical GPU7, canonical lock,
quiescence and contamination abort. Same six main workloads plus three signed
boundary fixtures; same20 warmups,40 rotating rounds,20 paths per sample,
two fresh forward/reverse processes, and500-path sustained diagnostics.
Individual eager stage times, if retained, remain explicitly diagnostic and
are NOT a decomposition of Graph latency.

After output reallocation, RECAPTURE using the new pointers before replay.
Require guards, exact integer dot checks, interval tests, stage-queue checks,
resolved main outputs, and memcheck/synccheck in the Graph mode. Full compiled
cubin set must equal A0's admitted set; a kernel change invalidates attribution.

Primary estimator and gate remain P16/candidate paired log ratio: >=1.15 in
BOTH processes on at least one ACTUAL layout and sustained sign agreement.
No extra process, tile/precision sweep or estimator selection after results.

If Graph also fails, close these two TC schedules, including the optimistic
captured regime. If Graph passes, record only a fixed-queue captured compute
opportunity. Dynamic queue/count formation, encoding amortization and final
output export still need a NEW fully-costed protocol. Never report a Graph win
as an eager win, an online implementation, whole-join speedup, or novelty pass.

Labels: g9_graph_check_a0, g9_graph_memcheck_a0, g9_graph_synccheck_a0,
g9_graph_p0_a0, g9_graph_p1_a0. Original A0 evidence is immutable.
