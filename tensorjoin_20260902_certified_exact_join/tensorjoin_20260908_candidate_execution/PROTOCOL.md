# Online candidate execution experiment

Frozen before GPU measurements, 2026-09-08. This tests a concrete implementation,
not a novelty claim. Full represented FP32 CIFAR 60000x512, all six inherited
thresholds, complete sorted directed CPU results including self, frozen FP64
terminal and common CUB output. Previous artifacts are read-only.

Primary layout is the input-only PCA sort from the preceding structural probe.
Controls: original full FP16, reordered full FP16, conservative projected AABB
64-block filtering, projected pair filtering followed by full 64-block fallback.
Candidates: the same projected pair filter with occupied 16x16 subtile execution,
or packing <=16 active rows/columns into 16x64/64x16 matrices, falling back to
64x64 otherwise. Prefer the smaller active dimension; ties choose rows. No tuning
from timings, no oracle information at runtime. Rank is fixed at 64. KD layout is
a secondary diagnostic at the original threshold, not a separate dataset.

Projection coordinates are computed in FP64, then converted to FP16 with
subnormals flushed explicitly. Let H bound the largest half-vector norm and E
bound each point's total projection/half conversion error (including sqrt(64)
times 2^-24 coordinate projection allowance). Squared-distance discrepancy is
bounded by 8HE+4E^2. Norms of half vectors are accumulated in FP64 and stored
FP32. Add (2*GAMMA+16*2^-24)*H^2+2^-20, using the existing full 512-dimension
FP16 accumulation GAMMA conservatively for 64 dimensions. Reject only if the
computed projected distance exceeds an upward-rounded FP32 limit consisting of
(T+2^-20)*(1+||P^T P-I||_F+2^-20) plus the above margins. AABB uses the same
inflated limit. Scope: finite admitted input; engineering numerical argument,
not a new universal proof. All runtime false exclusions are checked by complete
reference equality in admission and count/SHA256 thereafter.

All remaining matrix products use the inherited FP16 norm/residual enclosure
and directed FP32 interval calculation, with the inherited conservative GAMMA;
unchanged frozen FP32 refinement/FP64 terminal handle ambiguous pairs. New
rectangular kernels alter indexing and tile shape, not the acceptance formulas.
Capacity per batch is at most 2^24 pairs for every shape. GPU nonzero compaction,
packed index generation, host synchronizations and ID remapping are timed.

Queries include fresh GPU upload and metadata, candidate generation, work lists,
all computation/refinement and complete CPU output. Input-only CPU PCA, layout,
projection metadata and bounds are timed separately. Report both reused-index
query time and actual build-plus-query time; no single-query speed claim may
omit construction. Compilation and output verification are outside timing.

Admission: every method at every threshold, exact array equality, tail/diagonal
coverage through full data; separate fixed-sample projection and packing checks.
Then freeze sources/dependencies. Three fresh confirmation processes, three
retained repetitions each, order reversed/rotated across process and repetition;
one untimed warmup per configuration. Report process medians and the range of
paired speed ratios. Passing requires >=5% time saving over the fastest fixed
control on the six-threshold equal mixture in every confirmation process. Report
all points and failures, even if negative. Diagnostic GPU events are separate.
If the prototype is slower, stop after this bounded implementation; do not tune
until a gain appears. A positive result only warrants independent datasets and
closer prior-art comparisons.

Exclusive existing GPU2 guard: 30 seconds idle, 4 CPU threads, <4096 MiB device
use, <=1800 seconds per process; terminate only own descendants on violation.
Preserve sources, input hashes, raw logs, compiled identities and reports.
