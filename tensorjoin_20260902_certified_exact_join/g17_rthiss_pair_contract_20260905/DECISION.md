# G17 decision: RT-HiSS pair export admitted; FP64-reference equality not admitted

2026-09-05. Correctness and code-attribution experiment on gpu-host-8, physical
GPU 2 (RTX PRO 6000, SM120). No public performance result or manuscript revision.

## Conclusion and next action

The native RT-HiSS algorithm now has an independently checked, original-row-ID
output adapter. Its default CUDA refinement and compression instruction streams
match the unmodified-source control build. On the frozen CIFAR4096 input, native
RT-HiSS and the FP64 reference both return **262,144 directed pairs**, but there
are **2 missing and 2 extra IDs**. Equal counts concealed different answers.

All disputed pairs reached refinement. Scalar C `fmaf` replay and a same-launch
CUDA FP32/FP64 diagnostic reproduce all four disagreements. This is a localized
native-FP32 versus frozen-FP64 predicate discrepancy, not evidence of an output
decoder bug. It is not a claim that RT-HiSS is generally incorrect.

Keep this modern comparator. Next design and test a separately named two-sided
conservative repair that can recover false negatives as well as remove false
positives. Do not just rerank emitted pairs, weaken the reference, substitute an
FP64-all-pairs strawman, or discard RT-HiSS. See `NEXT_BASELINE_REPAIR.md`.

G16's positive full host-to-host screen remains unchanged: 1.499724/1.267190
same-block ratios versus the equally GPU-prepared pedantic FP32-first control.
G17 supplies neither a new speedup nor a novelty pass. The material paper gap
remains a differentiated, non-incremental mechanism plus modern equal-output
comparisons, not a lack of more favorable prose.

## Frozen contract and implementation

- Official RT-HiSS commit `a42fc69cc4b602dc83071b029d185a41e69a04bd`;
  OWL `c7c3a3ea35b17b5c096a3802ba74b9d8b4e2772a`. Source availability was already
  established in G7; G17 advances beyond its count-only, low-dimensional smoke.
- Separate `adapter_a0`; upstream/G7/G5/G15/G16 core sources remain untouched.
  Default shared/shared refinement, compressed mask, dimension and point order,
  RT candidate construction, batching and tiling are retained.
- Input: stored binary FP32, D512, N <= 4096, finite |coordinate| <= 1, positive
  epsilon. Output: sorted unique original directed IDs, including actual self
  bits. No filling, deduplication to hide errors, or threshold substitution.
- Capture/replay checks both permutations bitwise. Production compressed-bit
  decoding is cross-checked against independently reconstructed raw bitmap IDs.
  Prefixes, padding, bounds, candidate counts, popcounts and capacity fail closed.
- Debug mask capture and disk export are intentionally present. Any operational
  durations in logs are NOT comparable public latency; `performance_admitted`
  is false. Graph, sustained 60K, general arbitrary-input safety and exact-real
  predicates were not tested or established.

## Complete bounded result inventory

| Input / D512 | Native pairs | FP64-reference pairs | Missing / extra directed IDs | Independent structure |
|---|---:|---:|---:|---|
| real31 | 31 | 31 | 0 / 0 | pass |
| signed31 | 31 | 31 | 0 / 0 | pass |
| shuffled_onehot31 | 97 | 97 | 0 / 0 | pass |
| zeros31 | 961 | 961 | 0 / 0 | pass |
| boundary31, original fixture | 43 | 43 | 0 / 0 | pass |
| cancellation31 | 961 | 961 | 0 / 0 | pass |
| subnormal31 | 961 | 961 | 0 / 0 | pass |
| cifar4096, frozen G2A anchor | 262,144 | 262,144 | 2 / 2 | pass |
| boundary_zero32, additive coverage correction | 106 | 104 | 0 / 2 | pass |

All pairs are candidates on these particular inputs, including 16,777,216 on
CIFAR4096. There are zero candidate omissions of reference-positive pairs.
This establishes coverage on this matrix only; it does not prove conservative
RT traversal in general or that RT indexing never prunes useful workloads.

The original boundary31 fixture lacked the zero endpoint needed for its intended
unit-distance witness. It remains unchanged. The predeclared additive32 fixture
appends that endpoint; it finds the two directed `(1, 2^-13)` versus zero extras
(FP64 distance 1.0000000149011612, native FP32 1). This coverage gap and correction
are retained in `ATTEMPTS.md` and `ADDENDUM_BOUNDARY_ZERO32.md`. The smaller
`2^-27` coordinate also reminds us that FP64-reference equality is not exact-real
arithmetic; G10's distinction is not waived.

## First failing boundary on real data

The original epsilon text is 0.7541135250198396. Frozen reference threshold:
0.5686872086178483; native rounded-FP32 squared threshold: 0.5686871409416199.
Recomputing the FP64 oracle using the latter threshold yields the same set on
this input, and still leaves both missing and extra pairs. Threshold conversion
alone is therefore insufficient to explain the observed discrepancy.

| Original unordered pair (both directions disagree) | FP64 direct-distance sum | Native FP32 sum | Native / reference |
|---|---:|---:|---|
| (857, 1584) | 0.5686869777202842 | 0.5686873197555542 | reject / accept |
| (1188, 2087) | 0.5686874395154595 | 0.5686870813369751 | accept / reject |

Native output SHA-256:
`8b9220ad108bd29b09f15fec387aea2529b0234e838f9d6e4468d29f437218d9`.
Frozen oracle SHA-256:
`da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`.
The standalone CUDA probe runs both arithmetic paths in the same launch using
actual dimension-permuted coordinates, restoring original order for FP64. All
four sums and decisions reproduce. Its guard pass means *diagnostic divergence
reproduced*, not a pass for the native join's FP64 contract.

## Safety, isolation and selected-code gates

- CPU decoder: 3 valid cases and 6 invalid cases rejected as intended.
- Memcheck and synccheck: boundary31, CIFAR4096 and boundary_zero32, all six
  runs report zero errors. The same native numerical differences persist.
  Memcheck access checking used `--leak-check no`; no leak-free shutdown claim.
- Sixteen guarded adapter runs plus one diagnostic probe; occupancy and
  postflight receipts are retained. Exact-output guards remain failed on the
  numerical mismatches, even though predeclared structural characterization
  continued. Nothing reclassifies those failed exact gates as passes.
- Nsight Systems observes the adapter's default shared/shared refinement and
  compression. Their normalized selected-function SASS matches the native
  control: 336 and 408 static instructions respectively. Full executable and
  OWL hashes are retained independently; container hashes differ.
- Runtime-selected refinement: 36 registers/thread, 1024-thread block, 47,104
  bytes dynamic shared memory on the profiled 31-row fixture. Compression uses
  16 registers/thread and 1024-thread blocks. No latency or occupancy-win claim.
- OptiX driver-JIT code identity is not established. The RT sources are unchanged;
  selected CUDA equality is not a proof of all driver-generated code identity.
- Build, SSH/deployment interruptions and the pre-build diagnostic extraction
  correction are preserved, not converted into rejected performance samples.

## Three separate decisions

| Level | Status | Exact scope / reopen condition |
|---|---|---|
| Implementation | measured, bounded adapter admitted | Original-ID reconstruction and checked metadata pass; exact FP64 comparison remains rejected for native output on the two failing inputs. |
| Mechanism | measured numerical localization; repair unknown | Native refinement arithmetic explains all observed disagreements. A separately frozen conservative repair must recover both error directions and retain efficient native work. |
| Thesis | unknown, not promoted | Modern comparator semantics are clearer; generic filter/refine, adaptive precision, output compression and GPU preparation remain prior-art-adjacent. G16's positive is preserved, not generalized. |

## Reproduction and evidence

`PROTOCOL.md` preceded implementation and the original matrix;
`artifacts/frozen_execution.json` and `_r1.json` bind original and additive runs.
`results/native_matrix_a0.json`, `safety_matrix_a0.json`, `extension_a0.json`,
`case_*.json`, `pair_replay_a0.json`, and `code_audit_a0.json` contain full results.
`raw/<record_id>/` holds input maps, bitmaps, compressed positions, canonical
IDs and logs. Guard/preflight/occupancy copies and the final hash inventory are
in `artifacts/`; see `results/closure_checks.json` for verified scope.

Follow the frozen command arrays; fresh repetitions require new record IDs and
live host/GPU checks. Never overwrite the frozen runs. No paper was edited,
compiled, visually verified or synchronized to Overleaf by G17.
