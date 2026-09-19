# Design G4B: dimension-density-scale opportunity matrix

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g4b_public_breadth_opportunity`

## Target and hypothesis

- Hardware: RTX PRO 6000 Blackwell, physical GPU1, `sm_120`.
- Shapes: the 27 native-dimension cells frozen in `GATE0_G4B.md`: three
  public datasets, `N in {1024,2048,4096}`, and target degree
  `k in {1,16,64}`.
- Hypothesis: the existing analytic INT8 certificate plus certified FP32
  filter remains exact and selective outside the single CIFAR-4096 cell.
- Status boundary: correctness and routing geometry only. No G4B measurement is
  a latency, throughput, end-to-end, or external-system performance claim.

G4B introduces no new device-side arithmetic mechanism. It reuses the frozen
G3B-R1, G3C-B-R1, and G2A FP64 kernels and adds only a dimension-parameterized
host contract and public-data matrix runner.

## Numeric and output contract

- Source values are the exact prepared float32 rows; no normalization,
  projection, padding, truncation, or dataset-specific transform is allowed.
- Per-row symmetric INT8 codes lie in `[-127,127]`. INT32 dot accumulation and
  its conversion to FP32 are exact because every admitted native dimension
  satisfies `D*127^2 <= 2^24`.
- Residual-norm inflation uses `gamma_(2D+2)`, with `D` taken from each cell,
  then rounds the result upward to a normal-or-zero float32 bound.
- The G3B analytic constants and G3C certified-FP32 constants are unchanged
  from their accepted sources. Exact FP64 uses the original float32 operands
  widened to float64 and the frozen squared-distance threshold.
- Every stage handles canonical upper-triangle IDs `i*N+j`, including every
  self pair. The final sorted IDs must equal the independently generated
  direct-FP64 oracle byte-for-byte.

The frozen oracle uses SciPy `pdist(..., metric="sqeuclidean")` on the source
float32 rows widened exactly to float64. Radius selection uses a representable
strict midpoint between adjacent distance values. The oracle is independent of
all quantization and GPU certificate decisions.

## Layout, CTA roles, and dispatch

| Unit | Ownership and role |
|---|---|
| Host oracle process | Select strict thresholds and materialize exact upper-ID arrays |
| Host opportunity process | Validate all hashes, quantize once per dataset/scale, build the triangular tile schedule, and check stage partitions |
| G3B Triton program | Own one scheduled 64x64 upper tile; execute 64-wide INT8 dot steps and classify direct accept/reject/ambiguous |
| G3C Triton program | Own one actual G3B-ambiguous pair; execute 256-wide masked FP32 reduction and classify accept/reject/FP64 |
| G2A FP64 Triton program | Own one residual pair; execute 256-wide masked direct-FP64 reduction and append exact accepts |

All source and code matrices are contiguous row-major except the explicit
K-major transpose used by the second INT8 operand. Dataset dimensions 128, 512,
and 784 are compile-time `K` values; the last 64- or 256-wide block is masked.
Each cell allocates capacity equal to all entries covered by the scheduled
upper tiles. Dynamic counters determine the exact input length of the next
stage. No worst-case count is used as executable work.

## Persistent state and phase live set

| Phase | Live objects | Owner and last use |
|---|---|---|
| Oracle | float64 vector prefix, condensed distances, upper row/column map | CPU process; released after all three density oracles at that scale |
| Quantize/copy | float32 source, INT8 codes, scales, code norms, residual bounds | Host; temporaries released after device copies |
| G3B | codes, transpose, scales, norms, residual bounds, tile coordinates, result/ambiguity buffers, counters | GPU process; G3B ambiguity survives to G3C |
| G3C | source, ambiguity IDs, result/FP64 buffers, counters | GPU process; residual IDs survive to FP64 |
| FP64 | source, residual IDs, result buffer, counters | GPU process; final accepted prefix survives to validation |
| Validate | sorted stage IDs and independent oracle | CPU process; released after immutable JSON output |

The imported device kernels own all accumulator fragments and append positions.
The generic runner does not add persistent device state or reuse any output
buffer before a device synchronization and counter read establishes its valid
prefix.

## Ready graph and synchronization

```text
prepared file and row hashes valid
-> oracle threshold and exact upper IDs valid
-> dimension-specific quantization bounds ready
-> GPU copies and upper-tile schedule ready
-> G3B completes and dynamic ambiguity count is read
-> G3C launches on exactly that ambiguity prefix and completes
-> dynamic FP64 count is read
-> FP64 launches on exactly that residual prefix and completes
-> final counter and ID prefixes are copied
-> stage partitions, sound decisions, structure, and exact oracle equality pass
```

Each stage is deliberately synchronous. Counter reset and buffer reuse occur
only between density cells, after the prior cell's validation has completed.
The isolation wrapper owns only the child process group; discovery of another
GPU process terminates that group and never signals the foreign process.

## Work, movement, and lower-bound hypotheses

- G3B always performs one scheduled triangular INT8 Tensor-Core pass.
- G3C work is exactly the measured G3B ambiguity count.
- FP64 work is exactly the measured G3C residual count.
- The reusable quantization and device copies are performed once per
  dataset/scale, but G4B does not time or claim that amortization.
- The expected opportunity is meaningful only when both ambiguity fractions
  satisfy the predeclared cross-dataset gates. A full-forwarding certificate is
  a failure even if exact.

No roofline or timing floor is asserted at this stage. The matrix records
useful dynamic pair counts and stage traffic opportunities; any later timing
subset must separately freeze compilation, launch, host, transfer, and
end-to-end denominators.

## Comparator, rejection, and verification ladder

- Correctness comparator: independent direct-FP64 oracle for every cell.
- Mechanism comparator: G3B ambiguity before G3C and residual FP64 count after
  G3C under identical source rows and threshold.
- Reject immediately on source/oracle hash mismatch, failed accumulator bound,
  unsafe G3B/G3C accept or reject, partition mismatch, duplicate, invalid or
  lower-triangle ID, overflow, or final mismatch.
- Stop before generic timing if either selectivity condition in
  `GATE0_G4B.md` fails. Preserve every cell and the failed aggregate decision.

Verification ladder for this gate:

1. static Python compilation and immutable source hashes;
2. CPU oracle construction with source, row-prefix, threshold, and output hashes;
3. isolated GPU execution on the live target with occupancy evidence;
4. 27/27 exact stage and final-output validation;
5. aggregate opportunity decision against the frozen gate.

Compilation, generated-code inspection, sanitizer results, timing, and
end-to-end comparisons are explicitly not inherited by G4B and remain separate
future gates if the opportunity matrix passes.
