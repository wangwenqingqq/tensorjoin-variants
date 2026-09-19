# G3C-B-R1 certified-FP32 cascade admission decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_g3c_b_r1_gpu_cascade`

## Decision

**PASS for the frozen `4096x512` G2A plus seven-case adversarial correctness,
generated-code, and safety contract.  The complete three-stage cascade routes
only 97 of 102,079 first-stage ambiguous pairs to FP64.  This is not timing
evidence and does not retroactively prove the accepted G2B performance
implementation.**

G3C restores the intended precision router as a separate proof candidate:

1. the unchanged G3B-R1 INT8/INT32 Tensor Core certificate;
2. an outward FP32 squared-distance interval for the unresolved pairs;
3. direct FP64 refinement only when that interval crosses the threshold.

## Opportunity gate

The frozen G3C-A host model first reproduced the exact 102,079 G3B-R1
ambiguous IDs and evaluated a reduction-order-independent `gamma_511` interval.
It directly accepted 42,154 pairs, rejected 59,885, and left 40 pairs for FP64,
versus 3,176 under the historical fixed `1e-3` guard.  Every exact source
distance was inside its interval and both unsafe-decision counts were zero.

This admitted a GPU candidate; it was not counted as GPU proof or performance.

## GPU validation and scale repair

The first frozen G3C-B GPU candidate used a deliberately padded implementable
envelope.  Two isolated processes were exact and stable.  They left 97 pairs
(0.0950% of G3B-R1 ambiguity) for FP64.  All seven adversarial cases also
returned exact output with zero unsafe decisions.

The adversarial ledger nevertheless exposed a material extensibility weakness:
the `scale_2^-8` case left 49,452 of 103,266 pairs for FP64.  The cause was the
final `2^-22 * max(magnitude, 1)` expression margin, whose unit floor does not
scale with the squared-distance problem.

G3C-B-R1 is a separate source and changes only that expression to
`2^-22 * magnitude`; the unchanged `4096*tiny` absolute terms retain the FTZ
floor.  Before execution, R1 required at most 10,326 FP64 pairs on the downscale
case.  It observes 97, reducing the original count by 509.8x while retaining
exact output and zero unsafe decisions.  The base, negation, row permutation,
dimension permutation/sign flip, and `2^8` cases also leave 97 pairs; all-zero
needs none; the signed maximum-accumulator case safely accepts all 4,196,352
bitwise-equal ambiguous pairs without FP64.

## Frozen G2A result

Both isolated R1 processes reproduce identical counts and sorted hashes:

- 8,390,656 evaluated upper-triangle pairs;
- 90,946 G3B-R1 direct accepts;
- 102,079 G3B-R1 ambiguous pairs;
- 42,123 FP32 direct accepts and 59,859 FP32 direct rejects;
- 97 FP64 refinements, or 0.0950% of first-stage ambiguity;
- 133,120 final upper pairs and 262,144 directed IDs;
- canonical SHA-256
  `da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`;
- zero unsafe direct decisions, missing/extra pairs, duplicates, invalid IDs,
  partition errors, or overflow.

The host opportunity model's 40 and the GPU candidate's 97 are not a
contradiction: the GPU interval intentionally adds power-of-two padding for its
actual reduction and final expression.

## Generated-code gate

The unchanged G3B stage remains byte-identical to the accepted G3B-R1 cubin,
SHA-256 `dd32e979...114ac`, including its 16 INT8/INT32 IMMA instructions.

The R1 FP32 stage has cubin SHA-256 `db750295...a20670` and normalized selected
function SHA-256 `9fdad164...a725c` in both isolated processes.  Static audit
finds:

- only the intended FP32 data arithmetic in PTX/SASS; no FP16/BF16/TF32/FP64
  data arithmetic and no low-precision MMA in this middle-stage kernel;
- 25 FADD, 4 FFMA, and 3 FMUL instructions in the selected SASS function;
- three `HFMA2` instructions used only as zero-initialization idioms with zero
  operands, not half-precision data computation;
- 32 registers per thread, 1,024 shared bytes, zero stack/local bytes, and no
  `LDL`/`STL` instructions;
- implemented/derived coefficient ratios of 1.996x for the distance term,
  4.000x for the input-magnitude term, 4 unit roundoffs for final-expression
  padding, and 2x for the explicit FTZ operation allowance.

The coefficient screen and SASS inspection are complementary.  Neither should
be presented as a timing result or as an architecture-independent theorem.

## Safety gates

- Full-process `compute-sanitizer --tool memcheck --target-processes all
  --error-exitcode 99 --leak-check full` executes all three stages and reports
  `0 bytes leaked` and `ERROR SUMMARY: 0 errors`.
- The sustained run completes 1,000 complete cascades: 1,000 G3B launches,
  1,000 certified-FP32 launches, and 1,000 FP64-refinement launches.
- Every iteration reproduces all six stage hashes and counts.  Count, hash,
  duplicate, partition, and overflow mismatch totals are zero.
- Both safety processes end with zero PyTorch allocated/reserved bytes, no
  foreign-process overlap, and empty physical-GPU0 postflight state.
- Memcheck and stress emit the same accepted G3B and G3C cubins.

## Preserved negative and non-admitted evidence

1. The first G3C-B generated-code audit is retained as a false negative.  It
   compared a new `cuobjdump` address-target normalization with an older
   `nvdisasm` symbolic-label normalization.  The full G3B cubin was already
   byte-identical.  The read-only re-audit uses a single normalization method
   across both new processes and exact cubin equality to the frozen binary.
2. The original G3C-B `scale_2^-8` result of 49,452 FP64 pairs is retained and
   is the predeclared trigger for R1.  It is not counted as R1 evidence.
3. G3C-A's 40-pair host estimate is not substituted for the actual 97-pair GPU
   result.

## Claim boundary and next gate

Allowed wording: **On the frozen `4096x512` G2A workload and seven same-shape
adversarial/metamorphic cases, the generated-code-audited three-stage cascade
has exact output and zero unsafe direct decisions; on G2A it routes only 97 of
102,079 first-stage ambiguous pairs to FP64.**

Not allowed:

- any G3C latency or speedup claim;
- claiming that G3C proves the separately timed G2B kernel;
- generalizing to arbitrary float32 inputs, dimensions, radii, datasets,
  architectures, or compiler versions;
- claiming submission readiness before a same-contract end-to-end timing gate
  and compact public breadth matrix.

The next gate is G3D: freeze a complete-pipeline comparator and test whether the
certified FP32 stage improves end-to-end latency over the unchanged G3B-R1
two-stage proof candidate.  If it passes, the higher-priority paper gate remains
public breadth across dimensions, output densities, and scales—not another
modality port.

Subsequent status: G3D passed as a narrower complete resident-GPU attribution
result, not ingest-inclusive end to end.  See `DECISION_G3D.md`.  The compact
public breadth/dynamic-router gate remains open.

## Evidence

- Aggregate: `results/g3c_b_r1_final_summary.json`, SHA-256
  `82aa22b30ca0380fcb807b3ffd4cef5e33456d17cb39cd784ba234c5c3547f8d`.
- Protocols: `PROTOCOL_G3C_A.md`, `PROTOCOL_G3C_B.md`, and
  `PROTOCOL_G3C_B_R1.md`.
- Opportunity: `results/g3c_a_host_fp32_interval.json`.
- R1 validations: `results/g3c_b_r1_gpu_cascade_process_{0,1}.json`.
- Generated code: `results/g3c_b_r1_generated_code_audit.json`.
- Adversarial: `results/g3c_b_r1_adversarial.json` and manifest.
- Safety: `results/g3c_b_r1_safety_manifest.json`.
- Candidate SHA-256:
  `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d`.
- Frozen source receipt: `receipts/g3c_b_r1_source_sha256.txt`, SHA-256
  `dcd73691a71232c9e7c9b935815e7806441977cba371cee3154e67a6db99b041`.
