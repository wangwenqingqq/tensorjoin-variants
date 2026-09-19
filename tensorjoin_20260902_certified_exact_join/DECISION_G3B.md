# G3B-R1 GPU analytic-certificate admission decision

Date: 2026-09-03

Experiment: `tensorjoin_20260903_gpu_analytic_certificate_g3b_r1`

## Decision

**PASS for the frozen G2A plus seven-case adversarial proof contract. This is
correctness, generated-code, and safety evidence—not a performance result, a
proof of the accepted G2B timing kernel, or a theorem over all float32 inputs.**

The initial G3B child protocol passed real-data validation, code inspection,
memcheck, and sustained execution, but a parent-level audit found that
`GATE0_G3.md` also required adversarial correctness. The first adversarial run
then exposed a host preprocessing domain defect: an exact zero residual became
a positive float32 subnormal through unconditional outward `nextafter`.

G3B-R1 is a separate candidate. It keeps exact zero at zero and rounds every
positive subnormal residual upper bound further upward to the smallest normal
float32 value. It does not change the GPU certificate kernel, constants,
schedule, predicate, or FP64 refinement.

## Frozen-data validation

Two independent isolated R1 processes on physical GPU0 reproduced the original
G3B result exactly:

- 90,946 direct accepted upper pairs;
- 102,079 ambiguous upper pairs, below the frozen 127,846 limit;
- zero unsafe direct accepts and zero unsafe direct rejects;
- 133,120 accepted upper pairs after direct FP64 refinement;
- the exact 262,144 directed-ID oracle with SHA-256
  `da2c8160...6d5d`;
- zero missing, extra, duplicate, invalid, overlap, or overflow events.

This proves that the R1 preprocessing repair is inactive on the frozen G2A
source. Every unresolved pair still takes direct FP64 refinement, so the
unproved FP32 middle stage is not hidden inside this result.

## Adversarial and metamorphic gate

All seven frozen `4096x512` cases pass with zero unsafe direct decisions and
exact final output:

1. global negation;
2. seeded row permutation;
3. seeded dimension permutation plus component sign flips;
4. exact `2^-8` scaling of input and radius;
5. exact `2^8` scaling of input and radius;
6. all-zero input at epsilon `2^-20`, accepting all 8,390,656 upper pairs;
7. 2,048 all-`+1` and 2,048 all-`-1` vectors at epsilon zero, reaching the
   maximum code norm `512*127^2 = 8,258,048` and exactly returning the two
   same-sign blocks.

These are strong falsification tests, not exhaustive proof over every admissible
float32 vector.

## Generated-code gate

The R1 validation cubin targets `sm_120a` and has SHA-256
`dd32e979...114ac`. Its normalized instruction stream has SHA-256
`8a59d826...e040`, byte-identical to the normalized original G3B stream. Static
inspection finds:

- 16 PTX `mma.sync...s32.s8.s8.s32` operations and 16 corresponding
  `IMMA.16832.S8.S8.SAT` instructions;
- 64 PTX `sqrt.approx.ftz.f32` operations and 64 corresponding `MUFU.SQRT`
  instructions;
- 140 registers per thread, zero stack bytes, zero local bytes, and no
  `LDL`/`STL` instructions.

For `D=512` and codes in `[-127,127]`, every integer partial sum is below both
`2^24` and `INT32_MAX`; saturation cannot activate and INT32-to-FP32 conversion
is exact. The compiler fuses/reorders FP32 expression operations, so admission
uses the declared `2^-16` radius rather than assuming source order. It exceeds
a conservative 16-rounding `gamma_16` envelope by about 16x. The final
`2^-20 * max(magnitude, 1)` margin is 8x the PTX `sqrt.approx.f32` relative
bound and supplies an absolute floor for FTZ inputs.

The adversarial, two validation, memcheck, and stress processes all emitted the
same R1 cubin SHA-256.

## Safety gates

- Full `compute-sanitizer --tool memcheck --target-processes all
  --error-exitcode 99 --leak-check full` completed the certificate and FP64
  refinement with `0 bytes leaked` and `ERROR SUMMARY: 0 errors`.
- The sustained run completed 1,000/1,000 certificate launches on two
  alternating output-buffer sets. Counts and sorted hashes were invariant in
  every iteration; mismatch and overflow counts were zero.
- Both processes ended with zero PyTorch allocated/reserved bytes, no foreign
  process overlap, and empty physical-GPU0 postflight state.

## Preserved failures

The evidence keeps rather than erases:

1. the original NumPy-scalar Triton binder failure before candidate launch;
2. the original stress attempt stopped after 600 clean iterations when a
   never-owned `echophys` process appeared on GPU0;
3. the successful original stress retry's manifest-only `__file__` type error;
4. the first adversarial run's exact-zero/subnormal preprocessing-domain
   failure after five passing cases;
5. the R1 adversarial wrapper's wrong cache-prefix lookup after all seven cases
   had completed successfully; a read-only finalizer admitted the immutable
   result and actual cache without rerunning the GPU workload.

None is counted as a clean positive run, and no foreign process was modified.

## Claim boundary and next gate

Allowed wording: **G3B-R1 meets the frozen `4096x512` G2A and seven-case
adversarial proof contract on the audited SM120 compiler artifact.**

Not allowed:

- claiming that the accepted G2B timing kernel is retroactively proved;
- claiming that the FP32 refinement stage is proved;
- generalizing to all float32 inputs, dimensions, radii, datasets, GPUs, or
  toolchains;
- using any G3B/G3B-R1 duration as performance evidence.

Subsequent status: G3C-B-R1 separately passed the outward-FP32 correctness,
generated-code, adversarial, memcheck, and sustained gates, and G3D then passed
its complete resident-GPU attribution gate; see `DECISION_G3C.md` and
`DECISION_G3D.md`. Neither result retroactively extends the G3B-R1 claim. The
public breadth/dynamic-router gate remains open.

## Evidence

- Aggregate: `results/g3b_r1_final_summary.json` (SHA-256
  `19674be51c0a4ebe6ef8a60bb82ecab6281f7e399a0e85223f608ed4c93e0f0d`).
- R1 protocol: `PROTOCOL_G3B_R1.md`.
- Adversarial: `results/g3b_r1_adversarial.json` and
  `results/g3b_r1_adversarial_final_manifest.json`.
- Frozen validation: `results/g3b_r1_gpu_analytic_process_{0,1}.json`.
- Generated code: `results/g3b_r1_sass_validation.json`.
- Safety: `results/g3b_r1_safety_manifest.json`.
- R1 candidate SHA-256:
  `057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec`.
- Final 110-file evidence ledger: `receipts/g3b_r1_final_evidence_sha256.txt`
  (SHA-256
  `8544691615463a5cb205036f2786159075ac639edf8cc45c93d11e01e26f6ca4`).
