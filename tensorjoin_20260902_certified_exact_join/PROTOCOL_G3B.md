# Protocol G3B: GPU Analytic-Certificate Admission

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_gpu_analytic_certificate_g3b`

Status: frozen before implementation execution.

## Frozen target

- Physical GPU0 on live host `gpu-host-8`; `CUDA_VISIBLE_DEVICES=0`.
- Frozen G2A `4096x512` float32 input, radius, canonical FP64 oracle, upper-only
  tile schedule, and proof-bounded capacity.
- Candidate and constants: `DESIGN_G3B.md`.
- Keeper evidence: accepted G2A2 count/hash and 102,277-pair ambiguity. The
  keeper source is not edited.

## Execution ladder

1. **Validation smoke:** one isolated candidate process after 30 seconds of
   empty physical GPU0. Any output, stage, overflow, or ownership failure stops
   the ladder.
2. **Independent repeat:** a second isolated process must reproduce the same
   direct/ambiguous/final counts and hashes.
3. **Generated-code gate:** snapshot the exact compiled artifact. Require an
   INT8 MMA path with INT32 accumulation, no stack/local spill, and identify the
   square-root/floating path. If its semantics are not covered by the declared
   margins, stop rather than infer safety from output equality.
4. **Memcheck:** full process under `compute-sanitizer --tool memcheck
   --target-processes all --error-exitcode 99 --leak-check full` must report
   zero errors.
5. **Sustained gate:** at least 1,000 launches of the certificate stage on two
   alternating buffer sets with invariant direct/ambiguous counts and hashes,
   zero overflow, and no allocator residue.

Formal timing is forbidden in G3B.

## Correctness checks per validation process

- direct accepted IDs are sorted/unique valid upper IDs and are a subset of the
  oracle upper IDs;
- every oracle upper ID is present in direct-accept or ambiguity, proving no
  unsafe direct rejection on the frozen oracle;
- direct and ambiguity sets are disjoint;
- FP64-refined final upper IDs exactly match the frozen upper oracle;
- symmetric expansion exactly matches the frozen 262,144 directed-ID hash;
- all capacity/overflow counters are zero;
- ambiguity is at most 127,846 upper pairs.

## Acceptance and claim boundary

G3B passes only if all five ladder stages pass. A smoke/repeat pass alone is
`unvalidated`; compilation alone is not correctness evidence. Passing G3B
supports only the statement that this separate GPU analytic-certificate path
meets the frozen G2A proof contract. It does not prove the existing G2B kernel,
the FP32 refinement stage, other shapes/architectures, or a performance claim.

Any failed attempt, including sanitizer teardown leaks or generated-code
semantic mismatch, is retained with its exact command and hashes before repair.

