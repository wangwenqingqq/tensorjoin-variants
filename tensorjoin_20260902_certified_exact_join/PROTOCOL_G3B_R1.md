# Protocol G3B-R1: normal-or-zero residual-bound repair

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_gpu_analytic_certificate_g3b_r1`

Status: frozen before R1 execution.

## Trigger and preserved evidence

The first G3B adversarial process passed all five metamorphic G2A cases, then
stopped before the all-zero kernel launch because host preprocessing encoded an
exact zero residual upper bound as the smallest float32 subnormal. This violates
the declared finite-normal-or-zero GPU domain. The original candidate, partial
log, cache, and failed manifest remain immutable.

## Only admitted repair

R1 is a separate source:
`src/run_g3b_r1_gpu_analytic_certificate.py`, SHA-256
`057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec`.

Relative to the frozen G3B candidate, R1 changes only host residual-bound
storage and evidence names:

1. if the inflated float64 residual root is exactly zero, store float32 zero;
2. if a positive outward-rounded residual bound is subnormal, round it further
   upward to the smallest normal float32 value;
3. do not change the Triton certificate kernel, constants, schedule, predicate,
   or FP64 refinement.

The repair must be inactive on the frozen G2A source: the two validation runs
must reproduce the previous direct, ambiguous, final, and canonical hashes.

## Execution ladder

1. Run the seven cases frozen in `PROTOCOL_G3B_ADVERSARIAL.md` under isolation.
2. If all pass, run two isolated G2A validation processes.
3. Snapshot and re-audit the R1 cubin: INT8/INT32 MMA, square-root path,
   registers, stack/local memory, and floating margins.
4. Run full-process compute-sanitizer memcheck with leak checking.
5. Run 1,000 certificate launches on alternating buffer sets.

Each GPU process requires the physical-GPU0 lock, 30 seconds of prior
quiescence, no never-owned process overlap, and empty postflight compute state.
No timing from this ladder is performance evidence.

## Acceptance

R1 passes only if:

- all seven adversarial/metamorphic cases have zero unsafe decisions and exact
  final output;
- two G2A processes match all frozen hashes and counts;
- generated-code semantics remain covered, with zero stack/local spill;
- memcheck reports zero errors and leaks;
- 1,000 launches have invariant counts/hashes, zero overflow, and zero
  allocator residue.

Any failure is retained. Only a full R1 pass satisfies the adversarial
requirement in `GATE0_G3.md`; the claim remains limited to this shape,
arithmetic domain, GPU architecture, compiler artifact, and tested cases.
