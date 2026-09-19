# Protocol G3C-B-R1: scale-equivariant final FP32 margin

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g3c_b_r1_gpu_cascade`

Status: frozen before execution.

Candidate: `src/run_g3c_b_r1_gpu_cascade.py`, SHA-256
`637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d`.

## Trigger and preserved evidence

The frozen G3C-B candidate passed two isolated G2A processes, generated-code
semantics, and all seven adversarial correctness cases.  It reduced G2A FP64
work to 97 of 102,079 G3B-R1 ambiguous pairs.  However, the `scale_2^-8`
metamorphic case sent 49,452 of 103,266 pairs to FP64 because the final
expression margin used `max(abs(d2_hat)+R0, 1)`.  This is safe but not
scale-equivariant.  All original G3C-B evidence, including the initial
cross-normalizer audit false negative, remains immutable.

## Only admitted numerical repair

R1 is a separate source.  Relative to G3C-B it changes only:

```text
old: final_magnitude = max(abs(d2_hat) + R0, 1)
new: final_magnitude = abs(d2_hat) + R0
```

The unchanged `4096*tiny` absolute term remains before and after the final
relative margin, so the FTZ/near-zero floor is retained.  G3B-R1, the first
FP32 radius coefficients, threshold endpoints, equality path, FP64 refinement,
tile schedule, and all other source logic are unchanged.

## Acceptance

1. Two isolated G2A processes must reproduce the frozen G3B-R1 count/hash,
   have zero unsafe decisions, exact final output, no duplicates/overflow, and
   send no more than 97 pairs to FP64.
2. The G3B cubin must remain byte-identical to frozen G3B-R1.  The new G3C
   cubin must be stable across both processes, contain only the intended FP32
   data arithmetic, and have zero stack/local spill.
3. All seven adversarial/metamorphic cases must retain exact final output and
   zero unsafe decisions.
4. The `scale_2^-8` case must send at most 10,326 pairs (10% of 103,266) to
   FP64.  Failure stops R1 even if correctness passes.
5. Full-process compute-sanitizer memcheck must report zero errors and leaks.
6. A 1,000-launch alternating-buffer stress test must have invariant counts and
   hashes, zero overflow, and zero allocator residue.

No timing is admitted in this ladder.  Any later performance campaign must
measure the complete exact pipeline against a same-contract comparator.

## Claim boundary and safety

Passing R1 supports the frozen G2A shape plus the seven tested transformations
on the audited SM120a compiler artifacts.  It does not prove arbitrary float32
inputs, dimensions, architectures, compilers, or end-to-end speed.

Every GPU process uses physical GPU 0, the project lock, 30 seconds of prior
empty quiescence, continuous 0.25-second monitoring, and empty postflight
compute state.  A foreign process terminates only this experiment group; never
terminate the foreign process.
