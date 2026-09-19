# Protocol G3C-B: GPU certified-FP32 middle stage

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g3c_b_gpu_cascade`

Status: frozen before execution.

Candidate: `src/run_g3c_b_gpu_cascade.py`, SHA-256
`836223f2d0ecf175004948385a5e8e3f32f1605e5f3713f661a17e3174b17e0f`.

## Admission evidence

G3C-A passed its frozen opportunity gate on the exact 102,079 G3B-R1
ambiguous upper-triangle pairs: 42,154 were accepted, 59,885 rejected, and only
40 (0.0392%) remained for FP64, with zero interval violations and zero unsafe
decisions.  G3C-A is host opportunity evidence, not GPU proof or timing.

## Frozen pipeline and task

1. Run the unchanged G3B-R1 INT8/INT32 Tensor Core certificate.
2. On exactly its ambiguous IDs, run the new FP32 interval kernel.
3. Refine only the unresolved IDs with the unchanged FP64 kernel.

The primary task is the frozen G2A CIFAR-10 GIST exact Euclidean self-range
join, shape `4096 x 512`, upper triangle including self, with the exact float64
threshold and oracle already frozen in G2A/G3B-R1.  This is a correctness and
safety ladder only; no duration is performance evidence.

## Frozen GPU interval

For each pair, the kernel explicitly reduces both

```text
d2_hat = sum_f32((x_k - y_k)^2)
A_hat  = sum_f32((abs(x_k) + abs(y_k))^2)
```

using `BLOCK_K=256`.  It then evaluates

```text
R0 = 2^-14 * abs(d2_hat) + 2^-22 * abs(A_hat) + 4096*tiny
R  = R0 + 2^-22 * max(abs(d2_hat) + R0, 1) + 4096*tiny
L  = max(d2_hat - R, 0)
U  = d2_hat + R
```

where `tiny` is the smallest positive normal float32.  These power-of-two
coefficients are a deliberately padded GPU-computable envelope of the G3C-A
source-difference, rounded-square, `gamma_(D-1)` reduction, FTZ, and final
expression terms.  Exact coordinate equality is a separate safe accept path.
Otherwise `U <= threshold_lower_f32` accepts, `L > threshold_upper_f32`
rejects, and a crossing interval goes to FP64.

The mathematical envelope is not by itself generated-code evidence.  The
selected cubin must separately show the intended float32 path, constants and
control, no unintended low-precision arithmetic, and zero stack/local spill.
The claim remains scoped to the audited SM120a compiler artifact and tested
normal-or-zero stored-input domain; it is not a claim over arbitrary float32
inputs, shapes, architectures, or toolchains.

## Correctness acceptance

Run two isolated fresh processes.  Each passes only if:

- input and oracle hashes match;
- G3B-R1 reproduces exactly 90,946 direct accepts and 102,079 ambiguous pairs,
  including ambiguity SHA-256
  `6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42`;
- FP32 accept, FP32 reject, and FP64 sets uniquely partition the ambiguity;
- G3B and FP32 have zero unsafe direct accepts/rejects;
- no more than 10,207 pairs (10%) enter FP64;
- the final upper output has 133,120 pairs and the reconstructed directed output
  has 262,144 IDs with SHA-256
  `da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`;
- there are no duplicates, invalid IDs, overflow, missing pairs, or extra pairs.

## Promotion ladder after two-process validation

1. Snapshot every source and protocol dependency and the exact Triton cache.
2. Audit the runtime-selected G3B and G3C cubins, normalized SASS hashes,
   instruction types, registers, stack/local memory, and constants.
3. Run the seven cases frozen in `PROTOCOL_G3B_ADVERSARIAL.md`, with exact final
   outputs and zero unsafe decisions at both direct stages.
4. Run full-process compute-sanitizer memcheck with leak checking.
5. Run 1,000 complete three-stage launches on alternating buffer sets; require
   invariant counts/hashes, zero overflow, and zero allocator residue.

Any failure is retained.  Do not time G3C until every item passes.  A later
timing protocol must freeze a same-contract comparator and measure the complete
pipeline, not only the FP32 kernel.

## Execution safety

Every GPU process uses physical GPU 0, the project lock, a 30-second empty
quiescence window, continuous 0.25-second monitoring, and an empty postflight
compute state.  If a never-owned process appears, terminate only this experiment
group, retain the failure, and do not touch the foreign process.
