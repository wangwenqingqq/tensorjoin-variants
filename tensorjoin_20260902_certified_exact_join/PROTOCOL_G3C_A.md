# Protocol G3C-A: host opportunity gate for a certified FP32 middle stage

Date frozen: 2026-09-03

Experiment: `tensorjoin_20260903_g3c_a_host_fp32_interval`

Status: frozen before execution.

Runner: `src/run_g3c_a_host_fp32_interval.py`, SHA-256
`4ba20c4ea47b968cad8cd3b4703399bb1236ef561fde6b31c1384ccb65af4c4d`.

## Question and claim boundary

G3B-R1 proves an INT8/INT32 Tensor Core first-stage certificate but sends all
102,079 remaining upper-triangle pairs to FP64.  G3C-A asks one cheap question:
can an outward FP32 squared-distance interval resolve at least 90% of exactly
those pairs without an unsafe accept or reject?

The G3B-R1 GPU artifact is used only to reproduce and enumerate the frozen
ambiguous set.  All G3C-A interval evaluation is on the host.  Passing G3C-A
admits an independent GPU implementation; it does **not** prove GPU generated
code, safety, portability, or performance.

## Frozen task

- Input: the G2A CIFAR-10 GIST subset, shape `4096 x 512`, stored float32.
- Join: exact Euclidean self-range join, upper triangle including self.
- Squared threshold: the exact float64 value in frozen metadata.
- Oracle: frozen direct-difference float64 oracle.
- Source stage: exact live reproduction of G3B-R1 ambiguity count `102079` and
  sorted raw-u64 SHA-256
  `6f89c3126a11ce0c152a0596fdd26138a3e1e17359f6d67e58633eccbb5ada42`.
- Quality target: exact final output; direct decisions must have zero false
  accepts and zero false rejects.

## Frozen host interval

For each stored float32 coordinate pair, let

```text
q_k = RN_f32(x_k - y_k)
t_k = RN_f32(q_k * q_k)
s_hat = a float32 reduction of t_k
u = 2^-24
tiny = smallest positive normal float32
```

The host opportunity model uses:

```text
e_k       = u * (abs(x_k) + abs(y_k)) + tiny
E_source  = sum_k (2 * abs(q_k) * e_k + e_k^2)
E_square  = u * sum_k q_k^2 + D * tiny
gamma     = (D - 1) * u / (1 - (D - 1) * u)
E_reduce  = gamma * sum_k abs(t_k) + (D - 1) * tiny
E          = nextafter(E_source + E_square + E_reduce, +infinity)
interval   = [nextafter(max(s_hat-E, 0), -infinity),
              nextafter(s_hat+E, +infinity)]
```

The `gamma_(D-1)` term deliberately covers a serial worst-case count of
float32 additions rather than assuming a favorable reduction tree.  The
`tiny` terms cover a future FTZ-aware implementation domain at this opportunity
stage.  The exact GPU implementation must separately audit its actual
instructions and outward computation of the radius.

Acceptance uses the lower outward float32 endpoint of the squared threshold;
rejection uses the upper endpoint.  Pairs whose interval crosses the threshold
remain for FP64.  A fixed `1e-3` guard is recorded only as a non-admission
diagnostic against the previous stage.

## Acceptance and stop rule

G3C-A passes only if all conditions hold:

1. the source and oracle hashes match the frozen contract;
2. the live G3B-R1 ambiguity reproduces its count and hash with zero overflow;
3. every exact float64 source distance lies inside the proposed interval;
4. direct FP32 decisions have zero unsafe accepts and rejects;
5. accept, reject, and FP64 sets partition all 102,079 source pairs;
6. at most `10207` pairs, i.e. at most 10%, remain for FP64.

If any condition fails, stop G3C.  Do not implement or time the GPU candidate.
If all pass, freeze G3C-B as a separate source and require two isolated
validations, generated-code audit, the existing seven adversarial cases,
compute-sanitizer memcheck, and a 1,000-launch sustained test before any timing.

## Execution safety

The only GPU work in G3C-A is the already-audited G3B-R1 enumeration kernel.
Execution on physical GPU 0 requires the project lock, a 30-second empty
quiescence window, continuous occupancy monitoring, no overlap with a
never-owned process, and empty postflight compute state.  Never terminate a
foreign process.  No duration from this gate is performance evidence.
