# Frozen G16 GPU boundary-validation campaign

Record: `tensorjoin_20260907_frozen_boundary_g16d512`.
Created before implementation/measurement on 2026-09-07.

## Question and invariant

Validate the reviewed numerical contract on the intended GPU using the exact
retained G16 metadata, stage-1, stage-2 and FP64-terminal cubins. A new host
validation harness is explicitly a different artifact from the historical
timing runner. It must load those cubins without recompiling or modifying them.
No result here is a new latency comparison or a whole-program formal proof.

Frozen semantics: N=60000 address specialization, D512 finite stored FP32,
coordinate magnitude <=1, main epsilon=161/256, T=25921/65536, inclusive fixed
terminal predicate. Preserve numeric-equality, self/directed pair semantics,
and the explicit threshold/row-size caveats from the reviewed addendum.

## Gates and stop rules

1. Verify SSH topology, host/account/path, nearest instructions, exact code
   hashes and toolchain. Select an idle target GPU (prefer historical GPU2),
   acquire the existing per-GPU campaign lock, and reject concurrent occupancy.
   Never kill others, change clocks/power mode, or restart services.
2. Verify a new Driver-API launch wrapper's argument ABI, block/shared-memory
   configuration and scratch parameters against the frozen PTX/metadata. Hash
   loaded code objects before and after. No JIT-produced replacement is allowed.
3. On identical synthetic inputs, compare every tested early decision and the
   complete selected-pair cascade output against direct launches of that same
   terminal. Retain exact CPU semantic witnesses separately; another FP64 sum
   is not the deciding oracle. Fail at the first unexplained mismatch.
4. Include zero/signed-zero, subnormal/normal, exponent spread, equal/adjacent
   coordinates, quantized cancellation, threshold ties/neighbors, dense/empty
   selected-tile output, ragged final tile, pointer/order variation, and bounded
   replay stress. Check actual metadata enclosure on retained sample rows with
   rational arithmetic. Exercise graph replay if the wrapper supports capture.
5. After boundary correctness, check the frozen full public input's canonical
   output against its retained count/hash, using the same cubins. This is output
   validation, not a new performance claim or external-baseline comparison.
6. Run memcheck and synccheck on a bounded boundary workload. Treat tool output,
   correctness, stress, graph, and full output as separate gates. Preserve all
   errors; do not silently relabel a diagnostic or sanitizer failure as a pass.

No deliberate out-of-bounds access is authorized: the known row 2^22 address
wrap is tested as a host admission rejection, never launched. Queue capacity
is frozen at 4096*64*64 and normal per-batch work is bounded below that size;
test the new harness's prelaunch rejection of invalid workloads, not memory
corruption. This does not retrofit an upper-size guard into the original API.

## Execution card (resolved by live preflight before launch)

- Host/topology: local Mac -> configured tiaoban ProxyCommand -> gpu-host-8.
- Account/path: verify live; historical root and
  `@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join`.
- Environment: historical `@TENSORJOIN_ROOT@/isaacsim6/env/bin/python`;
  verify torch, CUDA, driver and sanitizer versions live.
- GPU: not selected until idle inventory; prefer GPU2; one campaign lock.
- Version: immutable cubin/source hashes, not an assumed Git branch.
- Dataset: synthetic bounded fixtures, then retained CIFAR-GIST 60000x512 FP32.
- New code/logs/results: this separate campaign directory only.
- Command/PID/log: recorded by the guarded launcher before each run.
- Do not touch: other processes, old evidence/drafts, Overleaf, GPU settings.
- Expected resource delta: no kernel change; bounded validation allocations
  (target below 4 GiB device memory), correctness-only host overhead.
- Rollback: stop only this campaign's child process group; preserve all records.

## Promotion boundary

Accept only the individually completed gates under their actual tested scope.
If blocked by connectivity, occupancy, driver incompatibility or mismatches,
retain the exact failure and continue useful local preparation without claiming
GPU completion. Novelty, arbitrary-radius support, all-size safety, exact-real
output and the historical mixed-artifact 5.083x claim are not promoted here.
