# D3 Decision

Date: 2026-09-03

## Conclusion

**PASS scientific-tensor certificate transfer, but STOP the current HSI
performance promotion before formal timing.** The exact numerical mechanism
extends to spatially disjoint hyperspectral patches; the unchanged two-stage
execution again loses its required margin at the high-output radius.

## Strong evidence

- Open artifact: verified Indian Pines `145x145x220` source; deterministic
  extraction produced 20,449 normalized `1,984`-D patch vectors, cache SHA-256
  `b95402cec4fd6037fea32934b8b8170ab51c9baa866d7c392737c06b6398adb2`.
- Source-pixel-disjoint geometry: exact 1/64 results per query, zero certificate
  containment or classification error, with 0.0222%/0.5515% ambiguity.
- D3B correctness: candidate and keeper both exactly match the FP64 oracle;
  zero unsafe direct accepts, duplicates, or overflows.
- Cheap performance result: target 1 is diagnostically 3.631x, but target 64 is
  only 1.191x (466.992/392.192 microseconds), below the frozen 1.35x rule.

## Boundary

The D3B numbers are smoke diagnostics, not a formal campaign. Memcheck,
1,000-launch stability, and eight-process paired timing were intentionally not
run after the decisive cheap kill. Do not report a positive HSI speedup. The
valid transferable result is exact certificate geometry.

## Decision

Retain D3A as breadth evidence and D3B as negative execution evidence. Do not
retune the D=1,984 tile, guard, or gate on this cache. The repeated target-64
failure across video and HSI points to precision/refinement routing rather than
another representation-specific kernel tune.
