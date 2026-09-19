# Decision G2A2: Triangular Production-Path Admission

Date: 2026-09-03

## Decision

**ACCEPT G2A2 and admit the triangular, chunk-bounded candidate to one G2B
full-scale resource/correctness smoke.** Two isolated executions reproduced the
frozen G2A oracle exactly, with identical stage counts and hashes.

This remains correctness evidence only. Neither diagnostic wall time is an
eligible performance observation.

## Frozen result

- Scheduled upper tiles: 2,080.
- Valid upper-triangle comparisons: 8,390,656 = 4,096 x 4,097 / 2.
- Per-stage capacity: 8,519,680 IDs, the worst-case 2,080 x 64 x 64 tile slots.
- Accepted unique upper pairs: 133,120.
- Expanded canonical directed pairs: 262,144.
- Upper raw uint64 hash:
  `036a4d84c1f1a10076287b9e4ad1cd950f979dac456093878c3cfd09d27ca959`.
- Directed raw uint64 hash:
  `da2c81605542e2079fbce2817cad3b42f651f33abd8e1b5b6000629068fb2f5d`.

Both executions produced the same work decomposition:

- direct INT8-certified accepts: 90,857 upper pairs;
- INT8 ambiguity: 102,277 upper pairs;
- FP32 accepts/rejects: 40,690 / 58,412 upper pairs;
- FP64 refinements: 3,175 upper pairs;
- unsafe INT8 decisions, unsafe FP32 decisions, duplicates, lower-triangle
  IDs, missing/extra IDs, and buffer overflows: zero.

## Mechanism result

The new schedule evaluates each unordered self-join pair once. It retains the
same numeric predicates as G2A and expands non-self results only after device
refinement. Per-batch storage is bounded by the number of tile elements, so
capacity is independent of output selectivity and no retry path is required.

This is a structural and correctness result, not evidence that the saved dense
work exceeds batching, atomic-compaction, transfer, and host-sort costs.

## Isolation and provenance

- Host: live `gpu-host-8` through `gpu-host-8`.
- GPU: physical GPU0, NVIDIA RTX PRO 6000 Blackwell Server Edition, capability
  12.0.
- Both runs held `@TENSORJOIN_ROOT@/.tensorjoin_gpu0_campaign.lock`, passed a
  30-second empty-GPU check, observed no foreign GPU0 PID, and left GPU0 empty.
- Runner SHA-256:
  `9eeaa7e21fa7137812e2fcccf09dde654e0a07ab4d95181f2e87bd6326f3f892`.
- Summary SHA-256:
  `f523909bb42c264cebd5677b94764d442d75696561bb29b391a28e15a167f1d3`.

## Next admitted action

Prepare a hash-receipted full Cifar60K float32 source plus method-specific
exact-widening artifacts. Then run one isolated, non-performance G2B smoke for
each exact method. Formal timing remains forbidden until all methods agree on
the complete canonical count and hash and the public timing adapters share the
same host-float32-array denominator.

