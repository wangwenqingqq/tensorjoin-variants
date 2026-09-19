# Decision G4B: reject the first public-breadth implementation

Date: 2026-09-03

## Decision

**G4B fails its frozen exactness gate and is not admitted.** SIFT-128 and
CIFAR-GIST-512 completed all nine cells exactly, but the first Fashion-MNIST
cell (`N=1024,D=784,k=1`) produced eight unsafe G3B direct accepts, four unsafe
G3B direct rejects, eight final extras, and four final misses. The run stopped
immediately, so the frozen 27-cell opportunity gate was not applied and no G4B
performance work is allowed.

This is an implementation-boundary failure rather than negative evidence
against the analytic certificate itself. The failure nevertheless remains a
hard rejection of the tested source and cannot be removed or overwritten.

## First failing boundary

An isolated in-process A/B on the identical Fashion source, threshold, oracle,
GPU, and stream localized the error to the G3B final-K-tile mask:

| Arm | Direct | Ambiguous | Unsafe accept | Unsafe reject | Sound |
|---|---:|---:|---:|---:|---|
| frozen G3B-R1 source | 1,471 | 168 | 8 | 4 | no |
| one-variable ragged-safe mask | 1,456 | 166 | 0 | 0 | yes |

The frozen G3B source advances its load pointers by 64 but tests the static
`offsets_k < K`. At `D=784`, the final block begins at coordinate 768; only 16
lanes are valid, while the suspect mask enables all 64. The ragged-safe arm
changes only the mask/address index to `block_start + offsets_k < K`. It changes
29 direct and 64 ambiguous classifications and removes every unsafe decision.

The diagnostic result is
`results/g4b_ragged_tail_diagnostic.json`, SHA-256
`dce93811383fd81fe4144a07ee373650eec1c733c2331d4d7ffd9b13eb3fc5a8`.
Its isolation manifest admitted a clean physical-GPU1 slot with no foreign or
postflight process.

Compute Sanitizer reported zero allocation-level errors on a minimal suspect
tile (`raw/g4b_ragged_memcheck.log`, SHA-256
`75d788b324fdd82b54f805cd6fff298392df2988cc589ca713e59e8da81cd979`).
This does not clear the logical tensor-boundary error: PyTorch's caching
allocator can place adjacent tensors inside a larger valid CUDA allocation, so
the sanitizer result is recorded as inconclusive rather than contradictory.

## Preserved positive and negative evidence

- SIFT-128: 9/9 cells exact. At `N=4096`, G3B ambiguity was 0.0083%, 0.1389%,
  and 0.4745% of all upper pairs for `k=1,16,64`; the FP64 residual was 0.2882%,
  0.1631%, and 0.2085% of G3B ambiguity.
- CIFAR-GIST-512: 9/9 cells exact under the same failed generic runner.
- Fashion-MNIST-784: the first cell failed at G3B and execution stopped as
  required.
- No result above is timing evidence. No foreign process was killed or altered.

## Reopen condition

A separate G4B-R1 correctness gate may reopen breadth evaluation only if it:

1. uses a new immutable ragged-safe kernel source rather than editing G3B-R1 or
   overwriting G4B;
2. reruns all 27 cells from scratch, including the 18 multiple-of-64 control
   cells;
3. preserves the original G4B result, isolation evidence, and diagnostic; and
4. applies the same source rows, thresholds, exactness checks, and selectivity
   thresholds without dataset-specific tuning.

Only a clean G4B-R1 aggregate pass can reopen generic timing.
