# Protocol G4B ragged-tail diagnostic

Frozen: 2026-09-03 before diagnostic execution

## Trigger and scope

The frozen G4B opportunity run failed its first Fashion-MNIST cell
(`N=1024,D=784,k=1`) while its SIFT-128 and CIFAR-GIST-512 cells passed. The
failure was isolated to the G3B boundary: eight direct accepts were outside the
independent oracle and four oracle pairs were absent from both the G3B direct
and ambiguity sets. G3C itself reported no unsafe decision among the pairs it
received.

The failed run is immutable negative evidence:

- result SHA-256: `c22da759dd8fdec44c66c3eddbe1e26733fa719f05ae6ae5519434319098090e`;
- isolation manifest SHA-256:
  `be7fc398b38d860057db5a3955801fdf546ca6b1c7b5277715fd0bf5acf8a17e`;
- raw log SHA-256: `e73b6291b9f48860003b9a88d54caf067689f82ef217af43520ee61615ac3968`;
- occupancy log SHA-256:
  `7fca1f96ff3ca1a5ba4f1d6023fe8323b2e8cf30597f039500d0045f8aa843ce`.

This diagnostic does not rerun, overwrite, or admit G4B. It performs one
in-process A/B on the identical failing source, threshold, oracle, stream, and
GPU to localize the first failing operation. It is correctness evidence only.

## Frozen hypothesis and one-variable A/B

The existing G3B kernel was validated only at `D=512`, a multiple of its
`BLOCK_K=64`. Its loop advances operand pointers by 64 but masks each load with
the static `offsets_k < K`. At `D=784`, the thirteenth block begins at 768, but
the static mask treats all 64 coordinates as valid rather than only 16. The A
operand crosses into the next row and the transposed B operand reads beyond its
allocation.

The suspect arm is the immutable accepted G3B-R1 kernel. The diagnostic arm
changes only final-block addressing and masking: it forms
`k = block_start + offsets_k` and masks `k < K` on both operands. Every other
arithmetic expression, constant, tile, warp/stage setting, input, and output
contract remains identical.

The diagnosis passes only when:

1. the suspect arm reproduces an unsound G3B classification;
2. the ragged-safe arm has exact stage partition, zero overflow or duplicate,
   zero direct false accept, and zero oracle false reject; and
3. both arms execute in the same isolated process on physical GPU1.

Passing identifies the first failing boundary as the G3B final-K-tile load
mask. It does not promote the patch or permit a performance claim. A future
G4B-R1 breadth gate must retain the original failure and independently freeze
and validate a generic kernel across all 27 cells.

## Frozen artifacts and execution card

| Artifact | SHA-256 |
|---|---|
| suspect G3B source | `057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec` |
| failed generic runner | `a8a550dd8dfb8d17ed1bc7f799b56c6272c14e80e748a80f3e32884a08801d5c` |
| diagnostic runner | `cc35e65c45892bcb0f432f23bb449cb48c31a59f67ea7ebc4ceba4cf7e190c7f` |
| isolation wrapper | `7a10a980b8294f92ed5f9c62572a7688715610236280609a5b63c4832f705662` |
| Fashion prepared vectors | `16149e1a1deaa2afeb205a0d22d49d21d1b784ca654851985603777e9c8b29d8` |
| failing oracle file | `6b2573b3441999438311706077087759cd8993664fd247d09099203509e06d19` |
| failing oracle raw IDs | `15132c49796e6e52059bb395acb0f84bb993fcebcd7fa4735112922720357311` |

```text
Host: gpu-host-8 / gpu-host-8
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU: physical GPU1 / GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3
Lock: @TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock
Shape: Fashion N=1024, D=784, target k=1
Quiescence: 30 seconds empty; monitor every 250 ms
Do-not-touch: every process outside the diagnostic child process group
Result: results/g4b_ragged_tail_diagnostic.json
Rollback: separate source only; no accepted keeper or dispatch is modified
```
