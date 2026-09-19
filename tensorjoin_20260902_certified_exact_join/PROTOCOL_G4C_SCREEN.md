# Protocol G4C screen: public dynamic-router anchors

Frozen: 2026-09-03 before any G4C timing

Experiment: `tensorjoin_20260903_g4c_public_anchor_screen`

## Preconditions

G4B-R1 passed 27/27 exact public cells and both cross-dataset selectivity gates.
Its immutable opportunity summary has SHA-256
`4cc069e20b52412d267e5bb945f11362df7dbcf443fbbd5c7a96d4ddc097d1de`.

The screen uses the deterministic subset defined in `DESIGN_G4C.md`: the
`N=4096,k=64` cell from SIFT-128, CIFAR-GIST-512, and Fashion-MNIST-784. No
timing result was observed before this selection or protocol was frozen.

## Frozen sources and evidence

| Artifact | SHA-256 |
|---|---|
| `DESIGN_G4C.md` | `d1c288b1ddcdd1a1fc5fb4828958c49169a73d4313d5bf84d8d43055c76400a9` |
| `PROTOCOL_G4B_R1_OPPORTUNITY.md` | `9f7a44ee7bee8a0a656ddfa3cfeb88cef3802d432829eba6018e03b58ea0fe1a` |
| `DESIGN_G4B_R1.md` | `c6d804451e36b5d5688777bc7099b43c2f4f8ea80d1cbee95c7adfe6194bf575` |
| G4B-R1 SIFT result | `d987a0271d7945883328e0c95a3028d10276d1ed7bf8686016db4788cb743204` |
| G4B-R1 CIFAR result | `1bb98444527f64aba34a59df3b9ae96184b9477560c8b19471b1e2c25e00a09b` |
| G4B-R1 Fashion result | `4bf374d4b8e506f8c0fc0579c7f342acaf4852c99eb15f016fd8b1add2a90cbb` |
| ragged-safe G3B kernel | `439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6` |
| G3C-B-R1 source | `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d` |
| FP64 source | `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8` |
| G4C timing runner | `a9fac56d17becca52eeedc3dafa99e3bd3d49e4bcfc8a9ce40a2a15f952b6f15` |
| G4C isolation wrapper | `dfc7bad0c87cf68b0db9270f3f1e02b804bac9b266dd5c41b21268af22df5338` |
| G4C screen summarizer | `d487ff858ee4af76b67331b8ad4e2e392da5fbf079dd7c0b659dd9474b6ad251` |

## Execution card

```text
Host: gpu-host-8 / gpu-host-8
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU: physical GPU1 / GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3
Lock: @TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock
Cells: SIFT/CIFAR/Fashion at N=4096,k=64, native D
Processes: two fresh processes/cell, orders KC then CK
Per process: 10 warmups, 50 retained calls, 200 sustained calls/variant
Clock: natural dynamic clocks; never modified
Quiescence: 30 seconds empty; monitor every 250 ms
Do-not-touch: every process outside the G4C child process group
Validation: exact oracle equality before and after timing
Rollback: screen-only files; no active dispatch or accepted source modified
```

The declared timing scope and variants are exactly `DESIGN_G4C.md`. Each fresh
process uses a fresh Triton cache. Results preserve all ordered samples,
per-variant percentiles, medians, sustained totals, dynamic stage counts, GPU
state, pre/post correctness, and source/evidence hashes.

## Gate

The screen admits formal work only if:

1. all six slots are isolated, complete, and exact before and after timing;
2. at least two of three datasets exceed 1.10x on the paired geometric mean of
   process median speedups; and
3. the same at least two datasets exceed 1.10x on paired geometric mean
   sustained speedup.

All datasets proceed to formal measurement if the aggregate gate passes,
including any losing anchor. Screen values are not paper claims. A screen pass
does not substitute for runtime binary audit, sanitizers, or formal timing.

## Commands

```bash
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
for dataset in sift128 cifar_gist512 fashion784; do
  for process_id in 0 1; do
    @TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
      src/run_g4c_screen_isolated_gpu1.py \
      --dataset "$dataset" --process-id "$process_id" --attempt 0 \
      > "raw/g4c_screen_${dataset}_p${process_id}_wrapper.log" 2>&1 || exit $?
  done
done
@TENSORJOIN_ROOT@/isaacsim6/env/bin/python src/summarize_g4c_screen.py \
  > raw/g4c_screen_summary.log 2>&1
```
