# Protocol G4C formal: cross-dataset dynamic-router anchors

Frozen: 2026-09-03 before formal execution

Experiment: `tensorjoin_20260903_g4c_public_anchor_formal`

## Admission and claim scope

The six-slot screen was exact and isolated. All three datasets exceeded the
predeclared 1.10x threshold on both paired process-median and sustained
geometric means: SIFT 1.260x/1.270x, CIFAR 2.315x/2.321x, and Fashion
1.702x/1.717x. These remain screen observations, not paper claims.

The formal campaign retains every `N=4096,k=64` anchor chosen before the
screen. It measures the resident-GPU, host-dispatched dynamic-count operator
scope defined in `DESIGN_G4C.md`; it is not ingest-inclusive or
output-materializing end-to-end performance and is not an external-system
comparison.

## Frozen screen evidence and sources

| Artifact | SHA-256 |
|---|---|
| `DESIGN_G4C.md` | `d1c288b1ddcdd1a1fc5fb4828958c49169a73d4313d5bf84d8d43055c76400a9` |
| `PROTOCOL_G4C_SCREEN.md` | `07e48e01e942159af5551c198b01df0d61f9d70339bfdbade4725c376457f4e8` |
| screen summary | `3bc1c7aafb1488650eb68088c8b25b72085aeb308022e1e9fc63a49cca716037` |
| screen runtime/generated-code audit | `a19142fe8259cd63db343233835daf189a62a95be4e0ee73debd2e107bfe3542` |
| formal-only timing runner | `1f4d2f2e05a53a486d80f7d21bf163bfe4152dcf94c5fa133bdb6d4ea5726c05` |
| formal isolation wrapper | `194e51eb6dc2963f22c2604444e50d45b9037cb614e472493d56dc4bae8a011c` |
| formal runtime auditor | `3c17ec8501027f95e43a1bd0eb4ba29957f3f1ab644055a0136a39289e5ffdcb` |
| formal summarizer | `df6bc6a571f575ad0c3226442fdd6c1b164653e82b10700b7b0ba5daa80d6f3a` |
| ragged-safe G3B kernel | `439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6` |
| G3C-B-R1 source | `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d` |
| FP64 source | `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8` |

The formal runner refuses execution unless the exact screen summary and audit
hashes match and both admission flags are true.

## Runtime-selected binary contract

The screen observed byte-identical cubins across both fresh processes within
each dataset. All audited kernels target `sm_120a`, have zero stack/local bytes
and zero static LDL/STL. G3B contains low-precision MMA, G3C contains no
low-precision MMA or FP64 arithmetic, and the fallback contains FP64 arithmetic.

| Dataset | ragged-safe G3B cubin | G3C FP32 cubin | FP64 cubin |
|---|---|---|---|
| SIFT-128 | `76c90286c79fc4494e3152aa719bc6e9d621dc3d1d7fb6377a0cd6ef4a00f145` | `2db627fb239aab1ec8b8eb186c875ac5a10a8ba19f91971c23ab079bdc878056` | `60547d595b0b771d22339421140ddd801053e5aaa586ef857795c253c07df8d2` |
| CIFAR-GIST-512 | `8962ffab923c0d9dc7febf2eab5c71aa022cf46839f83b244d31f91f2f7b772a` | `db750295875f8363679cfe9f9762cef884e1ecd371e3e419fd500d91c3a20670` | `ff0572a7eefcb6ccba91c7da0020d370e45d7f8f7c03d3a55e8f5ca53c477c50` |
| Fashion-784 | `0ddc66281d023265069f319773501077264989523497e06efa1b412b915bea87` | `c00e2a07482c55a4200e5651061ca1d70b762aa41a161587d40473bb017f7f42` | `decb547a9450e0225dab254a80172702527ceefd03f6e8ddd35cf0416dd39dbe` |

Every formal fresh-cache process must reproduce its dataset's three exact
cubin hashes. The formal runtime audit occurs before statistical summarization.

## Execution card

```text
Host: gpu-host-8 / gpu-host-8
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU: physical GPU1 / GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3
Lock: @TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock
Cells: SIFT/CIFAR/Fashion, N=4096,k=64, native D
Processes: eight fresh processes per dataset
Orders: KC, CK, KC, CK, CK, KC, CK, KC
Per process: 20 warmups, 200 retained calls, 1000 sustained calls/variant
Clock: natural dynamic clocks; never modified
Quiescence: 30 seconds empty per slot; monitor every 250 ms
Do-not-touch: every process outside the formal child process group
Validation: exact oracle equality before and after timing
Rollback: separate formal evidence only; no active dispatch modified
```

Each call executes actual dynamic D2H count discovery and uses the discovered
count as the next launch size. All 200 observations preserve order, variant,
elapsed time, and dynamic counts. A foreign process invalidates the slot and
terminates only the campaign child group; retry requires a new attempt suffix
and the interrupted evidence remains excluded.

## Formal estimators and gates

For each dataset separately:

- primary estimator: geometric mean of eight paired process-median speedups;
- uncertainty: 20,000 process-level bootstrap resamples of log speedup, seeds
  `20260903`, `20260904`, and `20260905` in dataset order;
- sustained estimator: geometric mean of eight paired 1000-call mean ratios;
- retain marginal p10/median/p90 for both variants.

A dataset passes only if all dynamic counts are invariant, one runner/protocol
hash is present, the candidate wins all eight process medians, the 95% bootstrap
lower bound is at least 1.10x, the candidate wins all eight sustained ratios,
and their sustained geometric mean is at least 1.10x.

Cross-dataset promotion requires all three dataset gates. A failure retains the
per-dataset result and prevents a universal cross-dataset claim; it does not
permit deleting a losing anchor.

## Commands

```bash
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
for dataset in sift128 cifar_gist512 fashion784; do
  for process_id in 0 1 2 3 4 5 6 7; do
    @TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
      src/run_g4c_formal_isolated_gpu1.py \
      --dataset "$dataset" --process-id "$process_id" --attempt 0 \
      > "raw/g4c_formal_${dataset}_p${process_id}_wrapper.log" 2>&1 || exit $?
  done
done
@TENSORJOIN_ROOT@/isaacsim6/env/bin/python src/audit_g4c_formal_runtime.py \
  > raw/g4c_formal_runtime_audit.log 2>&1
@TENSORJOIN_ROOT@/isaacsim6/env/bin/python src/summarize_g4c_formal.py \
  > raw/g4c_formal_summary.log 2>&1
```
