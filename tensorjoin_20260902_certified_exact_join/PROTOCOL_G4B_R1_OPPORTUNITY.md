# Protocol G4B-R1: ragged-safe public breadth opportunity matrix

Frozen: 2026-09-03 before G4B-R1 GPU execution

Experiment: `tensorjoin_20260903_g4b_r1_public_breadth_opportunity`

## Admission rationale and scope

G4B was rejected on the first `D=784` cell. The separate, isolated in-process
diagnostic localized the first failing boundary to the final partial-K load
mask and proved a one-variable ragged-safe arm sound on the identical failing
cell. G4B-R1 therefore reopens only the implementation-correction condition in
`DECISION_G4B.md`; it does not weaken or tune any data, threshold, exactness, or
selectivity gate.

G4B-R1 reruns all 27 cells from scratch with one immutable ragged-safe G3B
kernel source. It is correctness and routing-geometry evidence, not timing,
throughput, end-to-end, external-system, or novelty evidence.

## Frozen predecessor and diagnosis evidence

| Artifact | SHA-256 |
|---|---|
| `DECISION_G4B.md` | `452bd4155677d896486461e3511d81582b020668f0a19f9bb65b685e840cb62e` |
| rejected Fashion G4B result | `c22da759dd8fdec44c66c3eddbe1e26733fa719f05ae6ae5519434319098090e` |
| rejected Fashion isolation manifest | `be7fc398b38d860057db5a3955801fdf546ca6b1c7b5277715fd0bf5acf8a17e` |
| admitted ragged diagnostic result | `dce93811383fd81fe4144a07ee373650eec1c733c2331d4d7ffd9b13eb3fc5a8` |
| admitted ragged diagnostic manifest | `d8ff9266a67b3ab6d2ccbf80b4e18aaf8a16d9064b92d633ffb8febc35660ce4` |
| `PROTOCOL_G4B_RAGGED_DIAGNOSTIC.md` | `66e00e58d7022358cb1298e785bf80d8702d466070a1e0118d1be3f27dd78090` |

The R1 runner refuses to start unless both diagnostic hashes match and both the
in-process diagnosis and isolation admission flags are true.

## One frozen mechanism change

G4B-R1 inherits `GATE0_G4B.md`, `DESIGN_G4B.md`, and every accepted G3B-R1,
G3C-B-R1, and G2A arithmetic constant. `DESIGN_G4B_R1.md` defines the only
change: each INT8 loop iteration forms `k=block_start+offsets_k` and masks both
operands with `k<K`. This is identical to the rejected kernel on full 64-wide K
blocks and zero-fills the 48 invalid lanes in the final `D=784` block.

| Artifact | SHA-256 |
|---|---|
| `GATE0_G4B.md` | `19fada127758d30eaa070bc1c60651e3c9418dcad011833b328a066b16715586` |
| `DESIGN_G4B.md` | `e08e184de64893040b05b07b79221a2e8d59f66a993cdadb2bdbb72d1f6ea0b6` |
| `DESIGN_G4B_R1.md` | `c6d804451e36b5d5688777bc7099b43c2f4f8ea80d1cbee95c7adfe6194bf575` |
| ragged-safe kernel | `439058576f156e75074995a29b048448ae480c5dfc72f2c2b537b23f8e79e9f6` |
| R1 dataset runner | `c1d539458233fb3268149e74847d8e6a5c8f4e3a3400954e7a5ee44a74f36234` |
| R1 isolation wrapper | `1a95e653498cd79a933ae7a87e79faccdd7d9f042094028ab7661c0d350ab085` |
| R1 summarizer | `0f5616cac44b6813db92be9479c7d07d35ccc52645e684b2ef05a2948cb79f67` |
| FP64 source | `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8` |
| G3B constants/reference source | `057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec` |
| G3C-B-R1 source | `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d` |

## Frozen data and oracle contract

Reuse the already constructed 27 direct-FP64 oracles without modification:

- prepared-data manifest SHA-256:
  `f68bd147f64a96a7b484fbe1661a5e5859614fbc151470f98fff9116b589c180`;
- exact-oracle manifest SHA-256:
  `4447f166505954c1d84f606697a5a8e23035544654861d480338b4c97dfa58c7`;
- one PCG64 seed `20260903` permutation/source, nested
  `N={1024,2048,4096}`;
- native `D={128,512,784}`, no padding/projection/truncation; and
- target degree `k={1,16,64}` with strict FP64 mid-gap thresholds.

Every input, row-prefix, metadata, oracle file, and raw oracle hash is rechecked
inside the R1 process. The prior G4B output is not used as an oracle or admitted
result.

## Execution card and isolation

```text
Host: gpu-host-8 / live hostname gpu-host-8
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
GPU: physical GPU1 / GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3
Lock: @TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock
Order: sift128, cifar_gist512, fashion784
Within dataset: N=1024,2048,4096 then k=1,16,64
Triton cache: one fresh R1 cache directory per dataset
Quiescence: 30 seconds empty before each process; monitor every 250 ms
Do-not-touch: every process outside the R1 child process group
Rollback: separate R1 sources/results only; no keeper or dispatch modification
```

Each G3C launch uses the measured G3B ambiguity count; each FP64 launch uses the
measured G3C residual count. Worst-case buffers are capacity, not executed work.
A foreign GPU process invalidates the dataset slot and terminates only the R1
child process group.

## Per-cell and aggregate gates

All 27 cells must pass exact stage partitions, zero unsafe G3B/G3C decisions,
zero duplicates/invalid/lower IDs/overflow, and exact count/hash/element equality
with the independent upper-ID oracle.

The unchanged aggregate conditions are:

1. 27/27 cells exact and every required hash recorded;
2. at `N=4096`, SIFT and Fashion each have at least two of three densities with
   FP64 residual no greater than 25% of G3B ambiguity; and
3. at `N=4096`, SIFT and Fashion each have at least one of `k=1,16` with G3B
   ambiguity no greater than 25% of all upper pairs.

The R1 runner stops at the first exactness failure. The summarizer runs only
after all three clean manifests exist. Only a full aggregate pass permits a
separately frozen generic timing subset.

## Commands

```bash
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join

for dataset in sift128 cifar_gist512 fashion784; do
  @TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
    src/run_g4b_r1_opportunity_isolated_gpu1.py --dataset "$dataset" \
    > "raw/g4b_r1_opportunity_${dataset}_wrapper.log" 2>&1 || exit $?
done

@TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
  src/summarize_g4b_r1_opportunity.py \
  > raw/g4b_r1_opportunity_summary.log 2>&1
```
