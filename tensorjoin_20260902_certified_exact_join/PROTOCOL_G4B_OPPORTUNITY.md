# Protocol G4B: public breadth opportunity matrix

Frozen: 2026-09-03 before oracle construction or GPU execution

Experiment: `tensorjoin_20260903_g4b_public_breadth_opportunity`

## Decision scope

This gate answers one question: does the existing G3B-R1 -> G3C-B-R1 -> direct
FP64 exact router retain both soundness and material selectivity across the 27
public dataset/scale/density cells frozen in `GATE0_G4B.md`?

The output is correctness and routing geometry. It is not kernel timing,
operator timing, end-to-end timing, an external-system comparison, or a novelty
claim. Failure stops the generic timing stage; no dataset-specific tuning may
rescue this gate.

## Execution card

```text
Host: gpu-host-8 (live hostname gpu-host-8)
Path: @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join
Version: immutable source hashes below; directory is not a Git checkout
Env: @TENSORJOIN_ROOT@/isaacsim6/env/bin/python
CPU oracle: nice -n 10; OMP/OPENBLAS/MKL/NUMEXPR threads all fixed to 1
GPU: physical GPU1, UUID GPU-daa88abc-ce4a-aa1c-c896-ae528bf9bce3
Lock: @TENSORJOIN_ROOT@/.tensorjoin_gpu1_campaign.lock
Dimension: native D=128, 512, 784; no projection/padding/truncation
Dataset: one frozen PCG64(20260903) permutation/source; nested N prefixes
Thresholds: strict direct-FP64 mid-gap thresholds for target k=1,16,64
Logs: raw/g4b_exact_oracle.log and raw/g4b_opportunity_*.log/occupancy.jsonl
PID: isolation wrapper records root and descendant GPU PIDs
Do-not-touch: every process not descended from the wrapper child
Validation: independent oracle equality and the four frozen aggregate gates
Rollback: no active dispatch is modified; failed outputs are retained/excluded
```

Live pre-freeze audit on 2026-09-03T05:36:08Z found all eight GPUs at 14 MiB
and no compute application. Each GPU dataset slot nevertheless requires a new
30-second empty-device quiescence window and continuous 250 ms process
monitoring. A foreign process terminates only the G4B child process group and
invalidates that slot.

## Frozen environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, physical index 1.
- NumPy 2.3.1, SciPy 1.17.0, PyTorch 2.11.0+cu130, Triton 3.6.0.
- Dataset execution order: `sift128`, `cifar_gist512`, `fashion784`.
- Within each dataset: `N=1024,2048,4096`; within each scale: `k=1,16,64`.
- One fresh isolated process and one fresh Triton cache per dataset.
- Dynamic stage counts are read after device synchronization and become the
  exact launch sizes of G3C and FP64. Worst-case capacity is storage only.

## Frozen sources and prepared data

| Artifact | SHA-256 |
|---|---|
| `GATE0_G4B.md` | `19fada127758d30eaa070bc1c60651e3c9418dcad011833b328a066b16715586` |
| `DESIGN_G4B.md` | `e08e184de64893040b05b07b79221a2e8d59f66a993cdadb2bdbb72d1f6ea0b6` |
| `src/prepare_g4b_public_data.py` | `e8510b4232a10d3cef66803f2b71b45ead635f817875053635bd2a06fa873880` |
| `src/build_g4b_exact_oracles.py` | `b465b1e42d58d7ab0b0d0dc032b52e197e2fc0c019afdb8b03378ea4fbfc00a3` |
| `src/run_g4b_public_opportunity.py` | `a8a550dd8dfb8d17ed1bc7f799b56c6272c14e80e748a80f3e32884a08801d5c` |
| `src/run_g4b_opportunity_isolated_gpu1.py` | `d5d3646b1c34bae3fb867819dacdf655457270deb0b769669d427a50ee9a50c4` |
| `src/summarize_g4b_opportunity.py` | `277b19e2d0f0256df2fd5f0eec8688ae499c85ec58d0ee7eb7e3031029045542` |
| FP64 source `src/run_g2a_tensorjoin.py` | `84f611fa029e0ce798c3945fad54600598bc7a708f630ebd5fc07fe7dac1e6d8` |
| G3B-R1 source `src/run_g3b_r1_gpu_analytic_certificate.py` | `057245c5288763702e7b916c6739980d07b622d595bb9d946fa14011437a67ec` |
| G3C-B-R1 source `src/run_g3c_b_r1_gpu_cascade.py` | `637e9455fdcdffcc8cddbed625ef74ef371f117876bc3f25977a79b4a95f706d` |
| prepared-data manifest `data/g4b_public/manifest.json` | `f68bd147f64a96a7b484fbe1661a5e5859614fbc151470f98fff9116b589c180` |
| preparation log `raw/g4b_public_data_prepare.log` | `5a32e708f631dbdc09c8fe0fc6107d7311ad036c9bea3b1b07ca88373f46bbde` |

| Dataset artifact | SHA-256 |
|---|---|
| SIFT1M base fvecs, read-only source | `21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816` |
| `sift128/vectors_f32.npy` | `91648432b84009381b22aedc3fba487e787ee4e659ed0aaf5392f8e1042d9126` |
| `sift128/source_row_ids_u32.npy` | `fcd014f989df767832e5028eda18607d5e726dcdfd5d9f17abbad3c1ac739fa5` |
| Cifar60K base fvecs | `a7170faaa80a072cd603ed472104049ead87fbaff224e94a529021d161f8aea4` |
| `cifar_gist512/vectors_f32.npy` | `cea1987b6a07df43a71d5361afa5f1630c8f1abbbc44493d60af0898e6394702` |
| `cifar_gist512/source_row_ids_u32.npy` | `b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c` |
| official Fashion train gzip | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `fashion784/vectors_f32.npy` | `16149e1a1deaa2afeb205a0d22d49d21d1b784ca654851985603777e9c8b29d8` |
| `fashion784/source_row_ids_u32.npy` | `b32024844cb503aebce28ddde3ce1309db51d6afeac773ea2debd3e8d2d2972c` |

The official Fashion artifact must also retain MD5
`8d4fb7e6c68d591d4c3dfef9ec88bf0d`. Dataset bytes remain outside any future
redistributable code bundle unless upstream terms are separately verified.

## Oracle contract

For each dataset and scale, load the frozen float32 prefix and widen it exactly
to float64. Compute one SciPy condensed `sqeuclidean` distance array. For each
target degree, partition at `N*k/2`, expand any equal-value tie through the next
strict gap, and choose a representable float64 midpoint. Materialize sorted
canonical upper IDs including all self pairs.

Every cell records vector-prefix, source-row-prefix, threshold metadata, oracle
file, and raw oracle-ID hashes. Any missing strict gap, invalid number, hash
mismatch, or output-count mismatch rejects the gate before GPU execution.

## GPU contract and checks

The generic runner changes only host-side dimension parameterization. Device
arithmetic constants, block shapes, stage predicates, and the exact FP64
fallback source remain the accepted G3B-R1/G3C-B-R1/G2A implementations.

For every cell, require:

1. `D*127^2 <= 2^24` and finite prepared inputs;
2. exact G3B and G3C stage partitions;
3. no unsafe G3B direct accept or reject;
4. no unsafe G3C direct accept or reject;
5. no duplicate, invalid, lower-triangle, or overflow event; and
6. sorted final upper IDs equal the independent oracle in count, SHA-256, and
   element-wise comparison.

The runner stops its dataset at the first correctness failure, writes all
completed cell evidence, and exits nonzero. The aggregate is not executed
unless all three isolation manifests are clean and all 27 cells are present.

## Aggregate opportunity gate

Apply the exact four conditions frozen in `GATE0_G4B.md`:

1. 27/27 cells exact and structurally valid;
2. every required source, row-prefix, oracle, threshold-metadata, and final
   output hash present;
3. at `N=4096`, both SIFT and Fashion have at least two of three densities with
   residual FP64 count no greater than 25% of G3B ambiguity; and
4. at `N=4096`, both SIFT and Fashion have at least one of `k=1` or `k=16` with
   G3B ambiguity no greater than 25% of all upper pairs.

Only an aggregate pass permits a separately frozen generic timing subset.

## Commands

```bash
cd @TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 nice -n 10 \
  @TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
  src/build_g4b_exact_oracles.py \
  > raw/g4b_exact_oracle.log 2>&1

for dataset in sift128 cifar_gist512 fashion784; do
  @TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
    src/run_g4b_opportunity_isolated_gpu1.py --dataset "$dataset" \
    > "raw/g4b_opportunity_${dataset}_wrapper.log" 2>&1 || exit $?
done

@TENSORJOIN_ROOT@/isaacsim6/env/bin/python \
  src/summarize_g4b_opportunity.py \
  > raw/g4b_opportunity_summary.log 2>&1
```
