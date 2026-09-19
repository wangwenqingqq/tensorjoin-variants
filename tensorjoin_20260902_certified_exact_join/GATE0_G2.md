# Gate 0: External Exact Self-Join and Scaling

Date frozen: 2026-09-03

## Decision

**CONTINUE, but only as an external-validity and scaling gate.** G2 does not
create a new novelty claim by itself. The broad thesis "use Tensor Cores or
mixed precision for a similarity self-join" is already subsumed by prior work.
The surviving research hypothesis is narrower:

> An output-sensitive, per-pair certified precision cascade can materialize the
> exact radius-join result while exploiting Tensor Cores, and can outperform
> exact GPU self-join systems under one frozen output and timing contract.

This hypothesis remains unvalidated until an exact, same-contract external
comparison passes. More modalities or a larger implementation alone cannot
rescue it.

## Nearest prior art and kill test

| Work | Closest overlap | Material difference from the surviving hypothesis | Gate-0 consequence |
|---|---|---|---|
| FaSTED, ICPP 2025 | FP16 input, FP32 accumulation, Tensor-Core all-pairs Euclidean self-join, pair materialization | The published Cifar60K result is not exact; the pinned implementation is tied to an A100 launch constant and has no license file | Kills any broad mixed-precision/Tensor-Core novelty claim. Retain only as a lower-quality contextual reference, not an exact keeper or redistributable dependency. |
| GDS-Join | Exact GPU distance-similarity self-join with a grid index and batched result materialization | Exact FP64, data-dependent index/pruning rather than a per-pair low-precision certificate cascade | Strong exact external keeper when it emits canonical pair IDs under G2. |
| MiSTIC | Exact GPU self-join using a tree/indexed search and result materialization | Exact FP64 and an indexing mechanism rather than a Tensor-Core certificate cascade | Strong exact external keeper when it emits canonical pair IDs under G2. |
| Tensor-Core similarity search / certified search / error-bounded dot products | Low-precision acceleration, refinement, or theoretical error control | None alone establishes the complete exact, output-sensitive GPU join contract claimed here | Prevents component-level novelty claims; novelty must be in the joined execution contract and measured consequence. |

## Source audit outcome

- FaSTED is pinned to commit `9af85ed8edc818d5aa7cc0473187631adab3e6ba`,
  the README-declared ICPP tag. No license file exists at that revision, so it
  is restricted to private compatibility testing and will not be copied into a
  public artifact.
- GDS-Join is pinned to `8093a4fdea93a24cf50a63ac9b074c31b7f66dfb`
  under MIT.
- MiSTIC is pinned to `656dd47b5594f1d9d7f961f008cfc6aedb0aed9d`
  under MIT.
- All three source snapshots build for SM120 after toolchain/configuration-only
  portability changes. Compilation is not evidence of correctness or speed.
- Cifar60K is obtained from the public GQR dataset page. No dataset license was
  found on that page, so the data are internal research input and must not be
  redistributed.

Exact revisions, source-tree hashes, archive hashes, build commands, binaries,
and failed build attempts are retained in `receipts/external_baseline_sources.json`,
`receipts/g2_external_builds_sha256.txt`,
`receipts/g2_external_binary_metadata.txt`, and
`raw/g2_external_build_audit*.log`.

## Cheap falsification sequence

1. **G2A input/oracle gate:** deterministically select 4,096 Cifar60K base
   vectors, calibrate a tie-safe radius for approximately 64 directed results
   per point including self-pairs, and emit one canonical FP64 oracle.
2. **G2A adapter gate:** make the minimum auditable adapters needed for
   GDS-Join and MiSTIC to return canonical pair IDs on physical GPU0. Any
   missing/extra/duplicate pair, overflow, crash, or unmapped reordered ID
   fails that baseline. No timing result from this stage is paper evidence.
3. **Candidate scaling admission:** demonstrate exact output on the same 4,096
   points without a fixed 65K output buffer or an all-pairs status tensor.
4. **G2B full-scale gate:** only after the first three gates pass, run Cifar60K
   at the frozen published radius and compare against both exact external
   keepers under the same end-to-end denominator.

## Thesis stop rule

Stop promoting the current direction as a systems-paper thesis if either of the
following is established under G2:

- the exact candidate cannot scale to full Cifar60K result materialization
  without violating the frozen memory/output contract; or
- it does not achieve the predeclared advantage over the faster validated exact
  external keeper.

FaSTED being faster does not alone fail an exact thesis because its output
quality differs. Conversely, engineering breadth, modality ports, or a small
internal-baseline speedup cannot override a failure against exact external
keepers.
