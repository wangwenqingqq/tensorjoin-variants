# TensorJoin — learned-layout

This branch contains the `tensorjoin_20260908_learned_layout/` snapshot only, plus archive metadata and the
shared path-materialization helper. Original directory names, source bytes,
third-party notices, nested experiments, and historical evidence are preserved.

- **Source directory:** [tensorjoin_20260908_learned_layout](tensorjoin_20260908_learned_layout/)
- **Curated source entries:** 65
- **Associated release entries:** 32
- **Snapshot origin:** [`bf01120`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/bf01120b51f4c7d88117fe0fb5b0ba67a1eb4c9b)
- **All variants and branch index:** [main](https://github.com/wangwenqingqq/tensorjoin-variants/tree/main)
- **Shared result attachments:** [snapshot-2026-09-19](https://github.com/wangwenqingqq/tensorjoin-variants/releases/tag/snapshot-2026-09-19)

## Scope

This is a branch-specific view of the existing archive, not a new benchmark,
a newly independent implementation, or a complete server backup. The parent
commit retains the combined archive; only this branch's tip is narrowed.
`archive_manifest.jsonl` and `publication_summary.json` describe this project only.
`result_assets.json` describes the unchanged combined release for all six projects.
Source datasets, pretrained packages, environments, caches, and opaque profiler
containers remain excluded as recorded in the manifest.

## Cross-variant dependencies

Historical scripts still refer to sibling projects. Branch splitting changes
repository organization, not their imports or execution contract. Direct sibling
names found in the inspected source include:

- [certified-exact-join](https://github.com/wangwenqingqq/tensorjoin-variants/tree/certified-exact-join): `tensorjoin_20260902_certified_exact_join/`
- [block-feasibility](https://github.com/wangwenqingqq/tensorjoin-variants/tree/block-feasibility): `tensorjoin_20260908_block_feasibility/`

The list is a text-reference inventory, not proof that all runtime dependencies
are known. For execution, use the combined `main` checkout or materialize the
required sibling directories at their original names before a new preflight.
Do not infer self-contained execution from a separate Git branch.

## Paths and binary results

Private paths remain explicit `@TOKEN@` placeholders. `tools/materialize.py`
creates a separate working copy, never overwrites an existing destination, and
does not start experiments. See [ARCHIVE_SCOPE.md](ARCHIVE_SCOPE.md).

The release is shared and unchanged. Verify its part hashes, then restore only
archive entries under `tensorjoin_20260908_learned_layout/` into a fresh checkout of this branch. Do not
extract the full release over a working experiment or assume all release files
belong to this branch.

## Validation and rights

This branch's project subtree is byte-for-byte identical to its subtree in the
original archived commit, including modes and symlink targets. The branch-specific
file manifest was checked against the committed tree. No CUDA compilation,
GPU correctness check, or performance experiment was performed for this split.
Existing third-party licenses remain applicable; this branch does not relicense
any material and remains private.
