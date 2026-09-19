# Variant Branches

The six project directories are available as separate branch-specific archive
views. `main` remains the combined archive. Source directory names and source
subtree contents are unchanged from the original `bf01120` snapshot.

| Branch | Project directory | Branch commit |
| --- | --- | --- |
| [certified-exact-join](https://github.com/wangwenqingqq/tensorjoin-variants/tree/certified-exact-join) | `tensorjoin_20260902_certified_exact_join/` | [`3dc8cfc`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/3dc8cfc657ea3d3f5df05b264731459c54324ff4) |
| [block-feasibility](https://github.com/wangwenqingqq/tensorjoin-variants/tree/block-feasibility) | `tensorjoin_20260908_block_feasibility/` | [`1c32c00`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/1c32c00799d379c7d09b19511bffbbc895c219f5) |
| [learned-layout](https://github.com/wangwenqingqq/tensorjoin-variants/tree/learned-layout) | `tensorjoin_20260908_learned_layout/` | [`0a77781`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/0a77781b6901611640c759eefe7a1629efbd95b4) |
| [pruning-objective](https://github.com/wangwenqingqq/tensorjoin-variants/tree/pruning-objective) | `tensorjoin_20260908_pruning_objective/` | [`b3fef1e`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/b3fef1ebbf802c37e65c26537328a5b62ad3d6e5) |
| [fm-control](https://github.com/wangwenqingqq/tensorjoin-variants/tree/fm-control) | `tensorjoin_20260908_fm_control/` | [`c611d58`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/c611d58c62980489d5f9af690d13954fed344a59) |
| [fm-solver](https://github.com/wangwenqingqq/tensorjoin-variants/tree/fm-solver) | `tensorjoin_20260908_fm_solver/` | [`ce27f5c`](https://github.com/wangwenqingqq/tensorjoin-variants/commit/ce27f5cb55d6d6b88a6d4c61af57b88c3f5a654c) |

## Scope and dependencies

Each branch tip retains one original project directory plus seven scoped/shared
support files. The original full archive remains in their common parent history;
this is not a history rewrite or a claim that other versions never existed.
Branch manifests contain only the selected project's inventory and exclusions.

Some historical implementations import sibling projects. Their source code and
path contract have not been rewritten to hide these dependencies. Each branch
README lists direct sibling-project references found in source text; the list is
not a complete runtime dependency analysis. Use `main` for a combined workspace,
or deliberately provide the required sibling directories before execution.

The [snapshot release](https://github.com/wangwenqingqq/tensorjoin-variants/releases/tag/snapshot-2026-09-19)
and all result part checksums are unchanged. The release is shared by all branches;
restore only the selected project's result paths into a single-variant checkout.
Existing data/environment/profiler exclusions and third-party rights still apply.

## Validation

For every branch, the committed project subtree hash is identical to the same
subtree in `bf01120`, preserving file bytes, modes, and symlink targets. The scoped
manifest's source paths equal the branch tree's source paths. Only repository
organization and root-level explanatory metadata changed. No GPU experiment,
CUDA build, numerical validation, or performance rerun was performed for this split.
