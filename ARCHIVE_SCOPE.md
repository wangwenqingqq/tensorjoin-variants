# Branch Archive Scope

This branch contains only `tensorjoin_20260908_pruning_objective/` and seven shared/scoped archive support files.
It retains 71 curated source entries and records 32
associated binary release entries. See `archive_manifest.jsonl` for every included
and excluded entry belonging to this project.

The original publication's [full scope and verification record](https://github.com/wangwenqingqq/tensorjoin-variants/blob/bf01120b51f4c7d88117fe0fb5b0ba67a1eb4c9b/ARCHIVE_SCOPE.md)
remains authoritative for redaction rules, source-transfer checks, exclusions,
third-party rights, original limitations, and binary-container checks.
All project files in this branch match the original project subtree exactly;
only the repository selection and root-level explanatory metadata differ.

The shared [result release](https://github.com/wangwenqingqq/tensorjoin-variants/releases/tag/snapshot-2026-09-19) and its four part
hashes are unchanged. Its `SHA256SUMS` validates all parts; `result_assets.json`
describes the entire release, while this branch's manifest selects the project.
Restore only entries prefixed `tensorjoin_20260908_pruning_objective/`; do not restore other variants here
unless intentionally preparing a combined workspace.

`tools/materialize.py` resolves `@TENSORJOIN_ROOT@` in a separate destination and
accepts explicit `--map TOKEN=/configured/path` settings. Destination and mapped
paths must contain only ASCII letters, digits, `/`, `.`, `_`, and `-`.
The helper does not supply omitted datasets, environments, cross-variant imports,
or private host configuration. Prefer `main` when preparing a combined execution
workspace. Historical hashes refer to the original source; publication hashes
refer to the already-redacted archived files.

The original server source, combined `main` archive, snapshot tag, and release
attachments are retained. No remote experiment or GPU operation was performed.
