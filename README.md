# TensorJoin Variants

Private source-and-evidence archive of six TensorJoin experiment directories,
snapshotted on 2026-09-19. The original directory names and nested experiment
structure are preserved, including rejected experiments and negative results.
This is an archival upload, **not a newly validated benchmark or a complete machine backup**.

## Projects

| Directory | Contents |
| --- | --- |
| [tensorjoin_20260902_certified_exact_join](tensorjoin_20260902_certified_exact_join/) | Main exploration, CUDA/Triton implementations, baseline adapters, protocols, decisions, experiment records, paper sources, and nested follow-up experiments |
| [tensorjoin_20260908_block_feasibility](tensorjoin_20260908_block_feasibility/) | Block-feasibility experiment and numerical-scope documentation |
| [tensorjoin_20260908_learned_layout](tensorjoin_20260908_learned_layout/) | Learned-layout experiment, reports, and analysis |
| [tensorjoin_20260908_pruning_objective](tensorjoin_20260908_pruning_objective/) | Pruning-objective experiment, reports, and analysis |
| [tensorjoin_20260908_fm_control](tensorjoin_20260908_fm_control/) | Flow-matching controls and experiment records |
| [tensorjoin_20260908_fm_solver](tensorjoin_20260908_fm_solver/) | Conditional flow-matching solver, certificates, controls, and resolution checks |

Read each project's `REPORT.md`, `CURRENT_STATUS.md`, protocols, and decision
records for its historical scope. Their dates and claims describe the original
experiments, not a fresh validation performed during this upload.

## Archive scope

- Git contains source code, build inputs, third-party notices, research documents,
  paper sources and existing figures, raw text logs, JSON/CSV records, summaries,
  and small upstream test fixtures.
- Bulk numeric result arrays are kept separately as release assets, where listed
  in [archive_manifest.jsonl](archive_manifest.jsonl).
- Bulk source datasets, pretrained model packages, installed environments,
  downloaded dependency packages, build/cache output, and operating-system
  metadata are not included in Git.
- [ARCHIVE_SCOPE.md](ARCHIVE_SCOPE.md) describes exclusions, path redactions,
  binary evidence limitations, and integrity checks. The manifest accounts for
  every inventoried source entry, including entries not uploaded.
- Existing source and result hashes remain historical evidence. A path-redacted
  publication file is not byte-identical to its original; the manifest records
  both hashes where applicable.

## Working with the sources

Private absolute paths have been replaced by explicit `@TOKEN@` placeholders.
Do not run the historical campaign scripts directly against this archive.
Create a separate working copy first:

```sh
python3 tools/materialize.py /path/to/new/tensorjoin-workspace
```

`@TENSORJOIN_ROOT@` is resolved to that new directory. Other external locations
can be supplied with repeated `--map TOKEN=/configured/location` arguments.
Destination and mapped paths must be absolute and contain only ASCII letters,
digits, `/`, `.`, `_`, and `-`; this avoids corrupting embedded source literals.
The helper refuses to overwrite an existing destination and never starts an
experiment or installs dependencies. External dependencies, datasets, GPU
selection, and machine-specific launch settings still require a new preflight.
Host aliases in historical logs are descriptive, not usable SSH configuration.

## Rights and provenance

This private collection does not relicense the included material. Existing
third-party copyright and license notices are retained, including NVIDIA OptiX,
OWL, pybind11, GLFW, and baseline implementations. Paper and literature PDFs
retain their original attribution. Do not redistribute the collection publicly
without a separate rights review.
