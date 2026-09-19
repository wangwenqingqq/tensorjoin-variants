# Archive Scope and Integrity

## Inventory

The source inventory contains 54,974 entries totaling 27,375,097,685 bytes across six TensorJoin roots.

- **Git source entries:** 7,916, plus the archive's README, scope, manifests, and path-materialization helper.
- **Release result entries:** 1,518, totaling 3,855,250,378 bytes before compression.
- **Opaque profiler/trace entries withheld:** 44, totaling 97,628,661 bytes. Binary profiler reports and trace databases can embed compressed host/process information; existing text exports remain in Git where present.
- **Redacted text entries:** 4,307. Original and publication SHA-256 values are distinguished in the manifest.

The archive manifest is the authoritative per-entry inclusion/exclusion list.
A path listed as excluded does not imply that it was deleted from the original source.
This is a curated research archive, not a byte-for-byte server backup.

## Included and excluded material

All six top-level experiment directories and the main directory's nested follow-up
experiments are represented. Source, build inputs, reports, negative decisions,
textual raw observations, summary data, existing paper material, and upstream
notices are retained. Small upstream test fixtures are intentionally retained.
Bulk numeric arrays and small experiment-generated checkpoints are release assets.

Excluded categories include bulk source datasets and pretrained model packages,
installed Python environments, downloaded wheels/archives, compilation and runtime
caches, compiled/transient files, nested Git metadata, and filesystem metadata.
The 43.77 MB exact-vector CSV is a bulk input dataset, not a small test fixture.
The duplicate packaged Overleaf source ZIP is omitted; unpacked sources are kept.
No unredacted server Git history or environment credential store is published.

## Redactions and path handling

Private absolute roots are replaced by `@TENSORJOIN_ROOT@`, `@LOCAL_WORKSPACE@`,
`@LOCAL_HOME@`, `@EXTERNAL_DATA_ROOT@`, `@EXTERNAL_HOME@`,
`@EXTERNAL_USER_HOME@`, and `@ADMIN_HOME@` placeholders where applicable.
Machine-specific host/user labels are generalized. Authorization values in an
historical signed download URL are replaced with `REDACTED` before the first commit.
Public vendor contact information, public repository URLs, package versions,
third-party license notices, measurements, and historical process inventories
that do not disclose private identity remain attribution/evidence, not new authorship.

`tools/materialize.py` creates a new copy and resolves explicitly configured path
roots. It does not install the historical external environment, provide omitted
inputs, validate GPU compatibility, or make old absolute cross-project dependencies
available. New experiment execution always needs a fresh machine/data preflight.

## Restoring binary results

Download every result part and `SHA256SUMS` from the
[2026-09-19 snapshot release](https://github.com/wangwenqingqq/tensorjoin-variants/releases/tag/snapshot-2026-09-19).
Verify the part checksums before extraction:

```sh
shasum -a 256 -c SHA256SUMS
cat tensorjoin-results-2026-09-19.tar.gz.part-* | tar -xz -C /path/to/this/archive
```

The parts form one ordered gzip-compressed tar stream, not independent archives.
Use a fresh checkout for restoration. Release entries retain their source bytes
and SHA-256. Existing files at matching result paths would be overwritten by tar;
do not extract over a working experiment. Binary metadata was checked without
executing or unpickling checkpoint objects. Numeric NPY schemas were validated
and inspected for text metadata; opaque profiler/database containers were withheld.

## Verification performed for this upload

- Verified transferred source and binary files against source-side SHA-256 values.
- Parsed 820 Python files (including the new helper) and 1,817 archived JSON files.
- Tested helper path substitution, source preservation, symlink preservation,
  refusal to overwrite, and rejection of unsafe path characters using isolated fixtures.
- Preserved pre-existing formatting; the legacy import has whitespace diagnostics,
  while the new archive documentation/helper passes the staged whitespace check.
- Scanned source text and reachable Git history for sensitive data and forbidden
  author trailers; independently reviewed the publication copy.
- Inspected text/metadata from the 24 retained PDFs for publication privacy.
- Retained numerical evidence and historical result hashes without presenting them
  as a fresh performance campaign.

**Not performed:** CUDA compilation, dependency reconstruction, GPU correctness,
sanitizer runs, performance reruns, or a fresh visual/layout review of paper files.
No remote experiment was started or stopped, and original source files were not edited.

## Known inherited limitation

The following archived rejected-generation symlink was already broken in the source
and is preserved as historical evidence, not silently repaired:

`tensorjoin_20260902_certified_exact_join/g19_reusable_complete_cost_20260905/artifacts/rejected_generation_a0/adapter_a0/OWL`

Its original target is `../../adapters/rthiss_g7_a0/OWL`. The three main adapter
symlinks resolve correctly inside the staged source collection.

## Licensing

There is no blanket new license for this mixed research collection. Existing
third-party notices and licenses are retained. Private hosting is not permission
for later public redistribution; perform a separate rights review before publication.
