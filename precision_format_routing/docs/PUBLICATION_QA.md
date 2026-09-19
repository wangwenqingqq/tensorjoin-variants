# Publication QA

- The publication source bytes match every source hash recorded by all four
  measured processes. Rebuilding SUMMARY.json from the staged raw records was
  checked offline. No unmeasured numerical edits were made after the campaign.
- `git diff --cached --check` reports only blank lines at EOF in kernels.py and
  retained.py. These harmless whitespace warnings are explicitly retained to
  preserve the exact tested source bytes; they are not numerical/test failures.
- The private staging tree excludes input datasets, installed environments,
  raw GPU process paths/identifiers and binary/cache objects. Final raw numeric
  logs are included. Selected-GPU telemetry omits identifiers/process paths and
  retains original-log digests; guard results establish the ownership checks.
- Compiler/refinement manifest coverage is deliberately limited as stated in
  README.md; full cache bytes remain at the experiment origin. No claim of a
  complete execution-to-specialization mapping is made.
- Older diagnostic records are clearly separated and are not acceptance gates.
  Earlier false compiler assumptions and the scalar-threshold ABI hazard are
  documented, not rewritten into successful final runs.
- Existing third-party notices and the inherited archive remain unchanged.
  This branch is private; no public release or data redistribution is authorized.
