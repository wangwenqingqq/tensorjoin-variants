# TensorJoin Overleaf draft

This directory is the upload bundle for a new anonymous ACM-style paper
project.

## Evidence boundary

- The external end-to-end comparison comes from G2B.
- The generated-code and dynamic-router evidence comes from G3/G4.
- G5 proves that a full-scale artifact can be exact and audited, but its frozen
  performance screen fails.  The manuscript therefore does not make a
  shared-binary performance claim.
- FaSTED is approximate context, not an exact comparator.

## Build

Compile `main.tex` with pdfLaTeX and BibTeX.  The intended template is
`acmart` in anonymous review mode.

The numeric inventory is frozen by `paper_assets/data/paper_numbers.json` in
the parent experiment directory.  Do not edit repeated numbers independently.
