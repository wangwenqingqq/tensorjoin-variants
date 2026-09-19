# TensorJoin Paper Assets

These assets implement the post-G5 paper spine without merging the G2B
external-system evidence and the G4/G5 audited-router evidence.

## One-command build

From the repository root:

```bash
python3 paper_assets/build.py
```

The build:

1. extracts numbers from the four frozen JSON summaries and verifies their
   acceptance/rejection states;
2. generates two vector PDF figures and one vector table preview;
3. generates the booktabs LaTeX source for the main table;
4. renders RGB and grayscale 600-DPI previews;
5. places assets at the intended 510 pt paper width on QA pages; and
6. validates provenance hashes, repeated numbers, required scope labels, page
   sizes, and PDF font resources.

## Primary deliverables

- `figures/fig1_evidence_separated_teaser.pdf`
- `figures/fig2_exact_precision_routing_pipeline.pdf`
- `figures/table1_external_full_public.pdf`
- `tables/table1_external_full_public.tex`
- `FIGURE_CAPTIONS.md`
- `qa/QA_REPORT.md`
- `qa/paper_figures_placement.pdf`
- `qa/paper_table_placement.pdf`

Editable sources are under `src/`; all quantitative inputs are in
`data/paper_numbers.json` with source-file SHA-256 values.

## Current tool boundary

The local host has no `pdflatex`, `latexmk`, or `pdffonts`.  The vector PDFs
are generated directly with ReportLab, rendered with `pdftoppm`, and inspected
for Type3 fonts through pypdf.  The generated booktabs table still requires a
real manuscript LaTeX compile after it is inserted into the target template.
