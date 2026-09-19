# Overleaf synchronization receipt

Date: 2026-09-04

## Canonical project

- Project: `TensorJoin: Certified Precision Routing for Exact GPU Similarity Joins`
- URL: `https://www.overleaf.com/project/6a999de1dba22542e76a0ad6`
- Local source: `overleaf_bundle/`
- Upload archive: `overleaf_upload/TensorJoin_Overleaf_Draft_final.zip`
- Upload archive SHA-256:
  `72cc87c2eebfd41f31f8eb2d0298aa0114fbefbfad040e13aad9c946ec9db559`

## G6 latest-baseline update

- Server-synchronized source count: nine files, including
  `tables/table4_latest_baseline_audit.tex`.
- Current local upload archive:
  `overleaf_upload/TensorJoin_Overleaf_Draft_g6.zip`.
- Current upload archive SHA-256:
  `1106432b8667f6d7436ca630e40a3e95a81d6a9c597608f0892461e4ea4c48fc`.
- Server source round trip:
  `overleaf_compile/TensorJoin_Overleaf_source_g6.zip`.
- Server source ZIP SHA-256:
  `a6d2fe2b88d2e97c21eb4523a5d36e68938ff46dbf05b4b7c37afadccd01b3da`.
- All nine server files match `overleaf_bundle/` byte-for-byte.
- Overleaf compiled the update with zero errors.  The output remains six
  pages; text extraction finds no unresolved-reference marker.
- Current compiled PDF:
  `overleaf_compile/TensorJoin_Overleaf_output_g6.pdf`.
- Current compiled PDF SHA-256:
  `93310f9a22337fcd786b87eda6790f135556d8fc25c4271a403986d4922152cf`.
- All pages were rendered at 180 DPI and checked as a contact sheet.  The new
  audit table on page 5 was additionally rendered and inspected at 600 DPI;
  no clipping, overlap, or unreadable cell was observed.
- Pre-edit local backup:
  `backups/overleaf_bundle_pre_g6_20260904_020610.tar.gz`, SHA-256
  `fdb3098852774f575d9e1dc0ad5ba0a87dcd409952215fd9faf68fa924c421e0`.

## Verification

- The server accepted all eight uploaded files.
- A server-side ZIP round trip reproduced all eight files byte-for-byte.
- Overleaf compiled `main.tex` successfully with pdfLaTeX and BibTeX.
- The compiled artifact has six pages and no missing citations, undefined
  references, LaTeX errors, or overfull boxes.
- All six pages were rendered at 180 DPI and visually checked for clipping,
  overlap, broken figures, unreadable tables, and malformed glyphs.
- Local downloaded PDF: `overleaf_compile/TensorJoin_Overleaf_output.pdf`.
- Downloaded PDF SHA-256:
  `cc07a9200303a3232e7b53c84909a10c884982b35b305209d38dd70d8853a7ba`.

The remaining ACM-class warning is intentional: the anonymous draft suppresses
the publication-reference footer until venue metadata and rights information
are assigned.  Underfull vertical boxes are float/page balancing notices and
do not correspond to visible overlap or clipping.

Four intermediate projects created during render-and-fix QA were moved to the
Overleaf trash.  The canonical project above remains active and was rechecked
after cleanup.

No account credential or session cookie is stored in this receipt, the source
bundle, the upload archive, or the compiled output.
