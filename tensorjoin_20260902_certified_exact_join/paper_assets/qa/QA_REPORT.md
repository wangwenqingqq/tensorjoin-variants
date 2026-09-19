# Paper Asset QA Report

Date: 2026-09-03

## Automated gates

- Frozen JSON provenance hashes: pass.
- Repeated headline-number co-location: pass.
- Required evidence/scope labels in vector PDFs: pass.
- One-page media boxes at intended 510 pt paper width: pass.
- PDF font inspection through pypdf: Type1 Helvetica/Helvetica-Bold only; no Type3: pass.
- RGB and grayscale 600-DPI renders at 4,250 px width: pass.
- `pdffonts` is unavailable on this host; pypdf font-resource inspection is the recorded fallback.

## Visual gates

- Standalone RGB and grayscale renders inspected at 600 DPI: no observed clipping, overlap, ambiguous connector, or unreadable label.
- Both figures inspected on a letter page at their intended 510 pt width with captions: no observed width overflow or figure-caption collision.
- The table inspected at 510 pt width: no observed cell overflow; FaSTED remains explicitly non-exact.

## Vector PDF receipts

```json
[
  {
    "fonts": [
      {
        "base": "/Helvetica",
        "name": "/F1",
        "subtype": "/Type1"
      },
      {
        "base": "/Helvetica-Bold",
        "name": "/F2",
        "subtype": "/Type1"
      }
    ],
    "page_size_points": [
      510.0,
      220.0
    ],
    "path": "paper_assets/figures/fig1_evidence_separated_teaser.pdf",
    "sha256": "41d9e4d32f9ebeb590f88397b55f50a8001ffe7feb936d0a33c9a3f900ea8332"
  },
  {
    "fonts": [
      {
        "base": "/Helvetica",
        "name": "/F1",
        "subtype": "/Type1"
      },
      {
        "base": "/Helvetica-Bold",
        "name": "/F2",
        "subtype": "/Type1"
      }
    ],
    "page_size_points": [
      510.0,
      234.0
    ],
    "path": "paper_assets/figures/fig2_exact_precision_routing_pipeline.pdf",
    "sha256": "d638033f7faf1375c983d5dab09e1166a3fbba3ff468f0f92d42efd453223ad4"
  },
  {
    "fonts": [
      {
        "base": "/Helvetica",
        "name": "/F1",
        "subtype": "/Type1"
      },
      {
        "base": "/Helvetica-Bold",
        "name": "/F2",
        "subtype": "/Type1"
      }
    ],
    "page_size_points": [
      510.0,
      130.0
    ],
    "path": "paper_assets/figures/table1_external_full_public.pdf",
    "sha256": "1f34a634f2442964504e8fbbc169632561b160f05fdcaaf2170f4ed1581d07fc"
  }
]
```

## Render receipts

```json
[
  {
    "path": "paper_assets/qa/fig1_600dpi.png",
    "pixels": [
      4250,
      1834
    ]
  },
  {
    "path": "paper_assets/qa/fig1_600dpi_gray.png",
    "pixels": [
      4250,
      1834
    ]
  },
  {
    "path": "paper_assets/qa/fig2_600dpi.png",
    "pixels": [
      4250,
      1950
    ]
  },
  {
    "path": "paper_assets/qa/fig2_600dpi_gray.png",
    "pixels": [
      4250,
      1950
    ]
  },
  {
    "path": "paper_assets/qa/table1_600dpi.png",
    "pixels": [
      4250,
      1084
    ]
  },
  {
    "path": "paper_assets/qa/table1_600dpi_gray.png",
    "pixels": [
      4250,
      1084
    ]
  }
]
```
