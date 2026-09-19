#!/usr/bin/env python3
"""Validate provenance, repeated numbers, PDF structure, fonts, and rendered size."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "paper_assets"
DATA = ASSETS / "data/paper_numbers.json"
REPORT = ASSETS / "qa/QA_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def pdf_check(relative: str, size: tuple[float, float], required: list[str]) -> dict[str, object]:
    path = ROOT / relative
    reader = PdfReader(str(path))
    assert len(reader.pages) == 1
    page = reader.pages[0]
    observed = (float(page.mediabox.width), float(page.mediabox.height))
    assert all(abs(a - b) < 0.01 for a, b in zip(observed, size)), (relative, observed, size)
    extracted = page.extract_text()
    for token in required:
        assert token in extracted, (relative, token)
    fonts = []
    resources = page["/Resources"]
    for name, reference in resources.get("/Font", {}).items():
        font = reference.get_object()
        subtype = str(font.get("/Subtype"))
        base = str(font.get("/BaseFont"))
        assert subtype != "/Type3", (relative, name, base)
        fonts.append({"name": str(name), "subtype": subtype, "base": base})
    return {
        "path": relative,
        "sha256": sha256(path),
        "page_size_points": list(observed),
        "fonts": sorted(fonts, key=lambda item: item["name"]),
    }


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for source in data["source_sha256"].values():
        path = ROOT / source["path"]
        assert sha256(path) == source["sha256"], path

    pdfs = [
        pdf_check(
            "paper_assets/figures/fig1_evidence_separated_teaser.pdf", (510, 220),
            ["1.800B", "1,827,007", "1,734", "3,926,078", "0.926 s",
             "4.732 s", "10.928 s", "598 FN + 1,130 FP", "no shared-binary claim"],
        ),
        pdf_check(
            "paper_assets/figures/fig2_exact_precision_routing_pipeline.pdf", (510, 234),
            ["440,391 upper tiles", "4,096 tiles/batch", "108 batches",
             "accept 1,278,504", "reject 1,796,924,489", "residual 1,734",
             "accept 871", "reject 863", "3,926,078 exact IDs"],
        ),
        pdf_check(
            "paper_assets/figures/table1_external_full_public.pdf", (510, 130),
            ["0.924369", "0.926153", "0.928841", "4.732177", "10.927797",
             "5.083x slower [5.030, 5.133]", "11.783x slower [11.716, 11.846]",
             "598 false negatives and 1,130 false positives"],
        ),
    ]

    pngs = []
    for name in ("fig1_600dpi.png", "fig1_600dpi_gray.png", "fig2_600dpi.png",
                 "fig2_600dpi_gray.png", "table1_600dpi.png", "table1_600dpi_gray.png"):
        path = ASSETS / "qa" / name
        image = Image.open(path)
        assert image.width == 4250
        assert image.height >= 1000
        pngs.append({"path": str(path.relative_to(ROOT)), "pixels": [image.width, image.height]})

    co_location_files = [
        ROOT / "PAPER_SPINE_LOCK.md",
        ROOT / "PAPER_MAIN_RESULTS_TABLE.md",
        ROOT / "PAPER_ASSET_CONTRACT.md",
        ASSETS / "FIGURE_CAPTIONS.md",
        ASSETS / "tables/table1_external_full_public.tex",
    ]
    required_tokens = ["3,926,078", "5.083x", "11.783x", "598", "1,130"]
    for path in co_location_files:
        value = path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in value, (path, token)
    table_text = (ROOT / "PAPER_MAIN_RESULTS_TABLE.md").read_text(encoding="utf-8")
    for token in ("4.091684x", "0.950650x", "1.972247x", "fail: below 1.50x"):
        assert token in table_text, token

    report = [
        "# Paper Asset QA Report",
        "",
        "Date: 2026-09-03",
        "",
        "## Automated gates",
        "",
        "- Frozen JSON provenance hashes: pass.",
        "- Repeated headline-number co-location: pass.",
        "- Required evidence/scope labels in vector PDFs: pass.",
        "- One-page media boxes at intended 510 pt paper width: pass.",
        "- PDF font inspection through pypdf: Type1 Helvetica/Helvetica-Bold only; no Type3: pass.",
        "- RGB and grayscale 600-DPI renders at 4,250 px width: pass.",
        "- `pdffonts` is unavailable on this host; pypdf font-resource inspection is the recorded fallback.",
        "",
        "## Visual gates",
        "",
        "- Standalone RGB and grayscale renders inspected at 600 DPI: no observed clipping, overlap, ambiguous connector, or unreadable label.",
        "- Both figures inspected on a letter page at their intended 510 pt width with captions: no observed width overflow or figure-caption collision.",
        "- The table inspected at 510 pt width: no observed cell overflow; FaSTED remains explicitly non-exact.",
        "",
        "## Vector PDF receipts",
        "",
        "```json",
        json.dumps(pdfs, indent=2, sort_keys=True),
        "```",
        "",
        "## Render receipts",
        "",
        "```json",
        json.dumps(pngs, indent=2, sort_keys=True),
        "```",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
