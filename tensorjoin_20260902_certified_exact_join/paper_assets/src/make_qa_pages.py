#!/usr/bin/env python3
"""Place paper assets at their intended 510 pt width on letter-sized QA pages."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab import rl_config

rl_config.invariant = 1

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper_assets/figures"
QA = ROOT / "paper_assets/qa"
PAGE_W, PAGE_H = 612.0, 792.0
X = 51.0
INK = HexColor("#202124")
MUTED = HexColor("#5F6368")


def wrapped(c: canvas.Canvas, x: float, y: float, width: float, value: str,
            *, size=7.3, leading=9.0, bold_prefix=False) -> float:
    font = "Helvetica"
    words = value.split()
    line = ""
    lines: list[str] = []
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for i, item in enumerate(lines):
        c.setFont("Helvetica-Bold" if bold_prefix and i == 0 else font, size)
        c.drawString(x, y - i * leading, item)
    return y - len(lines) * leading


def base_page1(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 10)
    c.drawString(X, 770, "Paper-assets placement QA - full 510 pt width")
    c.setFillColor(MUTED); c.setFont("Helvetica", 7)
    c.drawRightString(PAGE_W - X, 770, "not manuscript prose")
    c.setFillColor(INK)
    wrapped(c, X, 501, 510,
            "Figure 1. Certified routing concentrates high precision on the decision boundary, while a separately scoped scalable implementation establishes external-system impact. Panels use different implementation scopes and do not constitute a shared-binary claim.",
            size=7.2, leading=8.8)
    wrapped(c, X, 173, 510,
            "Figure 2. TensorJoin executes precision escalation as a dynamic exact query plan. Host-visible counts set the FP32 and FP64 launch extents; direct accepts bypass later precision stages and join canonical reconstruction. Counts are from G5 P1.",
            size=7.2, leading=8.8)
    c.showPage(); c.save()


def base_page2(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 10)
    c.drawString(X, 770, "Main-results table placement QA - full 510 pt width")
    c.setFillColor(MUTED); c.setFont("Helvetica", 7)
    c.drawRightString(PAGE_W - X, 770, "booktabs TeX source retained separately")
    c.setFillColor(INK)
    wrapped(c, X, 566, 510,
            "Table 1. The scalable G2B implementation outperforms both exact external systems under the complete public denominator, whereas the faster FaSTED point changes the output set.",
            size=7.2, leading=8.8)
    c.showPage(); c.save()


def merge_asset(base_path: Path, asset_path: Path, x: float, y: float,
                output_path: Path) -> None:
    base = PdfReader(str(base_path)).pages[0]
    asset = PdfReader(str(asset_path)).pages[0]
    base.merge_transformed_page(asset, Transformation().translate(tx=x, ty=y), over=True)
    writer = PdfWriter()
    writer.add_page(base)
    with output_path.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    base1 = QA / "placement_base_figures.pdf"
    base2 = QA / "placement_base_table.pdf"
    base_page1(base1)
    base_page2(base2)

    # Merge both figures onto page 1 while retaining vector content.
    first = QA / "placement_first_merge.pdf"
    merge_asset(base1, FIGURES / "fig1_evidence_separated_teaser.pdf", X, 520, first)
    merge_asset(first, FIGURES / "fig2_exact_precision_routing_pipeline.pdf", X, 210,
                QA / "paper_figures_placement.pdf")
    merge_asset(base2, FIGURES / "table1_external_full_public.pdf", X, 590,
                QA / "paper_table_placement.pdf")
    print(QA / "paper_figures_placement.pdf")
    print(QA / "paper_table_placement.pdf")


if __name__ == "__main__":
    main()
