#!/usr/bin/env python3
"""Generate a vector preview and LaTeX source for the main external-results table."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab import rl_config

rl_config.invariant = 1

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper_assets/data/paper_numbers.json"
FIGURES = ROOT / "paper_assets/figures"
TABLES = ROOT / "paper_assets/tables"

INK = HexColor("#202124")
MUTED = HexColor("#5F6368")
BLUE_LIGHT = HexColor("#DCEEF8")
GRAY_LIGHT = HexColor("#F2F2F2")


def make_pdf(data: dict[str, object]) -> Path:
    output = FIGURES / "table1_external_full_public.pdf"
    document = SimpleDocTemplate(
        str(output), pagesize=(510, 130), leftMargin=0, rightMargin=0,
        topMargin=2, bottomMargin=0, title="TensorJoin main external results table",
        author="TensorJoin artifact",
    )
    style = ParagraphStyle(
        "caption", fontName="Helvetica", fontSize=6.2, leading=7.4,
        textColor=INK, alignment=TA_LEFT, spaceAfter=3,
    )
    header_style = ParagraphStyle(
        "head", parent=style, fontName="Helvetica-Bold", fontSize=6.1,
        leading=7.1, textColor=INK,
    )
    cell_style = ParagraphStyle(
        "cell", parent=style, fontSize=6.15, leading=7.3, textColor=INK,
    )
    small_style = ParagraphStyle(
        "small", parent=style, fontSize=5.7, leading=6.8, textColor=MUTED,
    )
    methods = data["g2b_external"]["methods"]
    rows = [
        [Paragraph("Method", header_style), Paragraph("Arithmetic / output", header_style),
         Paragraph("p10 (s)", header_style), Paragraph("Median (s)", header_style),
         Paragraph("p90 (s)", header_style), Paragraph("Exact", header_style),
         Paragraph("Relative to TensorJoin", header_style)],
    ]
    labels = [
        ("TensorJoin", "tensorjoin", "certified mixed precision", "yes", "1.000x"),
        ("MiSTIC", "mistic", "FP64", "yes", "5.083x slower [5.030, 5.133]"),
        ("GDS-Join", "gds", "FP64", "yes", "11.783x slower [11.716, 11.846]"),
        ("FaSTED", "fasted", "FP16 / FP32; changed output", "no", "4.037x faster (context only)"),
    ]
    for label, key, arithmetic, exact, relative in labels:
        values = methods[key]
        method_text = f"<b>{label}</b>" if key == "tensorjoin" else label
        rows.append([
            Paragraph(method_text, cell_style), Paragraph(arithmetic, cell_style),
            Paragraph(f"{values['p10_s']:.6f}", cell_style),
            Paragraph(f"{values['median_s']:.6f}", cell_style),
            Paragraph(f"{values['p90_s']:.6f}", cell_style),
            Paragraph(exact, cell_style), Paragraph(relative, cell_style),
        ])
    table = Table(rows, colWidths=[58, 108, 50, 56, 50, 34, 154], rowHeights=[18, 18, 18, 18, 20])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GRAY_LIGHT),
        ("BACKGROUND", (0, 1), (-1, 1), BLUE_LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (5, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.55, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, HexColor("#C9CDD1")),
    ]))
    caption = Paragraph(
        "<b>Full public-denominator external comparison.</b> CIFAR-GIST-512, "
        "60K self-join, pageable-host FP32 input through sorted canonical host "
        "IDs; 3,926,078 exact directed pairs.", style,
    )
    note = Paragraph(
        "FaSTED has 598 false negatives and 1,130 false positives. Exact-system "
        "intervals are paired 95% round-bootstrap intervals; FaSTED is approximate context only.",
        small_style,
    )
    document.build([caption, Spacer(1, 1.5), table, Spacer(1, 2.5), note])
    return output


def make_tex(data: dict[str, object]) -> Path:
    output = TABLES / "table1_external_full_public.tex"
    m = data["g2b_external"]["methods"]
    lines = [
        r"% Generated from paper_assets/data/paper_numbers.json.",
        r"% Requires booktabs; no vertical rules.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full public-denominator external comparison on the CIFAR-GIST-512 60K self-join, from pageable-host FP32 input through 3,926,078 sorted canonical host IDs. Exact-system intervals are paired 95\% round-bootstrap intervals. FaSTED is approximate context only: it has 598 false negatives and 1,130 false positives.}",
        r"\label{tab:external-full-public}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrcl}",
        r"\toprule",
        r"Method & Arithmetic / output & p10 (s) & Median (s) & p90 (s) & Exact & Relative to TensorJoin \\",
        r"\midrule",
        f"\\textbf{{TensorJoin}} & certified mixed precision & {m['tensorjoin']['p10_s']:.6f} & {m['tensorjoin']['median_s']:.6f} & {m['tensorjoin']['p90_s']:.6f} & yes & 1.000x " + r"\\",
        f"MiSTIC & FP64 & {m['mistic']['p10_s']:.6f} & {m['mistic']['median_s']:.6f} & {m['mistic']['p90_s']:.6f} & yes & 5.083x slower [5.030, 5.133] " + r"\\",
        f"GDS-Join & FP64 & {m['gds']['p10_s']:.6f} & {m['gds']['median_s']:.6f} & {m['gds']['p90_s']:.6f} & yes & 11.783x slower [11.716, 11.846] " + r"\\",
        f"FaSTED & FP16 / FP32; changed output & {m['fasted']['p10_s']:.6f} & {m['fasted']['median_s']:.6f} & {m['fasted']['p90_s']:.6f} & no & 4.037x faster (context only) " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    print(make_pdf(data))
    print(make_tex(data))


if __name__ == "__main__":
    main()
