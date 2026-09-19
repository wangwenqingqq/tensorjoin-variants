#!/usr/bin/env python3
"""Generate the evidence-separated TensorJoin teaser and pipeline as vector PDFs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab import rl_config

rl_config.invariant = 1

from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper_assets/data/paper_numbers.json"
OUT = ROOT / "paper_assets/figures"

W = 510.0  # 7.083 in: full-width ACM/VLDB-style figure.
TEASER_H = 220.0
PIPELINE_H = 234.0

INK = HexColor("#202124")
MUTED = HexColor("#5F6368")
GRID = HexColor("#B8BDC3")
LIGHT = HexColor("#F4F6F7")
BLUE = HexColor("#0072B2")
BLUE_LIGHT = HexColor("#DCEEF8")
SKY = HexColor("#56B4E9")
SKY_LIGHT = HexColor("#E5F5FB")
ORANGE = HexColor("#E69F00")
ORANGE_LIGHT = HexColor("#FFF0CB")
GRAY = HexColor("#777777")
GRAY_LIGHT = HexColor("#E5E5E5")
GREEN = HexColor("#009E73")
GREEN_LIGHT = HexColor("#DDF3EA")


def load_data() -> dict[str, object]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def set_stroke(c: canvas.Canvas, color=INK, width=0.7) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)


def text(c: canvas.Canvas, x: float, y: float, value: str, size=7.0, *,
         font="Helvetica", color=INK, align="left") -> None:
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def text_fitted(c: canvas.Canvas, x: float, y: float, value: str, width: float,
                size=7.0, *, font="Helvetica", color=INK, align="center",
                minimum=5.8) -> float:
    fitted = size
    while fitted > minimum and stringWidth(value, font, fitted) > width:
        fitted -= 0.1
    text(c, x if align != "right" else x + width, y, value, fitted,
         font=font, color=color, align=align)
    return fitted


def rounded_box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                fill, *, stroke=INK, radius=4.0, width=0.7) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
          *, color=INK, width=1.0, head=4.0) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    left = angle + 2.55
    right = angle - 2.55
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(x2 + head * math.cos(left), y2 + head * math.sin(left))
    path.lineTo(x2 + head * math.cos(right), y2 + head * math.sin(right))
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def down_arrow(c: canvas.Canvas, x: float, y1: float, y2: float, **kwargs) -> None:
    arrow(c, x, y1, x, y2, **kwargs)


def panel_title(c: canvas.Canvas, x: float, y: float, label: str, title_value: str) -> None:
    text(c, x, y, label, 8.2, font="Helvetica-Bold", color=INK)
    text(c, x + 15, y, title_value, 8.2, font="Helvetica-Bold", color=INK)


def hatch_rect(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.saveState()
    path = c.beginPath()
    path.rect(x, y, w, h)
    c.clipPath(path, stroke=0, fill=0)
    c.setStrokeColor(HexColor("#555555"))
    c.setLineWidth(0.45)
    offset = -h
    while offset < w + h:
        c.line(x + offset, y, x + offset + h, y + h)
        offset += 5.0
    c.restoreState()


def make_teaser(data: dict[str, object]) -> Path:
    path = OUT / "fig1_evidence_separated_teaser.pdf"
    c = canvas.Canvas(str(path), pagesize=(W, TEASER_H), pageCompression=1)
    c.setTitle("TensorJoin evidence-separated teaser")
    c.setAuthor("TensorJoin artifact")

    funnel = data["g5_precision_funnel"]
    methods = data["g2b_external"]["methods"]

    panel_title(c, 14, 203, "(a)", "Exact precision funnel - audited G5 mechanism")
    panel_title(c, 301, 203, "(b)", "Full public latency - separate G2B system path")
    set_stroke(c, GRID, 0.55)
    c.line(289, 30, 289, 207)

    boxes = [
        (25, 150, 246, 30, BLUE_LIGHT, "Stage 1  INT8 Tensor Cores",
         f"{funnel['upper_pairs']/1e9:.3f}B upper pairs"),
        (61, 107, 174, 30, ORANGE_LIGHT, "Stage 2  certified FP32",
         f"{funnel['stage1_ambiguous']:,} pairs ({funnel['stage1_ambiguous_percent']:.4f}% of all)"),
        (95, 64, 106, 30, SKY_LIGHT, "Stage 3  exact FP64",
         f"{funnel['stage3_fp64']:,} pairs ({funnel['stage3_fraction_of_all_percent']:.6f}% of all)"),
    ]
    for x, y, w, h, fill, line1, line2 in boxes:
        rounded_box(c, x, y, w, h, fill)
        text_fitted(c, x + w / 2, y + 17.5, line1, w - 8, 7.8, font="Helvetica-Bold")
        text_fitted(c, x + w / 2, y + 7.0, line2, w - 8, 6.9, color=MUTED)
    down_arrow(c, 148, 149, 138, color=ORANGE, width=1.15, head=4.3)
    text(c, 153, 141.5, "0.1015% escalate", 6.2, color=MUTED)
    down_arrow(c, 148, 106, 95, color=ORANGE, width=1.15, head=4.3)
    text(c, 153, 98.5, "0.0949% of Stage 2", 6.2, color=MUTED)
    rounded_box(c, 72, 31, 152, 23, GREEN_LIGHT, stroke=GREEN, radius=3.0)
    text_fitted(c, 148, 39.3, f"{funnel['directed_output']:,} exact directed IDs", 142,
                7.5, font="Helvetica-Bold", color=INK)
    down_arrow(c, 148, 63, 55, color=GREEN, width=1.15, head=4.0)
    text(c, 148, 21.5, "Exact counts; box widths are schematic.", 5.9,
         color=MUTED, align="center")

    # Log-latency chart. Bars encode median complete-denominator latency.
    chart_x0, chart_x1 = 347.0, 493.0
    lo, hi = 0.2, 12.0
    def xpos(value: float) -> float:
        return chart_x0 + (math.log10(value) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)
        ) * (chart_x1 - chart_x0)

    rows = [
        ("FaSTED", "fasted", GRAY_LIGHT),
        ("TensorJoin", "tensorjoin", BLUE),
        ("MiSTIC", "mistic", SKY),
        ("GDS-Join", "gds", GRAY),
    ]
    ys = [158, 126, 94, 62]
    for (label, key, fill), y in zip(rows, ys):
        value = float(methods[key]["median_s"])
        x_end = xpos(value)
        text(c, 299, y + 3.0, label, 6.8,
             font="Helvetica-Bold" if key == "tensorjoin" else "Helvetica", color=INK)
        c.setFillColor(fill)
        c.setStrokeColor(INK)
        c.setLineWidth(0.55)
        c.rect(chart_x0, y, max(2.0, x_end - chart_x0), 11, fill=1, stroke=1)
        if key == "fasted":
            hatch_rect(c, chart_x0, y, max(2.0, x_end - chart_x0), 11)
        label_x = min(x_end + 4, 469)
        text(c, label_x, y + 2.5, f"{value:.3f} s", 6.5,
             font="Helvetica-Bold" if key == "tensorjoin" else "Helvetica", color=INK)
        if key == "mistic":
            text(c, 402, y - 9.0, "5.083x slower [5.030, 5.133]", 5.9, color=MUTED)
        elif key == "gds":
            text(c, 402, y - 9.0, "11.783x slower [11.716, 11.846]", 5.9, color=MUTED)
        elif key == "fasted":
            text(c, 361, y - 9.0, "approx.: 598 FN + 1,130 FP", 5.9, color=MUTED)

    axis_y = 39.0
    set_stroke(c, INK, 0.55)
    c.line(chart_x0, axis_y, chart_x1, axis_y)
    for tick in (0.2, 1.0, 5.0, 10.0):
        x = xpos(tick)
        c.line(x, axis_y, x, axis_y - 3)
        text(c, x, axis_y - 11.5, f"{tick:g}", 5.8, color=MUTED, align="center")
    text(c, (chart_x0 + chart_x1) / 2, 17.0, "median latency (s, log scale; lower is better)",
         6.2, color=MUTED, align="center")

    c.setFillColor(LIGHT)
    c.rect(0, 0, W, 12.0, fill=1, stroke=0)
    text(c, W / 2, 3.3,
         "Evidence boundary: (a) audited G5 router; (b) separate G2B implementation - no shared-binary claim.",
         5.8, font="Helvetica-Bold", color=INK, align="center")
    c.showPage()
    c.save()
    return path


def stage_box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
              fill, title_value: str, lines: list[str], *, title_color=INK) -> None:
    rounded_box(c, x, y, w, h, fill, radius=3.5)
    text_fitted(c, x + w / 2, y + h - 14, title_value, w - 8, 7.5,
                font="Helvetica-Bold", color=title_color)
    set_stroke(c, GRID, 0.4)
    c.line(x + 5, y + h - 20, x + w - 5, y + h - 20)
    yy = y + h - 32
    for line in lines:
        text_fitted(c, x + w / 2, yy, line, w - 7, 6.25, color=INK)
        yy -= 10.2


def make_pipeline(data: dict[str, object]) -> Path:
    path = OUT / "fig2_exact_precision_routing_pipeline.pdf"
    c = canvas.Canvas(str(path), pagesize=(W, PIPELINE_H), pageCompression=1)
    c.setTitle("TensorJoin exact precision-routing query plan")
    c.setAuthor("TensorJoin artifact")
    f = data["g5_precision_funnel"]

    text(c, 14, 219, "Executable exact query plan: actual populations shrink before precision escalates",
         9.0, font="Helvetica-Bold", color=INK)

    # Persistent state / memory band.
    rounded_box(c, 13, 186, 484, 24, LIGHT, stroke=GRID, radius=2.5, width=0.55)
    text(c, 20, 199, "GPU HBM", 6.4, font="Helvetica-Bold", color=MUTED)
    text(c, 73, 199, "X: 60,000 x 512 FP32", 6.4, color=INK)
    c.setStrokeColor(GRID); c.line(181, 190, 181, 206)
    text(c, 191, 199, "Q: INT8 + per-vector scale/error", 6.4, color=INK)
    c.line(352, 190, 352, 206)
    text(c, 362, 199, "ID buffers + six counters", 6.4, color=INK)
    text(c, 255, 188.7, "fixed-capacity buffers persist across 108 batches", 5.6,
         color=MUTED, align="center")

    y, h = 87.0, 87.0
    xs = [13.0, 96.0, 229.0, 330.0, 405.0]
    ws = [75.0, 125.0, 93.0, 67.0, 92.0]
    stage_box(c, xs[0], y, ws[0], h, LIGHT, "Plan + quantize", [
        "440,391 upper tiles",
        "tile: 64 x 64",
        "K step: 64",
        "4,096 tiles/batch",
        "108 batches",
    ])
    stage_box(c, xs[1], y, ws[1], h, BLUE_LIGHT, "Stage 1 - INT8 certificate", [
        "Q_i[64x512] x Q_j^T[512x64]",
        "INT32 TC score S8[64x64]",
        "analytic interval [L8,U8]",
        f"accept {f['stage1_direct_accept']:,}",
        f"reject {f['stage1_direct_reject']:,}",
        f"ambiguous {f['stage1_ambiguous']:,}",
    ], title_color=BLUE)
    stage_box(c, xs[2], y, ws[2], h, ORANGE_LIGHT, "Stage 2 - FP32 filter", [
        f"input {f['stage1_ambiguous']:,}",
        "squared-distance [L32,U32]",
        f"accept {f['stage2_direct_accept']:,}",
        f"reject {f['stage2_direct_reject']:,}",
        f"residual {f['stage3_fp64']:,}",
    ], title_color=HexColor("#9A6700"))
    stage_box(c, xs[3], y, ws[3], h, SKY_LIGHT, "Stage 3 - FP64", [
        f"input {f['stage3_fp64']:,}",
        "exact d^2",
        f"accept {f['stage3_accept']:,}",
        f"reject {f['stage3_reject']:,}",
    ], title_color=BLUE)
    stage_box(c, xs[4], y, ws[4], h, GREEN_LIGHT, "Canonical output", [
        f"{f['accepted_upper']:,} upper IDs",
        "mirror off-diagonal",
        "+ 60,000 self IDs",
        "D2H concatenate + sort",
        f"{f['directed_output']:,} exact IDs",
    ], title_color=GREEN)

    # Ambiguity/residual route and count-dependent launch edges.
    mid_y = y + 49
    arrow(c, xs[0] + ws[0], mid_y, xs[1] - 4, mid_y, color=INK, width=0.9, head=3.6)
    arrow(c, xs[1] + ws[1], mid_y, xs[2] - 4, mid_y, color=ORANGE, width=1.15, head=3.8)
    arrow(c, xs[2] + ws[2], mid_y, xs[3] - 4, mid_y, color=ORANGE, width=1.15, head=3.8)
    arrow(c, xs[3] + ws[3], mid_y, xs[4] - 4, mid_y, color=GREEN, width=1.0, head=3.8)
    text(c, 274, 178.2, "host count reads set the two dynamic launch extents", 5.7,
         font="Helvetica-Bold", color=HexColor("#9A6700"), align="center")

    # Accepted IDs bypass later precision stages and join the output path.
    bypass_y = 78.0
    set_stroke(c, GREEN, 1.05)
    c.line(xs[1] + ws[1] / 2, bypass_y, xs[4] - 7, bypass_y)
    arrow(c, xs[4] - 7, bypass_y, xs[4] - 2, y + 15, color=GREEN, width=1.05, head=3.6)
    for idx in (1, 2, 3):
        down_arrow(c, xs[idx] + ws[idx] / 2, y, bypass_y, color=GREEN, width=0.9, head=3.2)
    text(c, 284, 70.0, "accepted ID chunks", 5.8, color=GREEN, align="center")

    # Bottom-baseline stage labels.
    base_y = 47.0
    set_stroke(c, GRID, 0.55)
    c.line(13, base_y + 17, 497, base_y + 17)
    names = [
        ("PLAN", "host/GPU preparation"),
        ("CERTIFY", "Tensor Cores + epilogue"),
        ("FILTER", "CUDA cores, FP32"),
        ("REFINE", "CUDA cores, FP64"),
        ("RECONSTRUCT", "D2H + host sort"),
    ]
    for x, w, (name, subtitle) in zip(xs, ws, names):
        text_fitted(c, x + w / 2, base_y + 5.5, name, w - 4, 6.7,
                    font="Helvetica-Bold", color=INK)
        text_fitted(c, x + w / 2, base_y - 5.0, subtitle, w - 3, 5.6, color=MUTED)

    c.setFillColor(LIGHT)
    c.rect(0, 0, W, 28.0, fill=1, stroke=0)
    text(c, W / 2, 16.0,
         "All populations are measured in the exact G5 P1 full-scale run; compilation is outside the public timer.",
         5.9, color=INK, align="center")
    text(c, W / 2, 6.2,
         "Only unresolved IDs advance; direct accepts bypass later precision stages and enter canonical reconstruction.",
         5.9, font="Helvetica-Bold", color=INK, align="center")
    c.showPage()
    c.save()
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for path in (make_teaser(data), make_pipeline(data)):
        print(path)


if __name__ == "__main__":
    main()
