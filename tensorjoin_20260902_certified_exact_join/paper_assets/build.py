#!/usr/bin/env python3
"""Rebuild and validate all TensorJoin paper assets with one command."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets"
PYTHON = sys.executable


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def render(pdf: str, prefix: str, dpi: int) -> Path:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required for rendered QA")
    output = ASSETS / "qa" / f"{prefix}.png"
    run(pdftoppm, "-png", "-singlefile", "-r", str(dpi), pdf, str(output.with_suffix("")))
    return output


def main() -> None:
    run(PYTHON, "paper_assets/src/extract_paper_numbers.py")
    run(PYTHON, "paper_assets/src/make_paper_figures.py")
    run(PYTHON, "paper_assets/src/make_main_table.py")

    for pdf, prefix in (
        ("paper_assets/figures/fig1_evidence_separated_teaser.pdf", "fig1_600dpi"),
        ("paper_assets/figures/fig2_exact_precision_routing_pipeline.pdf", "fig2_600dpi"),
        ("paper_assets/figures/table1_external_full_public.pdf", "table1_600dpi"),
    ):
        image = render(pdf, prefix, 600)
        Image.open(image).convert("L").save(image.with_name(image.stem + "_gray.png"))

    run(PYTHON, "paper_assets/src/make_qa_pages.py")
    render("paper_assets/qa/paper_figures_placement.pdf", "paper_figures_placement_200dpi", 200)
    render("paper_assets/qa/paper_table_placement.pdf", "paper_table_placement_200dpi", 200)
    run(PYTHON, "paper_assets/src/validate_paper_assets.py")
    print("paper-assets build and QA: PASS")


if __name__ == "__main__":
    main()
