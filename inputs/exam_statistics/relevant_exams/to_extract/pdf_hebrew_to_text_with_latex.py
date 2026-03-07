#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract Hebrew PDF text + (best-effort) math expressions as LaTeX into a UTF-8 text file.

Main behavior:
1) If PDF has embedded text: extract it using PyMuPDF.
2) Optionally detect formula-like regions and run LaTeX OCR (pix2tex) to produce LaTeX.

Requirements:
  pip install pymupdf pillow
Optional (for LaTeX OCR):
  pip install pix2tex
  (pix2tex pulls torch; may be heavy)

Usage:
  python pdf_hebrew_to_text_with_latex.py input.pdf -o output.txt --latex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image


# -------------------------
# Text normalization helpers
# -------------------------

UNICODE_CONTROL_CHARS = [
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\ufeff",  # BOM/ZWNBSP
]

NBSP = "\u00a0"

def normalize_for_plain_text(s: str) -> str:
    """Make output copy/paste friendly: remove RTL marks, normalize whitespace/dashes."""
    for ch in UNICODE_CONTROL_CHARS:
        s = s.replace(ch, "")
    s = s.replace(NBSP, " ")
    # normalize “smart” punctuation to ASCII when possible
    s = s.replace("“", '"').replace("”", '"').replace("״", '"')
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    # collapse weird whitespace
    s = re.sub(r"[ \t]+", " ", s)
    return s


# -------------------------
# LaTeX OCR (optional)
# -------------------------

def load_latex_ocr():
    """Load pix2tex model if installed."""
    try:
        from pix2tex.cli import LatexOCR
    except Exception as e:
        raise RuntimeError(
            "pix2tex is not installed or failed to import.\n"
            "Install with: pip install pix2tex\n"
            "Note: this may require PyTorch and can be large."
        ) from e
    return LatexOCR()


def rect_area(r: fitz.Rect) -> float:
    return max(0.0, (r.x1 - r.x0)) * max(0.0, (r.y1 - r.y0))


def find_formula_like_blocks(page: fitz.Page) -> List[fitz.Rect]:
    """
    Heuristic: find blocks with high symbol density / short words / many operators.
    This is imperfect but works surprisingly well on DS exams.
    """
    blocks = page.get_text("blocks")  # (x0,y0,x1,y1,"text", block_no, block_type)
    rects: List[fitz.Rect] = []

    # Common mathy tokens:
    mathy = re.compile(r"[\=\+\-\*/\^_{}\[\]\(\)<>≤≥∈∑∏√∞≈≠→←↦]|\\[a-zA-Z]+")
    # Hebrew letters range:
    heb = re.compile(r"[\u0590-\u05FF]")

    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        text = text or ""
        t = text.strip()
        if not t:
            continue

        # Score math-likeness
        math_hits = len(mathy.findall(t))
        heb_hits = len(heb.findall(t))
        length = max(1, len(t))

        # A block that has many math symbols relative to length, and not mostly Hebrew
        math_ratio = math_hits / length
        heb_ratio = heb_hits / length

        # Tune thresholds as needed
        if math_hits >= 6 and math_ratio >= 0.08 and heb_ratio <= 0.35:
            r = fitz.Rect(x0, y0, x1, y1)
            # Skip huge blocks (likely entire paragraphs)
            if rect_area(r) < 0.25 * rect_area(page.rect):
                rects.append(r)

    # Merge nearby rects (simple pass)
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged: List[fitz.Rect] = []
    for r in rects:
        if not merged:
            merged.append(r)
            continue
        last = merged[-1]
        # if close vertically and overlapping horizontally, merge
        if abs(r.y0 - last.y1) < 10 and (r.x0 <= last.x1 and r.x1 >= last.x0):
            merged[-1] = last | r
        else:
            merged.append(r)

    return merged


def render_region(page: fitz.Page, rect: fitz.Rect, zoom: float = 2.5) -> Image.Image:
    """Render a region of a page to a PIL image."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


# -------------------------
# Main extraction
# -------------------------

def extract_pdf(
    pdf_path: Path,
    include_latex: bool,
    max_pages: Optional[int] = None,
) -> str:
    doc = fitz.open(pdf_path)

    latex_model = None
    if include_latex:
        latex_model = load_latex_ocr()

    out_lines: List[str] = []
    total_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)

    for i in range(total_pages):
        page = doc.load_page(i)
        out_lines.append(f"===== PAGE {i+1} =====")

        # 1) Extract embedded text (fast, best for Hebrew when present)
        text = page.get_text("text") or ""
        text = normalize_for_plain_text(text).strip()

        if text:
            out_lines.append(text)
        else:
            out_lines.append("[NO EMBEDDED TEXT DETECTED ON THIS PAGE]")

        # 2) Extract formulas as LaTeX (optional)
        if include_latex and latex_model is not None:
            rects = find_formula_like_blocks(page)
            if rects:
                out_lines.append("")
                out_lines.append("---- DETECTED FORMULAS (LaTeX OCR, best-effort) ----")
                for k, r in enumerate(rects, start=1):
                    try:
                        img = render_region(page, r, zoom=3.0)
                        latex = latex_model(img)
                        latex = (latex or "").strip()
                        latex = normalize_for_plain_text(latex)
                        if latex:
                            # Wrap in $...$ for consistency
                            out_lines.append(f"[FORMULA {k}] ${latex}$")
                    except Exception as e:
                        out_lines.append(f"[FORMULA {k}] [ERROR: {type(e).__name__}: {e}]")

        out_lines.append("")  # blank line between pages

    return "\n".join(out_lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Extract Hebrew PDF text + optional LaTeX math into a text file.")
    parser.add_argument("pdf", type=str, help="Path to input PDF")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output text file path (default: same name .txt)")
    parser.add_argument("--latex", action="store_true", help="Also run LaTeX OCR on detected formula regions (pix2tex)")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only first N pages")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else pdf_path.with_suffix(".txt")

    text = extract_pdf(pdf_path, include_latex=args.latex, max_pages=args.max_pages)

    out_path.write_text(text, encoding="utf-8", errors="replace")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
