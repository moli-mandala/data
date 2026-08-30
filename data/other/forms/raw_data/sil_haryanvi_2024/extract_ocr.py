#!/usr/bin/env python3
"""Render and structurally OCR Appendix A.3 of JLSR 2024-011.

The fourteen wordlist pages are raster images embedded in an otherwise digital
PDF.  Tesseract is used only to produce a reproducible transcription scaffold;
the checked source-facing transcription remains authoritative for phonetic
symbols and diacritics.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
WORKSPACE = REPO.parent
PDF = WORKSPACE / "tmp/pdfs/haryanvi/JLSR2024_011.pdf"
EXPECTED_SHA256 = "53121a1b9803ba502092866080e3bdb35457bc6040dcc7f47da508eca1fef2e2"
PDF_FIRST = 28
PDF_LAST = 41
PRINTED_PAGE_OFFSET = -7
RENDER_DPI = 240

def verify_pdf() -> None:
    if not PDF.exists():
        raise SystemExit(
            f"Missing {PDF}. Download JLSR2024_011.pdf from SIL and place it there."
        )
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"{PDF} SHA-256 is {digest}, expected {EXPECTED_SHA256}")


def find_pdftoppm() -> str:
    candidates = (
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm",
        Path("/opt/homebrew/bin/pdftoppm"),
        Path("/usr/local/bin/pdftoppm"),
        Path("/usr/bin/pdftoppm"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit("pdftoppm is required to render the SIL survey")


def adaptive_binary(crop: Image.Image) -> Image.Image:
    """Remove uneven photographed-paper shading without erasing diacritics."""
    gray = crop.convert("L")
    # Upsampling does not invent source detail, but it gives Tesseract enough
    # pixels to keep the typewriter's underdots, underlines and superscripts.
    gray = gray.resize((gray.width * 2, gray.height * 2))
    background = gray.filter(ImageFilter.GaussianBlur(24))
    difference = ImageChops.subtract(background, gray)
    return difference.point(lambda value: 255 if value < 18 else 0)


def photographed_table_box(page: Image.Image) -> tuple[int, int, int, int]:
    """Locate the large gray photographed rectangle inside a rendered page."""
    gray = page.convert("L")
    width, height = gray.size
    pixels = gray.load()
    qualifying_rows = []
    for y in range(height):
        dark = sum(pixels[x, y] < 235 for x in range(width))
        if dark > width * 0.40:
            qualifying_rows.append(y)
    if not qualifying_rows:
        raise RuntimeError("could not locate photographed wordlist table")
    y0, y1 = min(qualifying_rows), max(qualifying_rows) + 1
    qualifying_columns = []
    for x in range(width):
        dark = sum(pixels[x, y] < 235 for y in range(y0, y1))
        if dark > (y1 - y0) * 0.40:
            qualifying_columns.append(x)
    if not qualifying_columns:
        raise RuntimeError("could not locate photographed wordlist columns")
    return min(qualifying_columns), y0, max(qualifying_columns) + 1, y1


def render_and_crop(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jambu-haryanvi-render-") as temp:
        prefix = Path(temp) / "page"
        subprocess.run(
            [
                find_pdftoppm(), "-f", str(PDF_FIRST), "-l", str(PDF_LAST),
                "-r", str(RENDER_DPI), "-png", str(PDF), str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        rendered = sorted(Path(temp).glob("page-*.png"))
        expected = PDF_LAST - PDF_FIRST + 1
        if len(rendered) != expected:
            raise RuntimeError(f"expected {expected} rendered pages, found {len(rendered)}")
        crops: list[Path] = []
        for pdf_page, page_path in zip(range(PDF_FIRST, PDF_LAST + 1), rendered):
            with Image.open(page_path) as page:
                left, top, right, bottom = photographed_table_box(page)
                table_width, table_height = right - left, bottom - top
                for row in range(5):
                    for column in range(3):
                        item = (pdf_page - PDF_FIRST) * 15 + row * 3 + column + 1
                        x0 = left + round(table_width * column / 3)
                        x1 = left + round(table_width * (column + 1) / 3)
                        y0 = top + round(table_height * row / 5)
                        y1 = top + round(table_height * (row + 1) / 5)
                        # A few source alternates touch a cell boundary. Tiny
                        # overlaps retain them without reaching adjacent rows.
                        crop = page.crop(
                            (
                                max(left, x0 - 8), max(top, y0 - 3),
                                min(right, x1 + 8), min(bottom, y1 + 3),
                            )
                        )
                        target = output_dir / (
                            f"item{item:03d}-pdf{pdf_page:02d}-"
                            f"p{pdf_page + PRINTED_PAGE_OFFSET:02d}-c{column + 1}.png"
                        )
                        adaptive_binary(crop).save(target)
                        crops.append(target)
        return crops


def ocr(crops: list[Path], output: Path, psm: int, language: str) -> None:
    blocks = []
    for crop in crops:
        proc = subprocess.run(
            [
                "tesseract", crop.name, "stdout", "-l", language, "--psm", str(psm),
                "-c", "preserve_interword_spaces=1",
            ],
            check=True,
            text=True,
            errors="replace",
            capture_output=True,
            cwd=crop.parent,
        )
        blocks.append(f"@@ {crop.stem}\n{proc.stdout.rstrip()}\n")
    output.write_text("\n".join(blocks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ocr-output", type=Path)
    parser.add_argument("--psm", type=int, default=4)
    parser.add_argument("--language", default="eng")
    parser.add_argument(
        "--reuse-crops",
        action="store_true",
        help="reuse the 210 existing item crops instead of rerendering the PDF",
    )
    args = parser.parse_args()
    verify_pdf()
    if args.reuse_crops:
        crops = sorted(args.output_dir.glob("item*.png"))
        if len(crops) != 210:
            raise SystemExit(f"expected 210 reusable crops, found {len(crops)}")
    else:
        crops = render_and_crop(args.output_dir)
    if args.ocr_output:
        ocr(crops, args.ocr_output, args.psm, args.language)
    action = "reused" if args.reuse_crops else "rendered"
    print(
        f"{action} {len(crops)} item crops from PDF pages {PDF_FIRST}-{PDF_LAST}"
        + (f"; OCR {args.language} PSM {args.psm}" if args.ocr_output else "")
    )


if __name__ == "__main__":
    main()
