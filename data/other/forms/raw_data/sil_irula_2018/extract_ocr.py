#!/usr/bin/env python3
"""Render and OCR the image-only Appendix B of SIL ESR 2018-010.

The publisher PDF is intentionally not redistributed.  This script verifies the
local copy, renders the 24 word-list pages, crops their three printed columns,
and records a deterministic Tesseract pass used only as a transcription
scaffold.  The checked-in transcription/audit remains authoritative for IPA.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
WORKSPACE = REPO.parent
PDF = WORKSPACE / "tmp/pdfs/sil-surveys/silesr2018_010.pdf"
EXPECTED_SHA256 = "2e5a4ef0f4c941437d09a1c8fa49ba01d4fe79e0915ad9248ac7b83280fb4c62"
PDF_FIRST = 30
PDF_LAST = 53
PRINTED_PAGE_OFFSET = -5  # PDF 30 is printed page 25.
RENDER_DPI = 300

# Fractions of the rendered PDF page.  The three source columns are well
# separated; stopping before the following column matters because otherwise its
# site codes are falsely appended to short forms near the right edge.
COLUMNS = ((0.10, 0.355), (0.365, 0.625), (0.635, 0.930))
Y_BOUNDS = (0.055, 0.955)


def verify_pdf() -> None:
    if not PDF.exists():
        raise SystemExit(
            f"Missing {PDF}. Download silesr2018_010.pdf from SIL archive 76656 "
            "and place it at that path."
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


def render_and_crop(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jambu-irula-render-") as temp:
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
        if len(rendered) != PDF_LAST - PDF_FIRST + 1:
            raise RuntimeError(f"expected 24 rendered pages, found {len(rendered)}")
        crops: list[Path] = []
        for pdf_page, page_path in zip(range(PDF_FIRST, PDF_LAST + 1), rendered):
            with Image.open(page_path) as page:
                width, height = page.size
                for column, (left, right) in enumerate(COLUMNS, 1):
                    crop = page.crop(
                        (
                            int(width * left), int(height * Y_BOUNDS[0]),
                            int(width * right), int(height * Y_BOUNDS[1]),
                        )
                    ).convert("L")
                    crop = ImageOps.autocontrast(crop, cutoff=(0.1, 0.1))
                    target = output_dir / f"pdf{pdf_page:02d}-p{pdf_page + PRINTED_PAGE_OFFSET:02d}-c{column}.png"
                    crop.save(target)
                    crops.append(target)
        return crops


def ocr(crops: list[Path], output: Path) -> None:
    blocks = []
    for crop in crops:
        proc = subprocess.run(
            [
                # Homebrew Tesseract on this macOS host mishandles an absolute
                # /tmp pathname; run beside the crop and pass only its basename.
                "tesseract", crop.name, "stdout", "-l", "eng", "--psm", "6",
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
    args = parser.parse_args()
    verify_pdf()
    crops = render_and_crop(args.output_dir)
    if args.ocr_output:
        ocr(crops, args.ocr_output)
    print(f"rendered {len(crops)} column crops from {PDF_FIRST}-{PDF_LAST}")


if __name__ == "__main__":
    main()
