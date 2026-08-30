#!/usr/bin/env python3
"""Reproduce non-authoritative extraction aids for Appendix B.5.

Neither output is read by the installer.  ``manual_review.tsv`` is the sole
transcription authority after visual comparison with the rendered PDF pages.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PDF = ROOT / "tmp/pdfs/korwa-kodaku-2022/source.pdf"
IMAGES = ROOT / "tmp/pdfs/korwa-kodaku-2022/wordlist-200dpi"


def main() -> None:
    reader = PdfReader(PDF)
    text_parts = []
    ocr_parts = []
    for pdf_page in range(66, 91):
        text = reader.pages[pdf_page - 1].extract_text() or ""
        text_parts.append(f"===== PDF_PAGE {pdf_page} =====\n{text.rstrip()}\n")
        image = IMAGES / f"page-{pdf_page:03d}.png"
        result = subprocess.run(
            ["tesseract", str(image), "stdout", "-l", "eng", "--psm", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        ocr_parts.append(f"===== PDF_PAGE {pdf_page} =====\n{result.stdout.rstrip()}\n")
    (HERE / "text_layer_scaffold.txt").write_text(
        "\n".join(text_parts), encoding="utf-8"
    )
    (HERE / "tesseract_scaffold.txt").write_text(
        "\n".join(ocr_parts), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
