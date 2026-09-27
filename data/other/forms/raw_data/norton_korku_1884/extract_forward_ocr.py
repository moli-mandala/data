"""Extract review-only OCR for the complete Cust/Norton forward vocabulary.

The text layer is a sequencing aid, not the transcription authority. Every
entry still needs comparison with the printed page before installation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pdfplumber


EXPECTED_SHA256 = "13a46b83a9e5dbc381ea29ec84f16125e4da58c2374d0806391b90a701ea04a6"
FIRST_PRINTED_PAGE = 165
LAST_PRINTED_PAGE = 172
PDF_PAGE_OFFSET = 19


def extract(pdf_path: Path) -> str:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected scan SHA256: {digest}")
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for printed_page in range(FIRST_PRINTED_PAGE, LAST_PRINTED_PAGE + 1):
            pdf_page = printed_page + PDF_PAGE_OFFSET
            raw = pdf.pages[pdf_page - 1].extract_text(layout=True) or ""
            chunks.append(f"=== printed page {printed_page}; PDF page {pdf_page} ===\n{raw.rstrip()}\n")
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("forward_ocr.txt"))
    args = parser.parse_args()
    args.output.write_text(extract(args.pdf), encoding="utf-8")


if __name__ == "__main__":
    main()
