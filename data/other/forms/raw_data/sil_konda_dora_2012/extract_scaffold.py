#!/usr/bin/env python3
"""Extract the scan's unreliable text layer as a non-authoritative locator."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
DEFAULT_PDF = DATA_ROOT.parent / "tmp/pdfs/konda_dora/silesr2012_016.pdf"
DEFAULT_OUTPUT = HERE / "ocr_scaffold.txt"
EXPECTED_SHA256 = "6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if sha256(args.pdf) != EXPECTED_SHA256:
        raise SystemExit("PDF hash mismatch; refusing to extract an unpinned representation")
    reader = PdfReader(args.pdf)
    if len(reader.pages) != 106:
        raise SystemExit(f"expected 106 pages, found {len(reader.pages)}")
    chunks = [
        "NON-AUTHORITATIVE OCR/TEXT-LAYER SCAFFOLD\n"
        "Never use this file as accepted transcription. The reviewed ledger was typed and "
        "verified cell by cell from rendered PDF pages 89-106.\n"
    ]
    for pdf_page in range(88, 107):
        text = reader.pages[pdf_page - 1].extract_text() or ""
        chunks.append(f"\n===== PHYSICAL PDF PAGE {pdf_page} =====\n{text.rstrip()}\n")
    args.output.write_text("".join(chunks), encoding="utf-8")
    print(f"wrote {args.output} from {args.pdf} sha256={EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
