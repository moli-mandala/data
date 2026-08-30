#!/usr/bin/env python3
"""Extract the photographed handwritten Appendix C sheets from JLSR 2021-009.

The extractor creates deterministic page and cell crops for manual review. OCR
is deliberately a scaffold only; installed forms must come from the checked
manual transcription file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
WORKSPACE = REPO.parent
PDF = WORKSPACE / "tmp/pdfs/kullu/JLSR2021_009.pdf"
EXPECTED_SHA256 = "720a97198254160bfa88a9557b33955b2814878e346901ff399cacc53d5c4fdd"

# Each sixteen-item block consists of one English prompt sheet and six source
# sheets. The last block has six items (193-198), for 198 prompts in total.
FIRST_PROMPT_PAGE = 33
BLOCK_SIZE = 7
BLOCKS = 13
SITE_LAYOUT = (
    (None, "CHU", "LOR"),
    ("SHA", "CHI", "SHG"),
    ("MAN", "RAI", "MAR"),
    ("SID", "JIB", "BAT"),
    ("GAR", "KUL", "BHU"),
    ("MNK", "ANI", None),
)
SITES = {
    "CHU": "Churla / Lag Valley",
    "LOR": "Loren / South Kullu Valley",
    "SHA": "Shalwar Village",
    "CHI": "Chinninal Village",
    "SHG": "Shangarh",
    "MAN": "Manali",
    "RAI": "Raila Village",
    "MAR": "Maraur Village",
    "SID": "Sidua",
    "JIB": "Jibhi",
    "BAT": "Bathad",
    "GAR": "Garsah",
    "KUL": "Kullu (District Headquarters)",
    "BHU": "Bhutti Village / Lag Valley",
    "MNK": "Manikaran",
    "ANI": "Ani",
}

FIELDS = [
    "Item", "Site", "Site_Name", "PDF_Page", "Printed_Page", "Column",
    "Cell_Image", "Raw_OCR", "OCR_Alternates", "Transcription", "Review",
    "Uncertainty", "Blank", "Notes",
]


def verify_pdf() -> PdfReader:
    if not PDF.exists():
        raise SystemExit(
            f"Missing {PDF}; download JLSR2021_009.pdf from SIL archive record 88003."
        )
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"PDF SHA-256 is {digest}, expected {EXPECTED_SHA256}")
    reader = PdfReader(PDF)
    if len(reader.pages) != 126:
        raise AssertionError(f"expected 126 PDF pages, found {len(reader.pages)}")
    return reader


def extract(output_dir: Path, scaffold: Path) -> None:
    reader = verify_pdf()
    pages_dir = output_dir / "pages"
    cells_dir = output_dir / "cells"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for block in range(BLOCKS):
        first_item = block * 16 + 1
        count = 6 if block == BLOCKS - 1 else 16
        prompt_page = FIRST_PROMPT_PAGE + block * BLOCK_SIZE
        for pdf_page in range(prompt_page, prompt_page + BLOCK_SIZE):
            images = reader.pages[pdf_page - 1].images
            if not images:
                raise AssertionError(f"PDF page {pdf_page} has no photographed sheet")
            image = images[0].image.convert("RGB")
            page_path = pages_dir / f"pdf{pdf_page:03d}.png"
            image.save(page_path)
            if pdf_page == prompt_page:
                continue
            layout = SITE_LAYOUT[pdf_page - prompt_page - 1]
            width, height = image.size
            if not (1170 <= width <= 1190 and 1760 <= height <= 1780):
                raise AssertionError(
                    f"unexpected sheet dimensions on PDF page {pdf_page}: {image.size}"
                )
            # The source's ruled forms have a 150 px header and 100 px rows in
            # the embedded ~1180 x 1770 image. Crop with a small vertical
            # overlap so high/low diacritics at boundaries remain visible.
            for column, site in enumerate(layout):
                if site is None:
                    continue
                x0 = round(width * column / 3)
                x1 = round(width * (column + 1) / 3)
                for offset in range(count):
                    item = first_item + offset
                    y0 = max(0, 145 + 100 * offset)
                    y1 = min(height, 255 + 100 * offset)
                    crop = image.crop((x0, y0, x1, y1))
                    relative = Path("cells") / f"i{item:03d}-{site}-pdf{pdf_page:03d}.png"
                    crop.save(output_dir / relative)
                    rows.append({
                        "Item": item,
                        "Site": site,
                        "Site_Name": SITES[site],
                        "PDF_Page": pdf_page,
                        "Printed_Page": pdf_page - 7,
                        "Column": column + 1,
                        "Cell_Image": str(relative),
                        "Raw_OCR": "",
                        "OCR_Alternates": "",
                        "Transcription": "",
                        "Review": "pending manual source-image review",
                        "Uncertainty": "",
                        "Blank": "",
                        "Notes": "",
                    })

    expected = 198 * len(SITES)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} source cells, found {len(rows)}")
    if len({(row["Item"], row["Site"]) for row in rows}) != expected:
        raise AssertionError("duplicate or missing item/site source cell")
    with scaffold.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["Item"], row["Site"])))
    print(
        f"pages={BLOCKS * BLOCK_SIZE} target_sites={len(SITES)} "
        f"items=198 cells={len(rows)} scaffold={scaffold}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    args = parser.parse_args()
    extract(args.output_dir, args.scaffold)


if __name__ == "__main__":
    main()
