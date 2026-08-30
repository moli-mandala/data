#!/usr/bin/env python3
"""Extract the embedded Koya word-list scans and make non-authoritative OCR passes.

The OCR is only a structural/base-letter scaffold. ``manual_review_data.py`` is
the authoritative transcription and must be checked against every source image.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
DEFAULT_PDF = REPO / "tmp/pdfs/koya/JLSR2021_029.pdf"
DEFAULT_IMAGES = REPO / "tmp/pdfs/koya/images"
DEFAULT_ENHANCED = REPO / "tmp/pdfs/koya/enhanced"
RAW_OCR = HERE / "tesseract_raw.txt"
SHA256 = "a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d"
PAGE_FIRST = 80
PAGE_LAST = 123


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pdf(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(
            f"missing canonical source PDF {path}; download SIL archive entry 88873 "
            "as JLSR2021_029.pdf"
        )
    actual = digest(path)
    if actual != SHA256:
        raise SystemExit(f"source checksum drift: expected {SHA256}, found {actual}")


def extract(pdf: Path, image_dir: Path, enhanced_dir: Path) -> list[Path]:
    reader = PdfReader(pdf)
    if len(reader.pages) != 124:
        raise AssertionError(f"expected 124 PDF pages, found {len(reader.pages)}")
    image_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_number in range(PAGE_FIRST, PAGE_LAST + 1):
        images = reader.pages[page_number - 1].images
        if not images:
            raise AssertionError(f"PDF page {page_number} has no embedded scan")
        # Pages 115 and 123 contain a duplicate image object.  The first object
        # is the visible full-page scan; preserve the duplicate count in OCR metadata.
        image = images[0].image.convert("L")
        raw = image_dir / f"pdf{page_number:03d}-0.png"
        image.save(raw, optimize=True)
        cleaned = ImageOps.autocontrast(image, cutoff=(1, 1))
        cleaned = cleaned.resize(
            (cleaned.width * 3, cleaned.height * 3), Image.Resampling.LANCZOS
        ).filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=3))
        enhanced = enhanced_dir / f"pdf{page_number:03d}-0.png"
        cleaned.save(enhanced, optimize=True)
        paths.append(enhanced)
    return paths


def ocr(images: list[Path]) -> str:
    blocks = []
    for path in images:
        page = int(path.stem[3:6])
        for psm in (4, 6):
            result = subprocess.run(
                ["tesseract", str(path), "stdout", "--psm", str(psm), "-l", "eng"],
                check=True,
                text=True,
                capture_output=True,
            )
            blocks.append(f"===== PDF {page} PSM {psm} =====\n{result.stdout.rstrip()}\n")
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED)
    parser.add_argument("--write-ocr", action="store_true")
    args = parser.parse_args()
    require_pdf(args.pdf)
    images = extract(args.pdf, args.images, args.enhanced)
    text = ocr(images)
    if args.write_ocr:
        RAW_OCR.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
