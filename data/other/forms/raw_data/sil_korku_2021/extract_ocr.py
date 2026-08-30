#!/usr/bin/env python3
"""Extract Appendix F scans and reproduce non-authoritative OCR evidence."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/korku/JLSR2021_040.pdf"
DEFAULT_IMAGES = WORKSPACE / "tmp/pdfs/korku/images"
DEFAULT_ENHANCED = WORKSPACE / "tmp/pdfs/korku/enhanced"
RAW_OCR = HERE / "tesseract_raw.txt"
SHA256 = "d17426da3788d66c95f05824483941e7d5468e154c66d43c6354262fda00190d"
PAGE_FIRST = 44
PAGE_LAST = 100


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pdf(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing canonical PDF {path}; download SIL archive entry 90546")
    actual = digest(path)
    if actual != SHA256:
        raise SystemExit(f"source checksum drift: expected {SHA256}, found {actual}")


def extract(pdf: Path, image_dir: Path, enhanced_dir: Path) -> list[Path]:
    reader = PdfReader(pdf)
    if len(reader.pages) != 102:
        raise AssertionError(f"expected 102 PDF pages, found {len(reader.pages)}")
    image_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for page_number in range(PAGE_FIRST, PAGE_LAST + 1):
        images = reader.pages[page_number - 1].images
        if not images:
            raise AssertionError(f"PDF page {page_number} has no embedded scan")
        image = images[0].image.convert("L")
        raw = image_dir / f"pdf{page_number:03d}.png"
        image.save(raw, optimize=True)
        cleaned = ImageOps.autocontrast(image, cutoff=(1, 1)).resize(
            (image.width * 2, image.height * 2), resample=Image.Resampling.LANCZOS
        ).filter(ImageFilter.UnsharpMask(radius=1.4, percent=150, threshold=3))
        enhanced = enhanced_dir / f"pdf{page_number:03d}.png"
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
                check=True, text=True, capture_output=True,
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
    text = ocr(extract(args.pdf, args.images, args.enhanced))
    if args.write_ocr:
        RAW_OCR.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
