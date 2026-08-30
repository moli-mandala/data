#!/usr/bin/env python3
"""Extract the image-only Bagheli WordSurv appendix and make OCR review scaffolds.

The official JLSR 2022-015 PDF embeds each wordlist page as one lossless raster image.
OCR is deliberately *not* an installation input: it is saved only beside the manual
transcription so that reviewers can spot disagreements. The checked-in manual ledger
is the authoritative lexical snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
PDF = WORKSPACE / "tmp/pdfs/bagheli/JLSR2022_015.pdf"
IMAGE_DIR = WORKSPACE / "tmp/pdfs/bagheli/source-images"
CROP_DIR = WORKSPACE / "tmp/pdfs/bagheli/ocr-columns"
IMAGE_MANIFEST = HERE / "image_manifest.tsv"
OCR_OUTPUT = HERE / "tesseract_scaffold.txt"

PDF_SHA256 = "d1424f317dc12fe01d99d33abd917201575487f4de44529678ecce1c282a4627"
PDF_PAGES = 161
FIRST_WORDLIST_PAGE = 59
LAST_WORDLIST_PAGE = 81


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pdf() -> PdfReader:
    if not PDF.exists():
        raise SystemExit(
            f"Missing {PDF}. Download JLSR2022_015.pdf from SIL archive record 94596."
        )
    digest = sha256(PDF)
    if digest != PDF_SHA256:
        raise ValueError(f"Unexpected PDF SHA-256 {digest}; expected {PDF_SHA256}")
    reader = PdfReader(PDF)
    if len(reader.pages) != PDF_PAGES:
        raise ValueError(f"Expected {PDF_PAGES} PDF pages, got {len(reader.pages)}")
    return reader


def clean_crop(image: Image.Image) -> Image.Image:
    """Upscale without inventing strokes and gently flatten the pale scan background."""
    gray = ImageOps.grayscale(image)
    background = gray.filter(ImageFilter.GaussianBlur(radius=18))
    flattened = ImageOps.autocontrast(ImageOps.invert(ImageChops.subtract(background, gray)))
    flattened = ImageEnhance.Contrast(flattened).enhance(1.25)
    return flattened.resize((flattened.width * 3, flattened.height * 3), Image.Resampling.LANCZOS)


def extract(reader: PdfReader) -> list[dict[str, str]]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    raw_blocks: list[str] = []

    for pdf_page in range(FIRST_WORDLIST_PAGE, LAST_WORDLIST_PAGE + 1):
        images = reader.pages[pdf_page - 1].images
        if len(images) != 1:
            raise ValueError(f"Expected one embedded wordlist image on PDF page {pdf_page}")
        source = images[0].image.convert("RGB")
        source_path = IMAGE_DIR / f"pdf-{pdf_page:03d}.png"
        source.save(source_path)
        manifest.append({
            "PDF_Page": str(pdf_page),
            "Printed_Page": str(pdf_page - 9),
            "Image_Name": images[0].name,
            "Width": str(source.width),
            "Height": str(source.height),
            "Image_SHA256": sha256(source_path),
        })

        # The exported WordSurv page is four equal-flow columns.  A narrow overlap is
        # intentionally avoided because duplicate bracket codes would be dangerous.
        edges = [round(source.width * i / 4) for i in range(5)]
        for column in range(4):
            crop = source.crop((edges[column], 0, edges[column + 1], source.height))
            crop = clean_crop(crop)
            crop_path = CROP_DIR / f"pdf-{pdf_page:03d}-c{column + 1}.png"
            crop.save(crop_path)
            cmd = [
                "tesseract", str(crop_path), "stdout", "-l", "script/Latin",
                "--psm", "6", "-c", "preserve_interword_spaces=1",
            ]
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            raw_blocks.append(
                f"=== PDF_PAGE {pdf_page} PRINTED_PAGE {pdf_page - 9} COLUMN {column + 1} ===\n"
                f"{result.stdout.rstrip()}\n"
            )

    with IMAGE_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("PDF_Page", "Printed_Page", "Image_Name", "Width", "Height", "Image_SHA256"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest)
    OCR_OUTPUT.write_text("\n".join(raw_blocks), encoding="utf-8")
    print(f"pages={len(manifest)} columns={len(raw_blocks)} ocr={OCR_OUTPUT}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    extract(verify_pdf())


if __name__ == "__main__":
    main()
