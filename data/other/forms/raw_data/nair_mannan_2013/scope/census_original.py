"""Whole-chapter source-only page/font/layout evidence, not accepted transcription."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
PDF = ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    document = fitz.open(PDF)
    assert len(document) == 58
    assert sha(PDF) == "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
    originals = HERE.parent / "original-review"; originals.mkdir(exist_ok=True)
    records = []
    for printed in range(45, 50):
        page = document[printed - 1]
        image = originals / f"page-{printed}-150dpi.png"
        if not image.exists(): page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False).save(image)
        text = page.get_text("text")
        (HERE / f"page-{printed}-native.txt").write_text(text)
        (HERE / f"page-{printed}-rawdict.json").write_text(json.dumps(page.get_text("rawdict"), ensure_ascii=False))
        records.append({"printed_page": printed, "pdf_page": printed, "image": str(image.relative_to(HERE.parent)), "image_sha256": sha(image), "native_text_sha256": sha(HERE / f"page-{printed}-native.txt"), "font_inventory": page.get_fonts(), "status": "original evidence only; physical-unit census/literal review pending"})
    mentions = []
    for index, page in enumerate(document):
        text = page.get_text("text")
        if "mannan" in text.lower():
            mentions.append({"pdf_page": index + 1, "native_lines": [line for line in text.splitlines() if "mannan" in line.lower()], "inside_chapter": 45 <= index + 1 <= 49})
    (HERE / "page-and-font-census.json").write_text(json.dumps({"source_pdf_sha256": sha(PDF), "source_pdf_pages": len(document), "chapter_pages": records, "book_mannan_mentions": mentions, "canonical_mutations": False}, ensure_ascii=False, indent=2) + "\n")
    print("Original Mannan45–49 evidence rendered; whole-book mention leads recorded")


if __name__ == "__main__": main()
