#!/usr/bin/env python3
"""Build the resumable Kurumba Appendix C review scaffold.

The PDF OCR layer is deliberately kept in ``OCR_*`` fields only.  This script
never populates ``Manual_Form`` and its output must not be installed.  Run with
``--initialize`` only once; it refuses to overwrite the manual ledger.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/kurumba_2012/silesr2012_015.pdf"
LISTS_FILE = HERE / "list_registry.tsv"
PAGES_FILE = HERE / "page_review.tsv"
PROMPTS_FILE = HERE / "prompt_review.tsv"
MANUAL_FILE = HERE / "manual_transcription.tsv"
OCR_TEXT = HERE / "ocr_layer_scaffold.txt"


@dataclass(frozen=True)
class ListSpec:
    key: str
    label: str
    language: str
    dialect: str
    scope: str
    classification: str
    pages: tuple[int, int]
    column: str
    location: str


LISTS = [
    ListSpec("tamil_madras", "Standard Tamil, Madras", "Tamil", "sil-kurumba-1985-tamil-madras", "control", "Standard Tamil comparison", (217, 238), "left", "Madras variety; elicited 25 January 1985"),
    ListSpec("kannada_bangalore", "Standard Kannada, Bangalore", "Kannada", "sil-kurumba-1985-kannada-bangalore", "control", "Standard Kannada comparison", (217, 238), "right", "Bangalore variety; elicited 22 February 1985"),
    ListSpec("belavarthy", "SNSK, Belavarthy Kurumba", "Kannada", "sil-kurumba-1984-belavarthy", "target", "Southern Nonstandard Kannada", (239, 260), "left", "Belavarthy; Krishnagiri taluk; Dharmapuri district; Tamil Nadu"),
    ListSpec("pudukkottai", "SNSK, Pudukkottai Kurumba", "Kannada", "sil-kurumba-1976-pudukkottai", "target", "Southern Nonstandard Kannada", (239, 260), "right", "Pudukkottai district; Tamil Nadu; list dated 1976-1977"),
    ListSpec("kotagiri_alu", "ANSK, Kotagiri Alu Kurumba", "AluKurumba", "sil-kurumba-1985-kotagiri-alu", "target", "Alu Kurumba Nonstandard Kannada", (261, 282), "left", "Banigudisole village; Kotagiri taluk; Nilgiris district; Tamil Nadu"),
    ListSpec("badaga_arvenu", "Badaga, Arvenu/Kotagiri", "Badaga", "sil-kurumba-1985-badaga-arvenu", "control", "Badaga comparison", (261, 282), "right", "Arvenu; Kotagiri taluk; Nilgiris district; Tamil Nadu"),
    ListSpec("kolar_kuruba", "SNSK, Kolar Kurubas", "Kannada", "sil-kurumba-1985-kolar", "target", "Southern Nonstandard Kannada", (283, 304), "left", "Basavanatha village; Kolar district; Karnataka"),
    ListSpec("chitradurga_kuruba", "NSK, Chitradurga Kurubas", "Kannada", "sil-kurumba-1985-chitradurga", "target", "Nonstandard Kannada (CNSK in report analysis)", (283, 304), "right", "Malappanahatti village; Chitradurga district; Karnataka"),
    ListSpec("buringi", "SNSK, Buringi Kurumba", "Kannada", "sil-kurumba-1984-buringi", "target", "Southern Nonstandard Kannada", (305, 326), "left", "Buringi village; Tiruppattur taluk; North Arcot district; Tamil Nadu"),
    ListSpec("madapalli", "SNSK, Madapalli Kurumba", "Kannada", "sil-kurumba-1984-madapalli", "target", "Southern Nonstandard Kannada", (305, 326), "right", "Madapalli village; Tiruppattur taluk; North Arcot district; Tamil Nadu"),
    ListSpec("kurumbatheru", "SNSK, Kurumbatheru Kannada", "Kannada", "sil-kurumba-1984-kurumbatheru", "target", "Southern Nonstandard Kannada", (327, 348), "left", "Kurumbatheru hamlet; Kandikuppam village; Krishnagiri taluk; Dharmapuri district; Tamil Nadu"),
    ListSpec("thangiyadikuppam", "SNSK, Thangiyadikuppam Kurumba", "Kannada", "sil-kurumba-1984-thangiyadikuppam", "target", "Southern Nonstandard Kannada", (327, 348), "right", "Thangiyadikuppam village; Kuppam taluk; Chittoor district; Andhra Pradesh"),
    ListSpec("beerajjanur", "SNSK, Beerajjanur Kurumba", "Kannada", "sil-kurumba-1985-beerajjanur", "target", "Southern Nonstandard Kannada", (349, 370), "left", "Beerajjanur village; Krishnagiri taluk; Dharmapuri district; Tamil Nadu"),
    ListSpec("karmadai_kurumba", "SNSK, Karmadai Kurumba", "Kannada", "sil-kurumba-1985-karmadai-kurumba", "target", "Southern Nonstandard Kannada", (349, 370), "right", "Karamadai; Mettupalayam taluk; Coimbatore district; Tamil Nadu"),
    ListSpec("karmadai_vakkaliga", "SNSK, Karmadai Vakkaliga", "Kannada", "sil-kurumba-1985-karmadai-vakkaliga", "control", "Vakkaliga comparison", (371, 392), "left", "Karmadai; Mettupalayam taluk; Coimbatore district; Tamil Nadu"),
    ListSpec("kurumbapalayam", "SNSK, Kurumbapalayam Kurumba", "Kannada", "sil-kurumba-1985-kurumbapalayam", "target", "Southern Nonstandard Kannada", (371, 392), "right", "Kurumbapalayam village; Coimbatore district; Tamil Nadu"),
    ListSpec("kalangal", "SNSK, Kalangal Kurumba", "Kannada", "sil-kurumba-1985-kalangal", "target", "Southern Nonstandard Kannada", (393, 414), "left", "Kalangal village; Palladam taluk; Coimbatore district; Tamil Nadu"),
    ListSpec("masinagudi_jennu", "JNSK, Masinagudi Jennu Kurumba", "Kannada", "sil-kurumba-1985-masinagudi-jennu", "target", "Jennu Nonstandard Kannada", (393, 414), "right", "Masinagudi village; Gudalur taluk; Nilgiris district; Tamil Nadu"),
    ListSpec("maddur_betta", "NST, Maddur Colony Betta Kurumba", "BettaKurumba", "sil-kurumba-1985-maddur-betta", "target", "Nonstandard Tamil (report classification)", (415, 436), "left", "Maddur Colony; Gundlupet taluk; Mysore district; Karnataka"),
]

LIST_FIELDS = ["List_Key", "Source_Label", "Language_ID", "Dialect_ID", "Scope", "Report_Classification", "PDF_First", "PDF_Last", "Column", "Location"]
PAGE_FIELDS = ["PDF_Page", "Printed_Page", "Items_First", "Items_Last", "Left_List", "Right_List", "Conceptual_Cells", "Review_Status", "Attested", "Blank", "Ambiguous", "Illegible", "Notes"]
PROMPT_FIELDS = ["Item", "PDF_Page", "Printed_Page", "Row", "OCR_Gloss_Scaffold", "Manual_Gloss", "Review_Status", "Confidence", "Notes"]
CELL_FIELDS = ["Cell_Key", "PDF_Page", "Printed_Page", "Row", "Item", "List_Key", "Source_Label", "Scope", "Language_ID", "Dialect_ID", "OCR_Gloss_Scaffold", "OCR_Form_Scaffold", "Manual_Form", "Cell_Status", "Confidence", "Review_Method", "Reviewer", "Notes"]


def specs_for_page(page: int) -> list[ListSpec]:
    return [spec for spec in LISTS if spec.pages[0] <= page <= spec.pages[1]]


def item_range(page: int, spec: ListSpec) -> range:
    first = (page - spec.pages[0]) * 25 + 1
    return range(first, first + 25)


def fit_rows(
    words: list[dict], expected: range, width: float, height: float
) -> tuple[float, float, str]:
    points: list[tuple[int, float]] = []
    allowed = set(expected)
    for word in words:
        token = re.sub(r"[^0-9]", "", word["text"])
        if token and word["x0"] < width * 0.16:
            value = int(token)
            if value in allowed:
                points.append((value, (word["top"] + word["bottom"]) / 2))
    # The embedded OCR occasionally loses most or all row numbers.  These
    # fallbacks affect only the disposable OCR locator scaffold; they never
    # populate the manual transcription field.  Two anchors are sufficient
    # for a local fit.  With one or zero, use the stable scan geometry so that
    # every conceptual cell still gets a locator row.
    if len(points) == 1:
        slope = height * 0.0273
        value, y = points[0]
        return slope, y - slope * value, "one-anchor-layout-fallback"
    if not points:
        slope = height * 0.0273
        first_y = height * 0.178
        return slope, first_y - slope * expected.start, "fixed-layout-fallback"
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept, "ocr-number-fit"


def row_text(words: list[dict], y: float, tolerance: float, x0: float, x1: float) -> str:
    selected = [
        word for word in words
        if x0 <= (word["x0"] + word["x1"]) / 2 < x1
        and abs((word["top"] + word["bottom"]) / 2 - y) <= tolerance
    ]
    return " ".join(word["text"] for word in sorted(selected, key=lambda word: word["x0"]))


def extract_scaffolds(pdf_path: Path) -> tuple[dict[tuple[int, int, str], dict[str, str]], list[str], list[str]]:
    cells: dict[tuple[int, int, str], dict[str, str]] = {}
    raw_pages: list[str] = []
    fit_notes: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no in range(217, 437):
            page = pdf.pages[page_no - 1]
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            specs = specs_for_page(page_no)
            expected = item_range(page_no, specs[0])
            slope, intercept, fit_method = fit_rows(words, expected, page.width, page.height)
            fit_notes.append(
                f"PDF {page_no}: {fit_method}; slope={slope:.4f}; "
                f"items={expected.start}-{expected.stop - 1}"
            )
            tolerance = min(abs(slope) * 0.48, 18)
            raw_pages.append(f"===== PDF {page_no} / PRINTED {page_no - 5} =====\n{page.extract_text() or ''}\n")
            for item in expected:
                y = slope * item + intercept
                gloss = row_text(words, y, tolerance, page.width * 0.06, page.width * 0.34)
                # Remove a leading item number when OCR retained it in the gloss slice.
                gloss = re.sub(rf"^\s*{item}\s+", "", gloss).strip()
                for spec in specs:
                    if spec.column == "left":
                        form = row_text(words, y, tolerance, page.width * 0.33, page.width * 0.59)
                    else:
                        form = row_text(words, y, tolerance, page.width * 0.59, page.width * 0.93)
                    cells[(page_no, item, spec.key)] = {"gloss": gloss, "form": form}
    return cells, raw_pages, fit_notes


def write_registry() -> None:
    with LISTS_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LIST_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for spec in LISTS:
            writer.writerow(dict(zip(LIST_FIELDS, [spec.key, spec.label, spec.language, spec.dialect, spec.scope, spec.classification, spec.pages[0], spec.pages[1], spec.column, spec.location])))


def initialize(pdf_path: Path) -> None:
    for path in (MANUAL_FILE, PAGES_FILE, PROMPTS_FILE):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite review artifact: {path}")
    scaffold, raw_pages, fit_notes = extract_scaffolds(pdf_path)
    write_registry()
    OCR_TEXT.write_text("\n".join(raw_pages), encoding="utf-8")

    with PAGES_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAGE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for page_no in range(217, 437):
            specs = specs_for_page(page_no)
            items = item_range(page_no, specs[0])
            fit_note = fit_notes[page_no - 217]
            writer.writerow(dict(zip(PAGE_FIELDS, [page_no, page_no - 5, items.start, items.stop - 1, specs[0].key, specs[1].key if len(specs) == 2 else "", 25 * len(specs), "pending", 0, 0, 0, 0, f"OCR layer is locator-only; visual cell review not started; {fit_note}"])))

    # The first pair prints the shared English prompt list.
    tamil = LISTS[0]
    with PROMPTS_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROMPT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for page_no in range(tamil.pages[0], tamil.pages[1] + 1):
            items = item_range(page_no, tamil)
            for row_index, item in enumerate(items, 1):
                gloss = scaffold[(page_no, item, tamil.key)]["gloss"]
                writer.writerow(dict(zip(PROMPT_FIELDS, [item, page_no, page_no - 5, row_index, gloss, "", "pending", "", "must be checked against rendered prompt"])))

    with MANUAL_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CELL_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for spec in LISTS:
            for page_no in range(spec.pages[0], spec.pages[1] + 1):
                for row_index, item in enumerate(item_range(page_no, spec), 1):
                    ocr = scaffold[(page_no, item, spec.key)]
                    key = f"kurumba2012:{spec.key}:i{item:03d}"
                    writer.writerow(dict(zip(CELL_FIELDS, [key, page_no, page_no - 5, row_index, item, spec.key, spec.label, spec.scope, spec.language, spec.dialect, ocr["gloss"], ocr["form"], "", "pending", "", "", "", "OCR is an untrusted locator only; transcribe from rendered scan"])))

    print(f"lists={len(LISTS)} pages=220 prompts=550 cells={len(scaffold)} pending=10450")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if not args.initialize:
        raise SystemExit("Use --initialize to create the one-time pending review ledger")
    initialize(args.pdf)


if __name__ == "__main__":
    main()
