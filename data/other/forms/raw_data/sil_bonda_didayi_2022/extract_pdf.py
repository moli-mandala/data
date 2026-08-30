#!/usr/bin/env python3
"""Deterministically extract Appendix B's born-digital table text.

The output is an extraction scaffold, not a substitute for the checked
``visual_review.tsv`` ledger.  Every cell must be compared with the rendered PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/bonda_didayi/JLSR2022_004.pdf"
DEFAULT_OUTPUT = HERE / "extracted_cells.tsv"
SHA256 = "bb0548b4324224260b9618786dfd3aa40377138d0fbf4ae14c796df82f6190ce"
PDF_FIRST = 21
PDF_LAST = 50

SITES = [
    ("BIA", "Biapada U. Didayi"),
    ("CHI", "Chitrakonda L. Didayi"),
    ("KAL", "Kaluguda U. Didayi"),
    ("ORA", "Orapadar U. Didayi"),
    ("ORI", "Oringi L. Didayi"),
    ("RAS", "Rasabeda L. Bonda"),
    ("KEN", "Kendhuguda L. Bonda"),
    ("KAD", "Kadamguda L. Bonda"),
    ("DUM", "Dumripada U. Bonda"),
    ("GUT", "Tikrapada Gutob"),
    ("PAR", "Kinumun Parenga Parja"),
    ("RON", "Malenga Rona Desiya"),
    ("ODI", "Cuttack Oriya"),
]

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Raw_Response", "Raw_Line", "Extraction_Status",
]

# The embedded text map has one replacement character. This value was read
# directly from the rendered glyph on PDF p. 27 (printed p. 22).
MANUAL_VISUAL_CORRECTIONS = {(50, "CHI"): "6 bɾihumhaiʒã"}


def compact(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


SITE_KEYS = [(code, name, compact(name)) for code, name in SITES]
SITE_ALIASES = {
    # Two typesetting defects in Appendix B's printed locality labels.
    compact("Kaluguda U."): ("KAL", "Kaluguda U. Didayi"),
    compact("Orapadar U. Diday"): ("ORA", "Orapadar U. Didayi"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_lines(page, x0: float, x1: float) -> list[tuple[float, str]]:
    """Return (top, line text) in PDF object order."""
    raw: list[tuple[float, dict]] = []
    for char in page.chars:
        if x0 <= char["x0"] < x1 and 48 <= char["top"] < 735:
            raw.append((float(char["top"]), char))
    # A fallback font used for ŋ is placed about 2.2 points above the rest of
    # its line. Cluster near baselines so these glyphs remain in their forms.
    tops: list[float] = []
    for top, _ in raw:
        if not any(abs(top - known) <= 3.0 for known in tops):
            tops.append(top)
    by_top = {top: [] for top in tops}
    for _, char in raw:
        nearest = min(tops, key=lambda top: abs(float(char["top"]) - top))
        by_top[nearest].append(char)
    lines = []
    for top, chars in sorted(by_top.items()):
        # X-order is necessary for combining marks carried by a fallback font
        # (notably Gutob item 192). U+200A is a PDF positioning shim rather
        # than source whitespace and is therefore omitted.
        full = "".join(
            c["text"] for c in sorted(chars, key=lambda c: float(c["x0"]))
            if c["text"] != "\u200a"
        ).strip()
        # Fallback-font combining marks are sometimes preceded by a positioning
        # space although they visibly attach to the preceding base glyph.
        full = re.sub(r"\s+(?=[\u0300-\u036f])", "", full)
        if full:
            lines.append((top, full))
    return lines


def identify_site(label: str) -> tuple[str, str] | None:
    key = compact(label)
    if not key:
        return None
    for alias, site in SITE_ALIASES.items():
        if key.startswith(alias):
            return site
    for code, name, expected in SITE_KEYS:
        # The PDF contains occasional dropped spaces and one dropped final i in Didayi.
        if key.startswith(expected) or (len(key) >= 10 and expected.startswith(key)):
            return code, name
    return None


def extract(pdf_path: Path) -> list[dict[str, str | int]]:
    if digest(pdf_path) != SHA256:
        raise AssertionError("canonical source PDF checksum drift")
    rows: list[dict[str, str | int]] = []
    prompts: dict[int, str] = {}
    current_item = None
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != 64:
            raise AssertionError(f"expected 64 PDF pages, found {len(pdf.pages)}")
        last_row = None
        for pdf_page in range(PDF_FIRST, PDF_LAST + 1):
            page = pdf.pages[pdf_page - 1]
            for column, bounds in enumerate(((65, 300), (315, 550)), 1):
                for top, full in page_lines(page, *bounds):
                    heading = re.match(r"^(\d+)\.\s*(.*)$", full)
                    if heading:
                        item = int(heading.group(1))
                        gloss = re.sub(r"\s+DISQUALIFIED\s*$", "", heading.group(2)).strip()
                        prompts[item] = gloss
                        current_item = item
                        if "DISQUALIFIED" in full:
                            for code, name in SITES:
                                rows.append({
                                    "Item": item, "Gloss": gloss, "Site_Code": code,
                                    "Site_Name": name, "PDF_Page": pdf_page,
                                    "Printed_Page": pdf_page - 5, "Column": column,
                                    "Raw_Response": "DISQUALIFIED", "Raw_Line": full,
                                    "Extraction_Status": "disqualified",
                                })
                            current_item = None
                            last_row = None
                        continue
                    site = identify_site(full)
                    if not site:
                        # Wrapped alternatives/phrases have no repeated site label.
                        if full and re.match(r"^(?:\d+\s+)?[^.]+", full) and last_row is not None and current_item is not None:
                            separator = " " if str(last_row["Raw_Response"]).endswith(",") else " "
                            last_row["Raw_Response"] = unicodedata.normalize(
                                "NFC", str(last_row["Raw_Response"]) + separator + full
                            )
                            last_row["Raw_Line"] = unicodedata.normalize(
                                "NFC", str(last_row["Raw_Line"]) + " /continued/ " + full
                            )
                        continue
                    if current_item is None:
                        raise AssertionError(
                            f"site row without prompt at PDF {pdf_page}, column {column}, top {top}: {full!r}"
                        )
                    code, name = site
                    start = re.search(r"(?<!\S)(?:\d+\s+|---(?:\s|$)|no entry(?:\s|$))", full)
                    if not start:
                        raise AssertionError(f"cannot split site response at PDF {pdf_page}: {full!r}")
                    response = full[start.start():].strip()
                    printed_label = full[:start.start()].strip()
                    last_row = {
                        "Item": current_item, "Gloss": prompts[current_item],
                        "Site_Code": code, "Site_Name": name, "PDF_Page": pdf_page,
                        "Printed_Page": pdf_page - 5, "Column": column,
                        "Raw_Response": unicodedata.normalize("NFC", re.sub(r"\s+", " ", response)),
                        "Raw_Line": unicodedata.normalize("NFC", f"{printed_label} {response}".strip()),
                        "Extraction_Status": "text-layer",
                    }
                    rows.append(last_row)
    if set(prompts) != set(range(1, 211)):
        raise AssertionError(f"prompt topology drift: {sorted(set(range(1,211)) - set(prompts))}")
    coordinates = [(int(row["Item"]), row["Site_Code"]) for row in rows]
    # The typeset appendix skips the Orapadar row entirely at item 174; account
    # for the conceptual matrix cell explicitly rather than fabricating a form.
    if (174, "ORA") not in coordinates:
        rows.append({
            "Item": 174, "Gloss": prompts[174], "Site_Code": "ORA",
            "Site_Name": "Orapadar U. Didayi", "PDF_Page": 45,
            "Printed_Page": 40, "Column": 1, "Raw_Response": "",
            "Raw_Line": "", "Extraction_Status": "source-omitted",
        })
        coordinates.append((174, "ORA"))
    for row in rows:
        key = (int(row["Item"]), str(row["Site_Code"]))
        if key in MANUAL_VISUAL_CORRECTIONS:
            row["Raw_Response"] = MANUAL_VISUAL_CORRECTIONS[key]
            row["Extraction_Status"] = "manual-visual-correction"
    if len(rows) != 2730 or len(set(coordinates)) != 2730:
        from collections import Counter
        duplicates = [key for key, count in Counter(coordinates).items() if count != 1]
        missing = [(item, code) for item in range(1,211) for code, _ in SITES if (item,code) not in set(coordinates)]
        raise AssertionError(f"cell topology drift rows={len(rows)} duplicates={duplicates[:10]} missing={missing[:10]}")
    rows.sort(key=lambda row: (int(row["Item"]), [code for code, _ in SITES].index(str(row["Site_Code"]))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = extract(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cells to {args.output}")


if __name__ == "__main__":
    main()
