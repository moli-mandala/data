#!/usr/bin/env python3
"""Freeze Appendix B.1 of SIL ESR 2010-012 from its Doulos SIL text layer.

The appendix prints four concepts per page in two columns and two vertical
blocks.  Every concept has the same 16 list rows.  The list code is set in
Times and the response is set a few points below it in Doulos SIL, so font,
position, and content-stream order recover the grid without OCR.  Fourteen
printed ``AUS`` labels are source typographical errors in the fixed OSI row;
they are preserved as raw codes and normalized explicitly to OSI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/pahari-pothwari/silesr2010-012.pdf"
DEFAULT_OUTPUT = HERE / "wordlist_snapshot.tsv"

PDF_SHA256 = "e3695a807c4856118303eca74b68b192817ea69251fa8be62abb7b27e4c1ad6f"
PDF_PAGE_COUNT = 262
WORDLIST_PAGES = range(154, 209)
EXPECTED_HEADERS_SHA256 = "115e1e99fec34b21d426e33da32d0c3eebcdebc5923a59b1c09b09e8ae6d4293"
EXPECTED_ROWS_SHA256 = "70d3d99b2f97084d21873c278c0b7998e6615b7aa4a41c1f8a3eb66e17857355"
EXPECTED_SNAPSHOT_SHA256 = "9ef7a0f32c9b2d1d263c1d0fba213d9db67bf1927237915320227c1c6492e7e1"

CODES = (
    "MOS", "GHO", "DEW", "AYU", "KOH", "NIL", "THA", "LOR",
    "OSI", "MUZ", "DUN", "BHA", "ABB", "MAN", "MIR", "GUJ",
)
CODE_ALIASES = {"AUS": "OSI"}
CONTROL_CODES = {"ABB", "MAN"}
EXCLUDED_SIMILARITY_ITEMS = {22, 26, 49, 50, 58, 80, 104, 106, 113, 114, 126}

LECT_META = {
    "MOS": ("Mosyari", "Rawalpindi District", "A"),
    "GHO": ("Ghora Gali", "Rawalpindi District", "A"),
    "DEW": ("Dewal", "Rawalpindi District", "B"),
    "AYU": ("Ayubia", "Abbottabad District", "A"),
    "KOH": ("Kohala", "Rawalpindi District", "A"),
    "NIL": ("Nilabutt", "Bagh, Azad Kashmir", "B"),
    "THA": ("Thandiani", "Abbottabad District", "A"),
    "LOR": ("Lora", "Abbottabad District", "A"),
    "OSI": ("Osia", "Rawalpindi District", "A"),
    "MUZ": ("Muzaffarabad", "Muzaffarabad District", "B"),
    "DUN": ("Dunga Gali", "Abbottabad District", "A"),
    "BHA": ("Bharakoh", "Islamabad District", "B"),
    "ABB": ("Abbottabad", "Abbottabad District", "B"),
    "MAN": ("Mansehra", "Mansehra District", "B"),
    "MIR": ("Mirpur", "Mirpur, Azad Kashmir", "B"),
    "GUJ": ("Gujarkhan", "Rawalpindi District", "B"),
}

SLOTS = (
    ("left", "top", 165.0, 198.0, 199.0, 318.0, 140.0, 395.0),
    ("left", "bottom", 165.0, 198.0, 199.0, 318.0, 395.0, 680.0),
    ("right", "top", 318.0, 352.0, 353.0, 590.0, 140.0, 395.0),
    ("right", "bottom", 318.0, 352.0, 353.0, 590.0, 395.0, 680.0),
)

FIELDS = [
    "Item", "Gloss", "Excluded_From_Similarity", "Raw_Lect_Code",
    "Lect_Code", "Site", "District", "Reliability", "Raw_Form", "Form",
    "PDF_Page", "Printed_Page", "Column", "Block", "Source_Scope",
    "Status", "Review",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


def parse_header(
    words: list[dict], code_words: list[dict], code_x0: float,
    form_x0: float, form_x1: float,
) -> tuple[int, str, bool]:
    top = float(code_words[0]["top"])
    header = sorted(
        (
            word for word in words
            if code_x0 <= float(word["x0"]) < form_x1
            and top - 15.0 <= float(word["top"]) <= top - 5.0
        ),
        key=lambda word: float(word["x0"]),
    )
    raw_number = "".join(str(word["text"]) for word in header if float(word["x0"]) < form_x0)
    gloss = normalize(" ".join(
        str(word["text"]) for word in header if float(word["x0"]) >= form_x0
    ))
    match = re.fullmatch(r"(\d+)\.?(\*)?", raw_number)
    if not match or not gloss:
        raise AssertionError(f"unparsed concept header: {raw_number!r} {gloss!r}")
    return int(match.group(1)), gloss, bool(match.group(2))


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    if sha256(pdf_path) != PDF_SHA256:
        raise AssertionError("official Pahari/Pothwari PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != PDF_PAGE_COUNT:
        raise AssertionError(f"official PDF page-count drift: {len(reader.pages)}")

    rows: list[dict[str, str | int]] = []
    headers: list[tuple[int, str, bool]] = []
    raw_code_counts: Counter[str] = Counter()
    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in WORDLIST_PAGES:
            page = pdf.pages[pdf_page - 1]
            words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False)
            page_slots = 1 if pdf_page == 208 else 4
            for column, block, code_x0, code_x1, form_x0, form_x1, top, bottom in SLOTS[:page_slots]:
                code_words = sorted(
                    (
                        word for word in words
                        if str(word["text"]) in set(CODES) | set(CODE_ALIASES)
                        and code_x0 <= float(word["x0"]) < code_x1
                        and top <= float(word["top"]) < bottom
                    ),
                    key=lambda word: float(word["top"]),
                )
                normalized_codes = [CODE_ALIASES.get(str(word["text"]), str(word["text"])) for word in code_words]
                if normalized_codes != list(CODES):
                    raise AssertionError(
                        f"list-row topology drift on PDF p.{pdf_page} {column}/{block}: "
                        f"{[word['text'] for word in code_words]}"
                    )
                item, gloss, excluded = parse_header(words, code_words, code_x0, form_x0, form_x1)
                headers.append((item, gloss, excluded))

                for word in code_words:
                    raw_code = str(word["text"])
                    code = CODE_ALIASES.get(raw_code, raw_code)
                    raw_code_counts[raw_code] += 1
                    chars = [
                        char for char in page.chars
                        if "DoulosSIL" in str(char["fontname"])
                        and form_x0 <= float(char["x0"]) < form_x1
                        and float(word["top"]) + 1.0 <= float(char["top"]) <= float(word["top"]) + 10.0
                    ]
                    raw_form = "".join(str(char["text"]) for char in chars).strip()
                    form = normalize(raw_form)
                    site, district, reliability = LECT_META[code]
                    review = ""
                    if raw_code != code:
                        review = (
                            "source prints AUS in the fixed OSI inventory row; normalized to OSI "
                            "and retained verbatim in Raw_Lect_Code"
                        )
                    rows.append({
                        "Item": item,
                        "Gloss": gloss,
                        "Excluded_From_Similarity": "Yes" if excluded else "No",
                        "Raw_Lect_Code": raw_code,
                        "Lect_Code": code,
                        "Site": site,
                        "District": district,
                        "Reliability": reliability,
                        "Raw_Form": raw_form,
                        "Form": form,
                        "PDF_Page": pdf_page,
                        "Printed_Page": pdf_page - 6,
                        "Column": column,
                        "Block": block,
                        "Source_Scope": "hindko_control" if code in CONTROL_CODES else "target",
                        "Status": "response" if form else "blank",
                        "Review": review,
                    })

    if [item for item, _, _ in headers] != list(range(1, 218)):
        raise AssertionError("concept topology drift")
    if {item for item, _, excluded in headers if excluded} != EXCLUDED_SIMILARITY_ITEMS:
        raise AssertionError("source-excluded prompt topology drift")
    if len(rows) != 217 * 16:
        raise AssertionError(f"cell-count drift: {len(rows)}")
    if Counter(str(row["Lect_Code"]) for row in rows) != Counter({code: 217 for code in CODES}):
        raise AssertionError("per-list cell topology drift")
    if raw_code_counts["AUS"] != 14 or raw_code_counts["OSI"] != 203:
        raise AssertionError("printed AUS/OSI code topology drift")
    if Counter(str(row["Status"]) for row in rows) != Counter({"response": 3454, "blank": 18}):
        raise AssertionError("response/blank topology drift")
    if sum(row["Source_Scope"] == "target" and row["Status"] == "response" for row in rows) != 3038:
        raise AssertionError("target response topology drift")
    if any("\ufffd" in str(row["Form"]) for row in rows):
        raise AssertionError("replacement character in extracted response")

    header_hash = hashlib.sha256(
        "\n".join(f"{item}\t{gloss}\t{int(excluded)}" for item, gloss, excluded in headers).encode()
    ).hexdigest()
    row_hash = hashlib.sha256(
        "\n".join(
            "\t".join(str(row[field]) for field in ("Item", "Gloss", "Raw_Lect_Code", "Lect_Code", "Form", "PDF_Page"))
            for row in rows
        ).encode()
    ).hexdigest()
    if EXPECTED_HEADERS_SHA256 and header_hash != EXPECTED_HEADERS_SHA256:
        raise AssertionError("concept/gloss sequence drift")
    if EXPECTED_ROWS_SHA256 and row_hash != EXPECTED_ROWS_SHA256:
        raise AssertionError("response sequence drift")
    print(f"header_sha256={header_hash}")
    print(f"row_sha256={row_hash}")
    return rows


def write(pdf_path: Path, output: Path) -> None:
    rows = parse(pdf_path)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    snapshot_hash = sha256(output)
    if EXPECTED_SNAPSHOT_SHA256 and snapshot_hash != EXPECTED_SNAPSHOT_SHA256:
        raise AssertionError("frozen wordlist snapshot drift")
    print(f"snapshot_sha256={snapshot_hash}")
    print(f"rows={len(rows)} output={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write(args.pdf, args.output)


if __name__ == "__main__":
    main()
