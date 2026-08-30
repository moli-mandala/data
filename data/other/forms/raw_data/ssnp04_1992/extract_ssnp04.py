#!/usr/bin/env python3
"""Freeze Appendix B of SSNP volume 4 from its positioned legacy-font text.

The publisher PDF contains a complete text layer, but the wordlist forms are
encoded with SILDoulosNP keystrokes rather than Unicode IPA.  This extractor
uses page geometry to retain every one of the 36 list cells under its printed
concept, joins visual continuation lines, and decodes the legacy glyphs using
the repository's audited SSNP decoder plus five volume-specific glyphs checked
against the rendered pages and the report's phonetic chart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/ssnp04/32847_SSNP04.pdf"
DEFAULT_OUTPUT = HERE / "wordlist_snapshot.tsv"
SHARED_DECODER = HERE.parent / "ssnp.py"

PDF_SHA256 = "83e2d833c06ecb4e40bfb0d316061d6b398b743bac299dc870c90c88a4b96f18"
PDF_PAGES = range(98, 165)
PRINTED_PAGE_OFFSET = 18
EXPECTED_PDF_PAGES = 194
EXPECTED_PROMPTS = 200
EXPECTED_CELLS = 7200
EXPECTED_NO_ENTRY = 68
EXPECTED_BLANK = 1
MISSING_ITEMS = (24, 29, 32, 50, 173, 174, 175, 176, 195, 208)
HEADER_SEQUENCE_SHA256 = "12f31fa60e382c777d5d9e382429a5f0e1d6faa9b7c9773148657220220b5550"
ROW_SEQUENCE_SHA256 = "8c42b7bbeb4bf2ef9a60cb71e88c3f3c23396705a02472199e316d72bcf84e4c"

LIST_CODES = (
    "PES", "CHS", "MAR", "SWA", "MAD", "MIN", "BAT", "BAF", "OGI",
    "DIR", "BAJ", "MOH", "NIG", "SHN", "BAR", "MAL", "ZKH", "JAM",
    "TIR", "JAL", "CHE", "PAR", "HAN", "TAL", "KRK", "LAK", "BAN",
    "MIR", "WAA", "QUE", "CHA", "PAS", "KAK", "KHR", "WCI", "ORM",
)

FIELDS = [
    "Item", "Gloss", "List_Code", "Raw_Form", "Form", "PDF_Page",
    "Printed_Page", "Column", "Status", "Continuation_Lines", "Review",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_decoder():
    spec = importlib.util.spec_from_file_location("ssnp_shared_decoder", SHARED_DECODER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.decode_legacy


SHARED_DECODE = load_shared_decoder()


def decode_legacy(text: str) -> str:
    """Decode volume 4's additions to the common SILDoulosNP encoding.

    The unusual glyphs occur only eleven times.  Each was checked at high
    resolution against the printed page: lowercase ``v`` is the same wedge
    vowel as ``V`` here, ``å`` is eth, colon is vowel length, and ``4``/``ƒ``
    are displaced combining underdots.  The shared decoder handles the rest.
    """
    prepared = (
        text.replace("v", "V")
        .replace("å", "ð")
        .replace(":", "ë")
        .replace("ƒ", "4")
        # The common historical decoder flattened these chart glyphs; volume
        # 4's printed phonetic chart distinguishes them explicitly.
        .replace("G", "ɣ")
        .replace("P", "ɸ")
        .replace("B", "β")
        .replace("L", "ɭ")
    )
    return SHARED_DECODE(prepared)


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


def grouped_text(chars: list[dict], threshold: float) -> tuple[str, int]:
    """Rebuild one cell while keeping superscript glyphs on their baseline."""
    lines: list[list] = []
    for char in sorted(chars, key=lambda row: (float(row["top"]), float(row["x0"]))):
        top = float(char["top"])
        match = next((line for line in lines if abs(float(line[0]) - top) <= threshold), None)
        if match is None:
            lines.append([top, [char]])
        else:
            match[1].append(char)
    values = []
    for _, line_chars in sorted(lines, key=lambda row: float(row[0])):
        value = normalize("".join(
            str(char["text"])
            # PDF content order retains zero-width/overstruck diacritics in
            # their logical sequence. Pure x-order would turn ``V3N`` into
            # ``VN3`` because the tilde and following consonant overlap.
            for char in sorted(line_chars, key=lambda row: int(row["_seq"]))
        ))
        if value:
            values.append(value)
    return normalize(" ".join(values)), max(0, len(values) - 1)


def page_rows(page, pdf_page: int) -> tuple[list[tuple[int, str, int]], list[dict]]:
    indexed_chars = [dict(char, _seq=index) for index, char in enumerate(page.chars)]
    words = page.extract_words(extra_attrs=["fontname", "size"])
    headers = sorted(
        [
            word for word in words
            if float(word["top"]) < 205
            and re.fullmatch(r"\d{1,3}\.", str(word["text"]))
            and abs(float(word["size"]) - 9.0) < 0.2
        ],
        key=lambda row: float(row["x0"]),
    )
    if len(headers) not in {2, 3}:
        raise AssertionError(f"unexpected header count on PDF p.{pdf_page}: {len(headers)}")

    locations = [
        word for word in words
        if str(word["text"]) in LIST_CODES
        and float(word["x0"]) < 180
        and 170 < float(word["top"]) < 750
        and "TimesNewRoman" in str(word["fontname"])
    ]
    if tuple(str(word["text"]) for word in locations) != LIST_CODES:
        raise AssertionError(f"location topology drift on PDF p.{pdf_page}")

    # Forms begin at the numbered-header anchors.  A two-point tolerance is
    # needed because the same nominal column sometimes differs by <0.1 pt.
    boundaries = [float(word["x0"]) - 2.0 for word in headers] + [445.0]
    parsed_headers: list[tuple[int, str, int]] = []
    for column, header in enumerate(headers):
        chars = [
            char for char in indexed_chars
            if boundaries[column] <= float(char["x0"]) < boundaries[column + 1]
            and float(header["top"]) - 1 <= float(char["top"]) < float(locations[0]["top"]) - 1
            and "TimesNewRoman" in str(char["fontname"])
        ]
        text, _ = grouped_text(chars, threshold=1.5)
        item = int(str(header["text"])[:-1])
        gloss = re.sub(rf"^{item}\.\s*", "", text)
        parsed_headers.append((item, gloss, column + 1))

    rows = []
    for index, location in enumerate(locations):
        y0 = float(location["top"]) - 1.0
        y1 = (
            float(locations[index + 1]["top"]) - 1.0
            if index + 1 < len(locations) else 760.0
        )
        for column, (item, gloss, printed_column) in enumerate(parsed_headers):
            chars = [
                char for char in indexed_chars
                if boundaries[column] <= float(char["x0"]) < boundaries[column + 1]
                and y0 <= float(char["top"]) < y1
                and "SILDoulosNP" in str(char["fontname"])
            ]
            raw, continuations = grouped_text(chars, threshold=4.0)
            status = "response"
            if raw == "--":
                status = "no_entry"
            elif not raw:
                status = "blank"
            form = decode_legacy(raw) if status == "response" else ""
            rows.append({
                "Item": item,
                "Gloss": gloss,
                "List_Code": str(location["text"]),
                "Raw_Form": raw,
                "Form": form,
                "PDF_Page": pdf_page,
                "Printed_Page": pdf_page - PRINTED_PAGE_OFFSET,
                "Column": printed_column,
                "Status": status,
                "Continuation_Lines": continuations,
                "Review": (
                    "SILDoulosNP positioned text; visual continuation joined"
                    if continuations else "SILDoulosNP positioned text"
                ),
            })
    return parsed_headers, rows


def parse(pdf_path: Path) -> list[dict]:
    if sha256(pdf_path) != PDF_SHA256:
        raise AssertionError("official SSNP volume 4 PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != EXPECTED_PDF_PAGES:
        raise AssertionError(f"PDF page-count drift: {len(reader.pages)}")

    headers: list[tuple[int, str, int, int]] = []
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in PDF_PAGES:
            page = pdf.pages[pdf_page - 1]
            fonts = {str(char["fontname"]) for char in page.chars}
            if not any("SILDoulosNP" in font for font in fonts):
                raise AssertionError(f"legacy wordlist font missing on PDF p.{pdf_page}")
            page_headers, page_records = page_rows(page, pdf_page)
            headers.extend((item, gloss, pdf_page, column) for item, gloss, column in page_headers)
            rows.extend(page_records)

    expected_items = [item for item in range(1, 211) if item not in MISSING_ITEMS]
    if [item for item, _, _, _ in headers] != expected_items:
        raise AssertionError("printed concept topology drift")
    header_hash = hashlib.sha256(
        "\n".join(f"{item}\t{gloss}" for item, gloss, _, _ in headers).encode()
    ).hexdigest()
    if header_hash != HEADER_SEQUENCE_SHA256:
        raise AssertionError("concept/gloss sequence drift")
    if len(headers) != EXPECTED_PROMPTS or len(rows) != EXPECTED_CELLS:
        raise AssertionError(
            f"source topology drift: prompts={len(headers)} cells={len(rows)}"
        )
    if Counter(row["Status"] for row in rows) != Counter({
        "response": EXPECTED_CELLS - EXPECTED_NO_ENTRY - EXPECTED_BLANK,
        "no_entry": EXPECTED_NO_ENTRY,
        "blank": EXPECTED_BLANK,
    }):
        raise AssertionError("response/no-entry topology drift")
    if any(not row["Gloss"] for row in rows):
        raise AssertionError("blank source gloss")
    if any(row["Status"] == "response" and not row["Form"] for row in rows):
        raise AssertionError("decoded response became blank")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in rows):
        raise AssertionError("decoded form is not NFC")
    forbidden = set("VFKQàåƒ†3478ëŒ{}")
    if any(forbidden & set(row["Form"]) for row in rows):
        raise AssertionError("decoded form retains legacy-font keystrokes")
    row_hash = hashlib.sha256(
        "\n".join(
            f"{row['Item']}\t{row['Gloss']}\t{row['List_Code']}\t"
            f"{row['Raw_Form']}\t{row['Form']}"
            for row in rows
        ).encode()
    ).hexdigest()
    if row_hash != ROW_SEQUENCE_SHA256:
        raise AssertionError("response sequence drift")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = parse(args.pdf)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["Status"] for row in rows)
    print(
        f"prompts={EXPECTED_PROMPTS} lists={len(LIST_CODES)} cells={len(rows)} "
        f"responses={counts['response']} no_entry={counts['no_entry']} blank={counts['blank']}"
    )


if __name__ == "__main__":
    main()
