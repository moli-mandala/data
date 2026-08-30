#!/usr/bin/env python3
"""Extract the Batote wordlist in SIL ESR 2007-017 without OCR.

Appendix B has a complete text layer.  Its phonetic cells use the legacy
``SILManuscriptIPA93`` font and expose the original bytes as U+F000--U+F0FF.
The checked-in used-byte table pins the contextual Unicode result of every
byte used in this appendix against SIL's official ``SIL-IPA93-2001.map``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/dogri/silesr2007_017.pdf"
DEFAULT_OUTPUT = HERE / "wordlist_snapshot.tsv"
MAP_FILE = HERE / "sil_ipa93_used.tsv"

PDF_SHA256 = "04fa21ccf3ca7317ef1a1b3e587b4f1c058b3fb773ea56724d726945a12622c0"
PAGE_TRANSCRIPT_SHA256 = "b56b2d9ed0e9bd3279feb54812d79a34faa228a9be3f3c84932b0d3227169c83"
GLOSS_SEQUENCE_SHA256 = "9c60c1b2a97a84dd5e3c83fd914eb5917144290e47967c21ea4efbfc3036f590"
RAW_FORM_SEQUENCE_SHA256 = "fafcb62ce2d2373e0b151cc638828ee7638ff732124bde1c37ae12edbdd60be4"
PRINTED_PAGES = (26, 27, 28)
BLANK_ITEMS = {11: "breast", 23: "urine", 24: "feces"}
PAGE_COLUMN_COUNTS = {
    (26, 0): 33, (26, 1): 32,
    (27, 0): 44, (27, 1): 44,
    (28, 0): 44, (28, 1): 13,
}
FIELDS = [
    "Item", "Gloss", "Raw_Form", "Form", "PDF_Page", "Printed_Page",
    "Column", "Status", "Review",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map() -> tuple[dict[str, str], Counter[str]]:
    mapping: dict[str, str] = {}
    expected: Counter[str] = Counter()
    with MAP_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            char = chr(int(row["Codepoint"][2:], 16))
            mapping[char] = row["Glyph"].replace("◌", "")
            expected[char] = int(row["Occurrences"])
    return mapping, expected


def visual_lines(page, printed_page: int, column: int) -> list[tuple[float, str, str]]:
    x0, x1 = ((60, 300), (300, 560))[column]
    top0 = 230 if printed_page == 26 else 60
    words = [
        word for word in page.extract_words(extra_attrs=["fontname"])
        if x0 <= float(word["x0"]) < x1 and float(word["top"]) >= top0
        and (
            "Arial" in str(word["fontname"])
            or "SILManuscriptIPA93" in str(word["fontname"])
        )
    ]
    groups: list[tuple[float, list[dict]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not groups or abs(groups[-1][0] - top) > 1.0:
            groups.append((top, [word]))
        else:
            groups[-1][1].append(word)

    lines = []
    for top, group in groups:
        gloss = " ".join(
            str(word["text"]) for word in group if "Arial" in str(word["fontname"])
        ).strip()
        form = " ".join(
            str(word["text"])
            for word in group if "SILManuscriptIPA93" in str(word["fontname"])
        ).strip()
        lines.append((top, gloss, form))
    return lines


def parse_column(page, printed_page: int, column: int) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    pending_gloss = ""
    prefix_forms: list[str] = []
    for _, gloss, form in visual_lines(page, printed_page, column):
        if gloss:
            full_gloss = " ".join(value for value in (pending_gloss, gloss) if value)
            if form:
                records.append((full_gloss, " ".join(prefix_forms + [form])))
                pending_gloss = ""
                prefix_forms = []
            elif full_gloss in BLANK_ITEMS.values():
                if prefix_forms:
                    raise AssertionError("form prefix before a source-blank prompt")
                records.append((full_gloss, ""))
                pending_gloss = ""
            else:
                pending_gloss = full_gloss
        elif form:
            if pending_gloss:
                records.append((pending_gloss, " ".join(prefix_forms + [form])))
                pending_gloss = ""
                prefix_forms = []
            else:
                # The first line of the two-line 'rainbow' response is printed
                # above the prompt's baseline.  Buffer form-only lines forward.
                prefix_forms.append(form)
    if pending_gloss or prefix_forms:
        raise AssertionError(
            f"unresolved column tail p.{printed_page} c.{column}: "
            f"{pending_gloss!r} {prefix_forms!r}"
        )
    expected = PAGE_COLUMN_COUNTS[(printed_page, column)]
    if len(records) != expected:
        raise AssertionError(
            f"column topology drift p.{printed_page} c.{column}: "
            f"{len(records)} != {expected}"
        )
    return records


def decode(raw: str, mapping: dict[str, str]) -> str:
    # Byte 0x22 normally represents dotless i, but SIL's official converter
    # maps it contextually to ordinary i before an above diacritic.  Its sole
    # appendix occurrence is immediately followed by byte 0xE2 (tilde).
    if "\uf022" in raw and "\uf022\uf0e2" not in raw:
        raise AssertionError("unexpected IPA93 byte 0x22 context")
    value = "".join(mapping.get(char, char) for char in raw)
    return unicodedata.normalize("NFC", value)


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    if sha256(pdf_path) != PDF_SHA256:
        raise AssertionError("official Dogri PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 29:
        raise AssertionError(f"official PDF page-count drift: {len(reader.pages)}")
    transcript = "\n\f\n".join(
        reader.pages[page - 1].extract_text() or "" for page in PRINTED_PAGES
    ).encode()
    if hashlib.sha256(transcript).hexdigest() != PAGE_TRANSCRIPT_SHA256:
        raise AssertionError("wordlist page-text fingerprint drift")

    mapping, expected_counts = load_map()
    source_records: list[tuple[int, int, str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for printed_page in PRINTED_PAGES:
            for column in (0, 1):
                for gloss, raw in parse_column(pdf.pages[printed_page - 1], printed_page, column):
                    source_records.append((printed_page, column, gloss, raw))

    if len(source_records) != 210:
        raise AssertionError(f"source item-count drift: {len(source_records)}")
    gloss_hash = hashlib.sha256(
        "\n".join(record[2] for record in source_records).encode()
    ).hexdigest()
    raw_hash = hashlib.sha256(
        "\n".join(record[3] for record in source_records).encode()
    ).hexdigest()
    if gloss_hash != GLOSS_SEQUENCE_SHA256:
        raise AssertionError("printed gloss sequence drift")
    if raw_hash != RAW_FORM_SEQUENCE_SHA256:
        raise AssertionError("raw form sequence drift")

    observed = Counter(
        char for _, _, _, raw in source_records for char in raw if char in mapping
    )
    if observed != expected_counts:
        raise AssertionError(f"IPA93 used-byte census drift: {observed}")
    unknown = sorted({
        char for _, _, _, raw in source_records for char in raw
        if 0xF000 <= ord(char) <= 0xF0FF and char not in mapping
    })
    if unknown:
        raise AssertionError(
            f"unmapped legacy bytes: {[f'U+{ord(char):04X}' for char in unknown]}"
        )

    rows = []
    for item, (printed_page, column, gloss, raw) in enumerate(source_records, 1):
        form = decode(raw, mapping) if raw else ""
        status = "response" if form else "blank"
        if not form and BLANK_ITEMS.get(item) != gloss:
            raise AssertionError(f"unexpected blank item {item}: {gloss}")
        rows.append({
            "Item": item,
            "Gloss": gloss,
            "Raw_Form": raw,
            "Form": form,
            "PDF_Page": printed_page,
            "Printed_Page": printed_page,
            "Column": "left" if column == 0 else "right",
            "Status": status,
            "Review": (
                "official PDF text layer; SIL IPA93 bytes converted with "
                "SIL-IPA93-2001.map v14"
            ),
        })
    if {int(row["Item"]) for row in rows if row["Status"] == "blank"} != set(BLANK_ITEMS):
        raise AssertionError("source blank-item topology drift")
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
    print(
        f"items={len(rows)} responses={sum(row['Status'] == 'response' for row in rows)} "
        f"blanks={sum(row['Status'] == 'blank' for row in rows)} "
        f"legacy_bytes={sum(int(value) for value in load_map()[1].values())}"
    )


if __name__ == "__main__":
    main()
