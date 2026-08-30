#!/usr/bin/env python3
"""Freeze Appendix A.4 of SIL ESR 2019-006 from its Unicode text layer.

The appendix prints five concepts per landscape page.  Every concept column
contains the same 27 labelled comparison lists, with occasional additional
responses and ten forms whose similarity-group token is printed on the line
above the transcription.  No OCR or legacy-font conversion is involved.
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
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/lahul/silesr2019_006.pdf"
DEFAULT_OUTPUT = HERE / "wordlist_snapshot.tsv"

PDF_SHA256 = "17f8178505ef88879baecbd5d9fa6dd4f2bb885330722cbac21df70c71e47252"
PAGE_TRANSCRIPT_SHA256 = "b01e036e7812352f903f66ea97ffb80669168c66ab9cb4c6a015476e4cfa8e07"
HEADER_SEQUENCE_SHA256 = "eaea00243b149b79ec83373089e7edb48091c7c71fd1f19e471005c273874fd2"
ROW_SEQUENCE_SHA256 = "5762d4e6e5e3222491dcdb696bedddf24fcf13bd77bfd6d1d4e3f53d720405d8"
PDF_PAGES = range(46, 88)
PAGE_COLUMNS = 5
EXPECTED_ROWS = 6206
EXPECTED_CONTINUATIONS = 10

LECTS = (
    "Hi, Standard", "Pg, Tindi", "Ch, Gushal", "Ch, Nalda",
    "Lo, Gondhla", "Lo, Gawzang", "Pa, Jobrang", "Pa, Thirot",
    "Pa, Udeypur", "Pa, Gushal", "Pa, Mooling", "Pa, Tholang",
    "Pa, Chimrat", "Pa, Salgram", "Ti, Sissu", "Ti, Gondhla",
    "Ga, Keylong", "Ga, Stingri", "Ga, Gawzang", "Bh, Darcha",
    "Bh, Kolong", "Bh, Rarig", "Bh, Tingrat", "Bh, Khoksar",
    "Ld, Leh", "Bh, Spiti", "Tb, Lhasa",
)

EXPECTED_LECT_COUNTS = {
    "Hi, Standard": 263, "Pg, Tindi": 214,
    "Ch, Gushal": 214, "Ch, Nalda": 215,
    "Lo, Gondhla": 220, "Lo, Gawzang": 217,
    "Pa, Jobrang": 246, "Pa, Thirot": 221, "Pa, Udeypur": 244,
    "Pa, Gushal": 226, "Pa, Mooling": 228, "Pa, Tholang": 215,
    "Pa, Chimrat": 224, "Pa, Salgram": 215,
    "Ti, Sissu": 225, "Ti, Gondhla": 216,
    "Ga, Keylong": 225, "Ga, Stingri": 225, "Ga, Gawzang": 221,
    "Bh, Darcha": 270, "Bh, Kolong": 254, "Bh, Rarig": 228,
    "Bh, Tingrat": 282, "Bh, Khoksar": 225,
    "Ld, Leh": 248, "Bh, Spiti": 214, "Tb, Lhasa": 211,
}

LECT_META = {
    "Hi, Standard": ("Hi", "Hindi", "Standard", "comparison"),
    "Pg, Tindi": ("Pg", "Pangi", "Tindi", "prior_list"),
    "Ch, Gushal": ("Ch", "Chinali", "Gushal", "target"),
    "Ch, Nalda": ("Ch", "Chinali", "Nalda", "target"),
    "Lo, Gondhla": ("Lo", "Lohari", "Gondhla", "target"),
    "Lo, Gawzang": ("Lo", "Lohari", "Gawzang", "target"),
    "Pa, Jobrang": ("Pa", "Pattani", "Jobrang", "target"),
    "Pa, Thirot": ("Pa", "Pattani", "Thirot", "target"),
    "Pa, Udeypur": ("Pa", "Pattani", "Udeypur", "target"),
    "Pa, Gushal": ("Pa", "Pattani", "Gushal", "target"),
    "Pa, Mooling": ("Pa", "Pattani", "Mooling", "target"),
    "Pa, Tholang": ("Pa", "Pattani", "Tholang", "target"),
    "Pa, Chimrat": ("Pa", "Pattani", "Chimrat", "target"),
    "Pa, Salgram": ("Pa", "Pattani", "Salgram", "target"),
    "Ti, Sissu": ("Ti", "Tinani", "Sissu", "target"),
    "Ti, Gondhla": ("Ti", "Tinani", "Gondhla", "target"),
    "Ga, Keylong": ("Ga", "Gahri", "Keylong", "target"),
    "Ga, Stingri": ("Ga", "Gahri", "Stingri", "target"),
    "Ga, Gawzang": ("Ga", "Gahri", "Gawzang", "target"),
    "Bh, Darcha": ("Bh", "Bhoti", "Darcha", "target"),
    "Bh, Kolong": ("Bh", "Bhoti", "Kolong", "target"),
    "Bh, Rarig": ("Bh", "Bhoti", "Rarig", "target"),
    "Bh, Tingrat": ("Bh", "Bhoti", "Tingrat", "target"),
    "Bh, Khoksar": ("Bh", "Bhoti", "Khoksar", "target"),
    "Ld, Leh": ("Ld", "Ladakhi", "Leh", "prior_list"),
    "Bh, Spiti": ("Bh", "Bhoti", "Spiti", "prior_list"),
    "Tb, Lhasa": ("Tb", "Tibetan", "Lhasa", "comparison"),
}

FIELDS = [
    "Item", "Gloss", "Lect_Code", "Language_Label", "Site",
    "Similarity_Group", "Response_Index", "Raw_Form", "Form", "PDF_Page",
    "Printed_Page", "Column", "Source_Scope", "Status", "Review",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).replace(" ,", ",")


def column_lines(words: list[dict], column: int) -> list[tuple[float, str]]:
    x0 = 64.0 + 133.3 * column
    x1 = x0 + 133.3
    selected = [
        word for word in words
        if x0 <= float(word["x0"]) < x1 and 70 <= float(word["top"]) < 760
    ]
    groups: list[list] = []
    for word in sorted(selected, key=lambda row: (float(row["top"]), float(row["x0"]))):
        top = float(word["top"])
        if not groups or abs(float(groups[-1][0]) - top) > 1.2:
            groups.append([top, [word]])
        else:
            groups[-1][1].append(word)
    return [
        (
            float(top),
            normalize(" ".join(
                str(word["text"])
                for word in sorted(group, key=lambda row: float(row["x0"]))
            )),
        )
        for top, group in groups
    ]


def parse_column(
    lines: list[tuple[float, str]], pdf_page: int, column: int
) -> tuple[tuple[int, str], list[tuple[str, str, str, bool]]]:
    header_index = next(
        (index for index, (_, text) in enumerate(lines) if re.match(r"^\d+\.\s*", text)),
        None,
    )
    if header_index is None:
        raise AssertionError(f"missing concept header on PDF p.{pdf_page}, column {column + 1}")
    match = re.match(r"^(\d+)\.\s*(.*)$", lines[header_index][1])
    assert match
    item, gloss = int(match.group(1)), match.group(2)
    records: list[tuple[str, str, str, bool]] = []
    current_lect = ""
    pending: tuple[str, str] | None = None

    for _, text in lines[header_index + 1:]:
        printed_lect = next((lect for lect in LECTS if text.startswith(lect)), "")
        if printed_lect:
            current_lect = printed_lect
            rest = text[len(printed_lect):].strip()
        else:
            rest = text

        response = re.match(r"^([0-9a-z])(?:\s+(.*))?$", rest)
        if response:
            if pending is not None:
                raise AssertionError(
                    f"unresolved wrapped form before {text!r}, item {item}, PDF p.{pdf_page}"
                )
            if not current_lect:
                raise AssertionError(
                    f"response without lect label: {text!r}, item {item}, PDF p.{pdf_page}"
                )
            group, form = response.group(1), (response.group(2) or "").strip()
            if form:
                records.append((current_lect, group, normalize(form), False))
            else:
                pending = (current_lect, group)
            continue

        if pending is not None and rest:
            lect, group = pending
            records.append((lect, group, normalize(rest), True))
            pending = None
            continue
        raise AssertionError(
            f"unparsed line {text!r}, item {item}, PDF p.{pdf_page}, column {column + 1}"
        )

    if pending is not None:
        raise AssertionError(f"wrapped form has no continuation, item {item}, PDF p.{pdf_page}")
    return (item, gloss), records


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    if sha256(pdf_path) != PDF_SHA256:
        raise AssertionError("official Lahul PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 185:
        raise AssertionError(f"official PDF page-count drift: {len(reader.pages)}")
    transcript = "\n\f\n".join(
        reader.pages[page - 1].extract_text() or "" for page in PDF_PAGES
    )
    if hashlib.sha256(transcript.encode()).hexdigest() != PAGE_TRANSCRIPT_SHA256:
        raise AssertionError("Appendix A.4 text-layer fingerprint drift")
    if "\ufffd" in transcript or any(0xE000 <= ord(char) <= 0xF8FF for char in transcript):
        raise AssertionError("wordlist text layer unexpectedly contains replacement or PUA glyphs")

    parsed: list[tuple[int, str, str, str, str, int, int, bool]] = []
    headers: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in PDF_PAGES:
            font_words = pdf.pages[pdf_page - 1].extract_words(extra_attrs=["fontname"])
            fonts = {str(word["fontname"]) for word in font_words}
            if not fonts or not all("CharisSIL" in font or "DoulosSIL" in font for font in fonts):
                raise AssertionError(f"unexpected wordlist font set on PDF p.{pdf_page}: {fonts}")
            # Do not retain ``fontname`` in the parse pass: pdfplumber splits a
            # visually continuous form whenever its IPA glyphs switch between
            # the regular and phonetic Charis subsets.
            words = pdf.pages[pdf_page - 1].extract_words()
            for column in range(PAGE_COLUMNS):
                (item, gloss), records = parse_column(column_lines(words, column), pdf_page, column)
                headers.append((item, gloss))
                for lect, group, form, continued in records:
                    parsed.append((item, gloss, lect, group, form, pdf_page, column, continued))

    if [item for item, _ in headers] != list(range(1, 211)):
        raise AssertionError("concept topology drift")
    header_hash = hashlib.sha256(
        "\n".join(f"{item}\t{gloss}" for item, gloss in headers).encode()
    ).hexdigest()
    if header_hash != HEADER_SEQUENCE_SHA256:
        raise AssertionError("concept/gloss sequence drift")
    if len(parsed) != EXPECTED_ROWS:
        raise AssertionError(f"response-count drift: {len(parsed)} != {EXPECTED_ROWS}")
    if Counter(row[2] for row in parsed) != Counter(EXPECTED_LECT_COUNTS):
        raise AssertionError("per-lect response topology drift")
    row_hash = hashlib.sha256(
        "\n".join("\t".join(map(str, row[:5])) for row in parsed).encode()
    ).hexdigest()
    if row_hash != ROW_SEQUENCE_SHA256:
        raise AssertionError("response sequence drift")
    if sum(row[7] for row in parsed) != EXPECTED_CONTINUATIONS:
        raise AssertionError("wrapped-response topology drift")

    indices: Counter[tuple[int, str]] = Counter()
    rows: list[dict[str, str | int]] = []
    for item, gloss, lect, group, form, pdf_page, column, continued in parsed:
        indices[(item, lect)] += 1
        lect_code, language, site, scope = LECT_META[lect]
        status = "no_entry" if form.casefold() in {"no entry", "ɴo entry"} else "response"
        rows.append({
            "Item": item,
            "Gloss": gloss,
            "Lect_Code": lect_code,
            "Language_Label": language,
            "Site": site,
            "Similarity_Group": group,
            "Response_Index": indices[(item, lect)],
            "Raw_Form": form,
            "Form": form,
            "PDF_Page": pdf_page,
            "Printed_Page": pdf_page - 8,
            "Column": column + 1,
            "Source_Scope": scope,
            "Status": status,
            "Review": (
                "Unicode Charis/Doulos SIL text layer; wrapped form joined after visual review"
                if continued else "Unicode Charis/Doulos SIL text layer"
            ),
        })
    if Counter(row["Status"] for row in rows) != Counter({"response": 6139, "no_entry": 67}):
        raise AssertionError("no-entry topology drift")
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
    target = [row for row in rows if row["Source_Scope"] == "target"]
    print(
        f"prompts=210 rows={len(rows)} target_rows={len(target)} "
        f"target_responses={sum(row['Status'] == 'response' for row in target)} "
        f"audit_only={len(rows) - sum(row['Source_Scope'] == 'target' and row['Status'] == 'response' for row in rows)}"
    )


if __name__ == "__main__":
    main()
