#!/usr/bin/env python3
"""Extract Appendix B.3 of SIL ESR 2007-013 without OCR.

The preserved official PDF has a complete text layer. Its phonetic forms use
``SAG-IPASILManuscript``; the legacy bytes appear as U+F000--U+F0FF in text
extraction. The checked-in used-glyph table is a frozen subset of SIL's
official ``SAGIPA2Uni.map`` v1.0 and pins every used byte and occurrence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = Path("/tmp/silesr2007_013-war-jaintia.pdf")
DEFAULT_OUTPUT = HERE / "wordlists.tsv"
MAP_FILE = HERE / "sag_ipa_used.tsv"

PDF_SHA256 = "df28fa5fb8961c2b5029428cace0567d5aa2bb078112903d237517c25521657e"
PAGE_TRANSCRIPT_SHA256 = "43156d186f559eef3fc977b9abfa95f674142728d10171dae2d21f2d9c544750"
PRINTED_PAGES = range(57, 88)
SITE_COUNTS = Counter({
    "A": 288, "B": 292, "C": 283, "D": 287, "E": 293, "F": 288,
    "G": 282, "H": 277, "I": 293, "J": 294, "K": 289, "L": 292,
    "U": 1,
})
FIELDS = [
    "Item", "Gloss", "Similarity_Group", "Raw_Form", "Form", "Site_Codes",
    "Printed_Page", "Response", "Review",
]


def load_map() -> tuple[dict[str, str], Counter[str]]:
    mapping = {}
    expected = Counter()
    with MAP_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            char = chr(int(row["Codepoint"][2:], 16))
            mapping[char] = row["Glyph"].replace("◌", "")
            expected[char] = int(row["Occurrences"])
    return mapping, expected


def source_pages(pdf_path: Path) -> list[tuple[int, str]]:
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != PDF_SHA256:
        raise AssertionError("preserved official War-Jaintia PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 153:
        raise AssertionError(f"official PDF page-count drift: {len(reader.pages)}")
    raw = [reader.pages[index - 1].extract_text() or "" for index in PRINTED_PAGES]
    payload = "\n\f\n".join(raw).encode()
    if hashlib.sha256(payload).hexdigest() != PAGE_TRANSCRIPT_SHA256:
        raise AssertionError("wordlist page-text fingerprint drift")
    raw[0] = raw[0].split("B.3. Wordlists", 1)[1]
    return list(zip(PRINTED_PAGES, raw))


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    mapping, expected_counts = load_map()
    pages = source_pages(pdf_path)
    observed = Counter(char for _, text in pages for char in text if char in mapping)
    if observed != expected_counts:
        raise AssertionError(f"SAG-IPA glyph census drift: {observed}")
    unknown = sorted({
        char for _, text in pages for char in text
        if 0xF000 <= ord(char) <= 0xF0FF and char not in mapping
    })
    if unknown:
        raise AssertionError(
            f"unmapped legacy glyphs: {[f'U+{ord(char):04X}' for char in unknown]}"
        )

    headings: list[int] = []
    rows: list[dict[str, str | int]] = []
    item = None
    gloss = None
    response = 0
    for printed_page, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped == str(printed_page):
                continue
            match = re.fullmatch(r"([0-9A])\s+(.+?)\s+\[([A-LU]+)\]", stripped)
            if match:
                if item is None or gloss is None:
                    raise AssertionError(f"orphan response on p. {printed_page}: {line!r}")
                group, raw_form, codes = match.groups()
                decoded = unicodedata.normalize(
                    "NFC", "".join(mapping.get(char, char) for char in raw_form)
                )
                response += 1
                review = (
                    "official PDF text layer; SAG-IPA bytes converted with SIL's "
                    "SAGIPA2Uni.map v1.0"
                )
                if "U" in codes:
                    review += "; printed undefined site code U retained for audit"
                if group == "A":
                    review += "; printed group A follows groups 1-9 (tenth group)"
                rows.append({
                    "Item": item, "Gloss": gloss, "Similarity_Group": group,
                    "Raw_Form": raw_form, "Form": decoded, "Site_Codes": codes,
                    "Printed_Page": printed_page, "Response": response,
                    "Review": review,
                })
                continue
            match = re.fullmatch(r"(\d{1,3})\s+(.+?)", stripped)
            if match and int(match.group(1)) <= 307 and "[" not in stripped:
                item = int(match.group(1))
                gloss = match.group(2)
                headings.append(item)
                response = 0
                continue
            raise AssertionError(f"unparsed wordlist line p. {printed_page}: {line!r}")

    if headings != list(range(1, 308)):
        raise AssertionError(f"item topology drift: {headings}")
    if len(rows) != 1690:
        raise AssertionError(f"printed response-count drift: {len(rows)}")
    site_counts = Counter(code for row in rows for code in str(row["Site_Codes"]))
    if site_counts != SITE_COUNTS:
        raise AssertionError(f"site-count drift: {site_counts}")
    if any(not row["Form"] for row in rows):
        raise AssertionError("blank decoded form")
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
        f"items=307 printed_responses={len(rows)} expanded="
        f"{sum(len(str(row['Site_Codes'])) for row in rows)} legacy_glyphs="
        f"{sum(int(row['Occurrences']) for row in csv.DictReader(MAP_FILE.open(encoding='utf-8'), delimiter=chr(9)))}"
    )


if __name__ == "__main__":
    main()
