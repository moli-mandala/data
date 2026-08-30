#!/usr/bin/env python3
"""Extract Appendix B.3 of SIL ESR 2008-002 without OCR.

The official PDF has a complete text layer. Its wordlists use the legacy
``SAG-IPASILManuscript`` font, whose symbol bytes appear as U+F000--U+F0FF in
the PDF text extraction. The checked-in used-glyph table is transcribed from
SIL's official ``SAGIPA2Uni.map`` converter and pins every used byte/count.
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
DEFAULT_PDF = Path("/tmp/silesr2008_002.pdf")
DEFAULT_OUTPUT = HERE / "wordlists.tsv"
MAP_FILE = HERE / "sag_ipa_used.tsv"

PDF_SHA256 = "d86fcbb4da2124da0a3ba6a7b48a7c63288fbe45e73d9d7bd03dc83e5e0b4d47"
PAGE_TRANSCRIPT_SHA256 = "e4bb5bca6c847546590df9ca6795e10cf51541d1e72f88d6418c486fabf2c72c"
PRINTED_PAGES = range(45, 69)
SITE_COUNTS = Counter({
    "0": 307, "1": 291, "2": 298, "3": 295, "4": 296,
    "5": 298, "6": 300, "7": 317, "8": 311,
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
        raise AssertionError("official Meitei PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 126:
        raise AssertionError(f"official PDF page-count drift: {len(reader.pages)}")
    raw = [reader.pages[index - 1].extract_text() or "" for index in PRINTED_PAGES]
    payload = "\n\f\n".join(raw).encode()
    if hashlib.sha256(payload).hexdigest() != PAGE_TRANSCRIPT_SHA256:
        raise AssertionError("wordlist page-text fingerprint drift")
    raw[0] = raw[0].split("B.3. Wordlists", 1)[1]
    raw[-1] = raw[-1].split("C. RTT", 1)[0]
    return list(zip(PRINTED_PAGES, raw))


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    mapping, expected_counts = load_map()
    pages = source_pages(pdf_path)
    observed = Counter(
        char for _, text in pages for char in text if char in mapping
    )
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
            match = re.fullmatch(r"(\d+)\s{2,}(.+?)\s+\[([0-8]+)\]", stripped)
            if match:
                if item is None or gloss is None:
                    raise AssertionError(f"orphan response on p. {printed_page}: {line!r}")
                group, raw_form, codes = match.groups()
                decoded = unicodedata.normalize(
                    "NFC", "".join(mapping.get(char, char) for char in raw_form)
                )
                response += 1
                rows.append({
                    "Item": item, "Gloss": gloss, "Similarity_Group": group,
                    "Raw_Form": raw_form, "Form": decoded, "Site_Codes": codes,
                    "Printed_Page": printed_page, "Response": response,
                    "Review": (
                        "official PDF text layer; SAG-IPA bytes converted with SIL's "
                        "SAGIPA2Uni.map v1.0"
                    ),
                })
                continue
            match = re.fullmatch(r"(\d{1,3})\s{2,}(.+?)", stripped)
            if match and int(match.group(1)) <= 307 and "[" not in stripped:
                item = int(match.group(1))
                gloss = match.group(2)
                headings.append(item)
                response = 0
                continue
            raise AssertionError(f"unparsed wordlist line p. {printed_page}: {line!r}")

    if headings != list(range(1, 308)):
        raise AssertionError(f"item topology drift: {headings}")
    if len(rows) != 1219:
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
        f"{sum(len(str(row['Site_Codes'])) for row in rows)} "
        f"legacy_glyphs={sum(int(row['Occurrences']) for row in csv.DictReader(MAP_FILE.open(encoding='utf-8'), delimiter=chr(9)))}"
    )


if __name__ == "__main__":
    main()
