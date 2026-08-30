#!/usr/bin/env python3
"""Decode the legacy-SAG IPA wordlists in SIL ESR 2008-013.

The PDF's text layer preserves SAG-IPA bytes as U+F000 plus the original byte.
``sag_ipa_used.tsv`` is the exact 32-symbol subset used on wordlist pages
40--75, transcribed from SIL's official ``SAGIPA2Uni.map`` converter.
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
WORKSPACE = HERE.parents[5]
DEFAULT_PDF = WORKSPACE / "tmp/pdfs/sil-surveys/silesr2008_013.pdf"
DEFAULT_OUTPUT = HERE / "wordlists.tsv"
MAP_FILE = HERE / "sag_ipa_used.tsv"

PDF_SHA256 = "e6b3b6d54c061d03614b27618f0f06d2138f07c47dc1a266d45b0fe16bd75f68"
OFFICIAL_MAP_SHA256 = "a989926e91d4b562df20758cbb613f0177fce33d1c2e9e02195087e94f1f2930"
DISQUALIFIED = {11: "breast", 23: "urine", 24: "feces"}

SOURCES = {
    "Jaunsari-Korwa": ("K", "Korwa", "target"),
    "Jaunsari-Khanaad": ("D", "Khanaad", "target"),
    "Jaunsari-Chapnu": ("C", "Chapnu", "target"),
    "Jaunsari-Bhandroli": ("B", "Bhandroli", "target"),
    "MixedJB-Maindrath": ("M", "Maindrath", "target"),
    "Jr-Glmix-Lakhamandal": ("L", "Lakhamandal", "target"),
    "Jaunsari-Chakratha": ("A", "Chakrata", "target"),
    "Hindi": ("h", "Hindi", "comparison control"),
    "Bangani": ("S", "Bangani", "comparison control"),
    "Jaunpuri": ("J", "Jaunpuri", "comparison control"),
    "Nagpuriya": ("N", "Nagpuriya", "comparison control"),
    "Sirmauri": ("G", "Sirmauri", "comparison control"),
}

FIELDS = [
    "Item", "Gloss", "Source_Code", "Source_Label", "Site_Name", "Role",
    "Response", "Similarity_Group", "Form", "PDF_Page", "Printed_Page",
]


def load_map() -> dict[int, str]:
    mapping = {}
    with MAP_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            value = row["Unicode"].replace("◌", "")
            mapping[int(row["Byte"], 16)] = value
    return mapping


def decode_sag(text: str, mapping: dict[int, str]) -> tuple[str, set[int]]:
    decoded = []
    seen = set()
    for char in text:
        codepoint = ord(char)
        if 0xF000 <= codepoint <= 0xF0FF:
            byte = codepoint - 0xF000
            seen.add(byte)
            if byte not in mapping:
                raise AssertionError(f"unmapped SAG byte 0x{byte:02X}")
            decoded.append(mapping[byte])
        else:
            decoded.append(char)
    return unicodedata.normalize("NFC", "".join(decoded)), seen


def extract(pdf: Path) -> list[dict[str, str | int]]:
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise AssertionError(f"publisher PDF hash drift: {pdf}")
    reader = PdfReader(pdf)
    if len(reader.pages) != 117:
        raise AssertionError(f"publisher PDF page-count drift: {len(reader.pages)}")

    mapping = load_map()
    used_bytes = set()
    rows = []
    headings = []
    item = None
    gloss = None
    current_source = None
    pending_group = None
    responses = Counter()

    source_labels = sorted(SOURCES, key=len, reverse=True)
    for page_index in range(39, 75):
        text, page_bytes = decode_sag(reader.pages[page_index].extract_text() or "", mapping)
        used_bytes.update(page_bytes)
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line in {str(page_index + 1), "Wordlist transcriptions"}:
                continue

            header = re.fullmatch(r"(\d+)\.\s*(.+)", line)
            if header:
                if pending_group is not None:
                    raise AssertionError(f"orphan split response before item {header.group(1)}")
                item = int(header.group(1))
                gloss = header.group(2).strip()
                headings.append(item)
                current_source = None
                continue

            if pending_group is not None:
                if item is None or current_source is None:
                    raise AssertionError(f"orphan split response on PDF page {page_index + 1}")
                response_form = line
                group = pending_group
                pending_group = None
            else:
                source_label = next(
                    (label for label in source_labels if line.startswith(label + " ")),
                    None,
                )
                if source_label is not None:
                    current_source = source_label
                    rest = line[len(source_label):].strip()
                else:
                    rest = line
                response = re.fullmatch(r"(\d+)\s+(.+)", rest)
                if response:
                    group = int(response.group(1))
                    response_form = response.group(2).strip()
                elif source_label is not None and re.fullmatch(r"\d+", rest):
                    pending_group = int(rest)
                    continue
                else:
                    raise AssertionError(
                        f"unparsed wordlist line on PDF page {page_index + 1}: {raw!r}"
                    )

            if item is None or gloss is None or current_source is None:
                raise AssertionError(f"orphan response on PDF page {page_index + 1}: {raw!r}")
            if unicodedata.normalize("NFC", response_form) != response_form:
                raise AssertionError(f"non-NFC form on PDF page {page_index + 1}: {response_form!r}")
            code, site_name, role = SOURCES[current_source]
            responses[item, code] += 1
            rows.append({
                "Item": item,
                "Gloss": gloss,
                "Source_Code": code,
                "Source_Label": current_source,
                "Site_Name": site_name,
                "Role": role,
                "Response": responses[item, code],
                "Similarity_Group": group,
                "Form": response_form,
                "PDF_Page": page_index + 1,
                "Printed_Page": page_index + 1,
            })

    expected_items = set(range(1, 211)) - set(DISQUALIFIED)
    if headings != sorted(expected_items):
        raise AssertionError(f"item topology drift: {headings}")
    if used_bytes != set(mapping):
        raise AssertionError(
            f"SAG-map coverage drift: used={sorted(used_bytes)} mapped={sorted(mapping)}"
        )
    if pending_group is not None:
        raise AssertionError("orphan split response at end of appendix")
    if len(rows) != 2729:
        raise AssertionError(f"response-count drift: {len(rows)}")
    for item_number in expected_items:
        present = {row["Source_Code"] for row in rows if row["Item"] == item_number}
        if present != {details[0] for details in SOURCES.values()}:
            raise AssertionError(f"item {item_number} list coverage drift: {sorted(present)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = extract(args.pdf)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"responses={len(rows)} target={sum(row['Role'] == 'target' for row in rows)} "
        f"controls={sum(row['Role'] != 'target' for row in rows)} output={args.output}"
    )


if __name__ == "__main__":
    main()
