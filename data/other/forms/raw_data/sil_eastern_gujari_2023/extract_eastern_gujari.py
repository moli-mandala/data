#!/usr/bin/env python3
"""Extract the Appendix B Unicode text-layer scaffold from JLSR 2023-002.

The text layer is only a locating scaffold.  The importer consumes the
separate reviewed ledger, whose 3,150 cells were compared with page renders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF_SHA256 = "41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e"
PDF_PAGES = 121
FIRST_PAGE = 42
LAST_PAGE = 76

LISTS = (
    "Urdu", "Chitral/Pak", "Settlet Swat/Pak", "Gilgit/Pak", "Kaghan/Pak",
    "North. Azad/Pak", "Centr. Azad/Pak", "Udhampur/J&K", "Jammu/J&K",
    "Chamba/H.P.", "Rampur/H.P.", "Nalagarh/H.P.", "Dehradun/U.P.",
    "Kotdwara/U.P.", "Haldwani/U.P.",
)

FIELDS = [
    "PDF_Page", "Printed_Page", "Page_Line", "Item", "Gloss", "List",
    "Source_Cell", "Record_Type", "Extraction_Note",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pdf(path: Path) -> list[dict[str, str]]:
    if sha256(path) != PDF_SHA256:
        raise ValueError(f"unexpected PDF SHA-256 for {path}")
    reader = PdfReader(path)
    if len(reader.pages) != PDF_PAGES:
        raise ValueError(f"expected {PDF_PAGES} pages, found {len(reader.pages)}")

    records: list[dict[str, str]] = []
    expected_item = 1
    ignored = set(LISTS) | {
        "B.3 Reference wordlist", "Country: Pakistan", "Language: Urdu",
    }
    for page in range(FIRST_PAGE, LAST_PAGE + 1):
        lines = [(n, raw.strip()) for n, raw in enumerate(
            (reader.pages[page - 1].extract_text() or "").splitlines(), 1
        )]
        lines = [(n, text) for n, text in lines if text]
        cursor = 0
        page_first = (page - FIRST_PAGE) * 6 + 1
        if expected_item != page_first:
            raise ValueError(f"item/page drift before PDF p. {page}")
        for item in range(page_first, page_first + 6):
            heading_re = re.compile(rf"^{item}\s+(.+)$")
            while cursor < len(lines):
                line_number, text = lines[cursor]
                cursor += 1
                matched = heading_re.fullmatch(text)
                if matched:
                    gloss = matched.group(1).strip()
                    break
            else:
                raise ValueError(f"missing item {item} on PDF p. {page}")

            responses: list[tuple[int, str, str]] = []
            while cursor < len(lines) and len(responses) < len(LISTS):
                line_number, text = lines[cursor]
                cursor += 1
                if text in ignored or re.fullmatch(r"\d{2,3}", text):
                    continue
                if text.startswith("/"):
                    if not responses:
                        raise ValueError(f"orphan continuation PDF p. {page}:{line_number}")
                    old_line, old_text, old_note = responses[-1]
                    responses[-1] = (
                        old_line, old_text + " " + text,
                        f"source cell wraps onto PDF text line {line_number}",
                    )
                    continue
                responses.append((line_number, text, ""))

            if len(responses) != len(LISTS):
                raise ValueError(f"expected 15 responses for item {item}, found {len(responses)}")
            for list_name, (line_number, source_cell, note) in zip(LISTS, responses):
                source_cell = unicodedata.normalize("NFC", source_cell)
                record_type = "blank" if re.search(r"\b0\s+no entry\b", source_cell, re.I) else "response"
                records.append({
                    "PDF_Page": str(page), "Printed_Page": str(page - 8),
                    "Page_Line": str(line_number), "Item": str(item),
                    "Gloss": gloss, "List": list_name, "Source_Cell": source_cell,
                    "Record_Type": record_type, "Extraction_Note": note,
                })
            expected_item += 1

    if expected_item != 211 or len(records) != 3_150:
        raise ValueError(f"expected 3,150 cells/items 1-210, found {len(records)}")
    if len({(r['Item'], r['List']) for r in records}) != 3_150:
        raise ValueError("duplicate or missing conceptual cells")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "extraction_scaffold.tsv")
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args()
    records = parse_pdf(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    if args.review_template:
        review_fields = FIELDS[:-1] + [
            "Verified_Cell", "Review_Status", "Confidence", "Review_Note",
            "Extraction_Note",
        ]
        with args.review_template.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=review_fields, delimiter="\t")
            writer.writeheader()
            for record in records:
                reviewed = dict(record)
                reviewed.update({
                    "Verified_Cell": record["Source_Cell"], "Review_Status": "pending",
                    "Confidence": "", "Review_Note": "",
                })
                writer.writerow(reviewed)
    print(f"cells={len(records)} prompts=210 lists=15 output={args.output}")


if __name__ == "__main__":
    main()
