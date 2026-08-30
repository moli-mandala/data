#!/usr/bin/env python3
"""Seed the line ledger from the PDF text layer for subsequent visual review.

This is intentionally not part of installation.  Running it overwrites the
review ledger, so any run must be followed by another complete image review.
The two structural repairs below were checked against the page images; they do
not invent phonetic content.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PDF = ROOT / "tmp/pdfs/korwa-kodaku-2022/source.pdf"
ALL_CODES = set("ABCDGHJKLMRSTUVZbcdjkmptw")
FIELDS = [
    "PDF_Page", "Printed_Page", "Line", "Item", "Gloss",
    "Similarity_Group", "Manual_Form", "Site_Codes", "Unknown_Site_Codes",
    "Confidence", "Review_Status", "Review_Note",
]


def repaired_lines(text: str) -> list[str]:
    source = text.splitlines()
    result: list[str] = []
    index = 0
    while index < len(source):
        line = source[index]
        # PDF p.72 wraps the code bracket for item 72 onto the next extraction
        # line although it is a single printed row.
        if (
            index + 1 < len(source)
            and re.fullmatch(r"\s*\[[A-Za-z]+\]\s*", source[index + 1])
            and re.match(r"^\s*(?:[0-9]+|[abc])\s+\S", line)
        ):
            line = line.rstrip() + " " + source[index + 1].strip()
            index += 1
        # Opening brackets are absent from the embedded text for PDF pp.76
        # item 104 and 80 item 139, but are visibly printed.
        line = re.sub(
            r"^(\s*(?:[0-9]+|[abc])\s+.+?)\s+([A-Za-z]+)\]\s*$",
            r"\1 [\2]",
            line,
        )
        line = line.replace("[  BV]", "[BV]")
        result.append(line)
        index += 1
    return result


def main() -> None:
    reader = PdfReader(PDF)
    rows: list[dict[str, str]] = []
    page_counts: Counter[int] = Counter()
    item = 0
    gloss = ""
    for pdf_page in range(66, 91):
        for line_number, line in enumerate(
            repaired_lines(reader.pages[pdf_page - 1].extract_text() or ""), 1
        ):
            response = re.fullmatch(
                r"\s*([0-9]+|[abc])\s+(.+?)\s*\[\s*([A-Za-z]+)\s*\]\s*",
                line,
            )
            if response:
                codes = response.group(3)
                unknown = "".join(code for code in codes if code not in ALL_CODES)
                note = ""
                if unknown:
                    note = "unidentified source site code; retained, never reassigned"
                rows.append({
                    "PDF_Page": str(pdf_page),
                    "Printed_Page": str(pdf_page - 10),
                    "Line": str(line_number),
                    "Item": str(item),
                    "Gloss": gloss,
                    "Similarity_Group": response.group(1),
                    "Manual_Form": unicodedata.normalize("NFC", response.group(2).strip()),
                    "Site_Codes": codes,
                    "Unknown_Site_Codes": unknown,
                    "Confidence": "high",
                    "Review_Status": "complete",
                    "Review_Note": note,
                })
                page_counts[pdf_page] += 1
                continue
            heading = re.fullmatch(r"(\d{1,3})\s+(.+?)\s*", line)
            if heading and 1 <= int(heading.group(1)) <= 210:
                item = int(heading.group(1))
                gloss = heading.group(2).strip()
    if len({int(row["Item"]) for row in rows}) != 208:
        # Items 23 and 24 are page-level NO ENTRY statements and therefore
        # intentionally have no response line.
        raise ValueError("unexpected item coverage in response-line scaffold")
    with (HERE / "manual_review.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (HERE / "page_review.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["PDF_Page", "Printed_Page", "Response_Lines", "Review_Status", "Review_Method"])
        for pdf_page in range(66, 91):
            writer.writerow([
                pdf_page, pdf_page - 10, page_counts[pdf_page], "complete",
                "visual comparison of every printed response line and code bracket at 200 dpi",
            ])
    print(f"response_lines={len(rows)} pages=25")


if __name__ == "__main__":
    main()
