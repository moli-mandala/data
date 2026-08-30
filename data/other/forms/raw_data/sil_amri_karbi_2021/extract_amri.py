#!/usr/bin/env python3
"""Extract the Appendix B.3 text-layer scaffold from JLSR 2021-050.

This is a deterministic extraction aid, not an unchecked lexical authority.
The checked-in ``reviewed_transcription.tsv`` is the cell-by-cell visual review
ledger used by the importer.
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
PDF_SHA256 = "cd121ad102e96b43bf68a1cc5b44f1559c764bc4ae8d71988c6b292a1896ccb1"
PDF_PAGES = 165
APPENDIX_FIRST_PAGE = 37
APPENDIX_LAST_PAGE = 115

SITES = (
    "Holanki, Papumpare AP",
    "S Cherrapunjee",
    "Hajarongpi, E K A",
    "Amguri, Kamrup",
    "PaboiMisamari, Sonitpur",
    "Maina Kharong, Kamrup",
    "RongjariPlasha, Ri-Bhoi",
    "Assamese, Dibrugarh",
    "Amguri, W K A",
    "Sermansingner, E K A",
    "Langhemphi, W K A",
    "Umrinti, W K A",
    "Bankri, W K A",
    "Rongtheang, E K A",
    "Sunajoli, Lakhimpur",
    "Mikirgaon, Nagaon",
    "Sardoka Ingti, E K A",
)

FIELDS = [
    "PDF_Page", "Printed_Page", "Page_Line", "Item", "Gloss", "Site",
    "Similarity_Group", "Extracted_Form", "Record_Type", "Extraction_Note",
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
    item: int | None = None
    gloss = ""
    last_site = ""
    pending: dict[str, str] | None = None

    def add(page: int, line: int, site: str, group: str, form: str,
            record_type: str = "response", note: str = "") -> None:
        assert item is not None and gloss and site
        records.append({
            "PDF_Page": str(page),
            "Printed_Page": str(page - 10),
            "Page_Line": str(line),
            "Item": str(item),
            "Gloss": gloss,
            "Site": site,
            "Similarity_Group": group,
            "Extracted_Form": unicodedata.normalize("NFC", form.strip()),
            "Record_Type": record_type,
            "Extraction_Note": note,
        })

    for page in range(APPENDIX_FIRST_PAGE, APPENDIX_LAST_PAGE + 1):
        lines = (reader.pages[page - 1].extract_text() or "").splitlines()
        for line_number, raw in enumerate(lines, 1):
            text = raw.strip()
            if not text or re.fullmatch(r"\d{2,3}", text) or text == "B.3 Wordlist transcription":
                continue

            heading = re.fullmatch(r"(\d{1,3})\.\s+(.+?)\s*", text)
            if heading:
                if pending:
                    raise ValueError(f"unresolved wrapped record before PDF p.{page}: {pending}")
                item, gloss = int(heading.group(1)), heading.group(2)
                last_site = ""
                continue

            matched_site = next((site for site in SITES if text.startswith(site + " ")), "")
            if matched_site:
                if pending:
                    raise ValueError(f"unresolved wrapped record before PDF p.{page}: {pending}")
                last_site = matched_site
                remainder = text[len(matched_site):].strip()
                if re.fullmatch(r"0\.?\s*[Nn]o entry", remainder):
                    add(page, line_number, matched_site, "0", "", "blank", remainder)
                    continue
                response = re.fullmatch(r"(\d+)\s*(.*)", remainder)
                if not response:
                    raise ValueError(f"unparsed site line PDF p.{page}:{line_number}: {text!r}")
                group, form = response.groups()
                if form:
                    add(page, line_number, matched_site, group, form)
                else:
                    pending = {
                        "page": str(page), "line": str(line_number), "site": matched_site,
                        "group": group,
                    }
                continue

            continuation = re.fullmatch(r"(\d+)\s+(.+)", text)
            if continuation and last_site:
                if pending:
                    raise ValueError(f"unexpected numbered continuation after pending record: {pending}")
                add(page, line_number, last_site, continuation.group(1), continuation.group(2))
                continue

            if pending:
                add(
                    int(pending["page"]), int(pending["line"]), pending["site"],
                    pending["group"], text, note=f"form continues on PDF text line {line_number}",
                )
                pending = None
                continue

            # Three long Assamese infinitives have their final word wrapped to
            # an unnumbered text line.  Join it to the immediately preceding
            # response; the page render is authoritative for visual review.
            if (
                records and records[-1]["Item"] == str(item)
                and records[-1]["Site"] == "Assamese, Dibrugarh"
                and records[-1]["PDF_Page"] == str(page)
            ):
                records[-1]["Extracted_Form"] += " " + unicodedata.normalize("NFC", text)
                records[-1]["Extraction_Note"] = f"form continues on PDF text line {line_number}"
                continue
            raise ValueError(f"unparsed PDF p.{page}:{line_number}: {text!r}")

    if pending:
        raise ValueError(f"unresolved final wrapped record: {pending}")
    if {int(record["Item"]) for record in records} != set(range(1, 308)):
        raise ValueError("Appendix B.3 does not contain the expected items 1-307")
    cells = {(record["Item"], record["Site"]) for record in records}
    if len(cells) != 307 * 17:
        raise ValueError(f"expected 5,219 conceptual cells, found {len(cells)}")
    if len(records) != 5_966:
        raise ValueError(f"expected 5,966 printed records, found {len(records)}")
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
            "Verified_Form", "Review_Status", "Confidence", "Review_Note",
            "Extraction_Note",
        ]
        with args.review_template.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=review_fields, delimiter="\t")
            writer.writeheader()
            for record in records:
                reviewed = dict(record)
                reviewed.update({
                    "Verified_Form": record["Extracted_Form"],
                    "Review_Status": "pending",
                    "Confidence": "",
                    "Review_Note": "",
                })
                writer.writerow(reviewed)
    print(f"records={len(records)} cells={len({(r['Item'], r['Site']) for r in records})} output={args.output}")


if __name__ == "__main__":
    main()
