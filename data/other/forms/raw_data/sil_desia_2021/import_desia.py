#!/usr/bin/env python3
"""Install the visually reviewed Desia Appendix B.5 comparative wordlists.

Only ``manual_review.tsv`` is parsed.  The embedded text scaffold is retained
as a locating aid and never feeds installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE_ROOT = HERE.parents[5]
MANUAL = HERE / "manual_review.tsv"
PAGE_REVIEW = HERE / "page_review.tsv"
UNRESOLVED = HERE / "unresolved_readings.tsv"
GLYPH_CORRECTIONS = HERE / "glyph_order_corrections.tsv"
PDF = WORKSPACE_ROOT / "tmp/pdfs/desia-2021-056/source.pdf"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-desia.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-desia-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-desia-manifest.json"

SOURCE_KEY = "behera2021desia"
RECORD_URL = "https://www.sil.org/resources/publications/entry/91960"
ARCHIVE_URL = "https://www.sil.org/resources/archives/91960"
PDF_URL = "https://www.sil.org/system/files/reapdata/54/86/16/54861697763004359352591752899754568865/JLSR2021_056.pdf"
WAYBACK_URL = "https://web.archive.org/web/20240617131527id_/https://www.sil.org/system/files/reapdata/54/86/16/54861697763004359352591752899754568865/JLSR2021_056.pdf"
PDF_SHA256 = "04de0004c1375955c1adbeb8941b187aa4fc88f484ee00e9bc69655813e6690b"

# source label -> dialect id, display label, community label
SITES = {
    "Potenda": ("sil-desia-2007-potenda-rona", "Potenda Rona Desia", "Rona"),
    "Ghumar": ("sil-desia-2007-ghumar-rona", "Ghumar Rona Desia", "Rona"),
    "Sabhapatiguda": ("sil-desia-2007-sabhapatiguda-gaud", "Sabhapatiguda Gaud Desia", "Gaud"),
    "Kantigad": ("sil-desia-2007-kantigad-gaud", "Kantigad Gaud Desia", "Gaud"),
    "Kakalpoda": ("sil-desia-2007-kakalpoda-bod-mali", "Kakalpoda Bod Mali Desia", "Bod Mali"),
    "Konda Maliguda": ("sil-desia-2007-konda-maliguda-bod-mali", "Konda Maliguda Bod Mali Desia", "Bod Mali"),
    "Patta Maliguda": ("sil-desia-2007-patta-maliguda-san-mali", "Patta Maliguda San Mali Desia", "San Mali"),
    "Gumalput": ("sil-desia-2007-gumalput-gadaba", "Gumalput Gadaba Desia", "Gadaba"),
    "Gagnapur": ("sil-desia-2007-gagnapur-poroja", "Gagnapur Poroja Desia", "Poroja"),
    "Dame side": ("sil-desia-2007-dame-side-dom", "Dame side Dom Desia", "Dom"),
    "Burja": ("sil-desia-2007-burja-dom", "Burja Dom Desia", "Dom"),
    "Chhatrabor": ("sil-desia-2007-chhatrabor-harijan", "Chhatrabor Harijan Desia", "Harijan"),
    "Bodgaon": ("sil-desia-2007-bodgaon-dhulia", "Bodgaon Dhulia Desia", "Dhulia"),
    "Gemelput": ("sil-desia-2007-gemelput-mania", "Gemelput Mania Desia", "Mania"),
    "Sindhiguda": ("sil-desia-2007-sindhiguda-bonda", "Sindhiguda Bonda Desia", "Bonda"),
    "Souraguda": ("sil-desia-2007-souraguda-soura", "Souraguda Soura Desia", "Soura"),
    "Aunli": ("sil-desia-2007-aunli-bhotra", "Aunli Bhotra Desia", "Bhotra"),
    "Sourakundi": ("sil-desia-2007-sourakundi-bhotra", "Sourakundi Bhotra Desia", "Bhotra"),
    "Jujhari": ("sil-desia-2007-jujhari-kamar", "Jujhari Kamar Desia", "Kamar"),
}
SITE_ORDER = list(SITES)

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "PDF_Page", "Printed_Page", "Source_Line", "Item", "Gloss",
    "Similarity_Groups", "Site", "Community", "Manual_Form", "Confidence",
    "Review_Method", "Review_Status", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Citation", "Entry_Keys",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewed_rows() -> list[dict[str, str]]:
    with MANUAL.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 4696:
        raise ValueError(f"Expected 4,696 reviewed response lines, got {len(rows)}")
    if any(row["Review_Status"] != "complete" for row in rows):
        raise ValueError("Response-line visual review is incomplete")
    if any(row["Review_Method"] != "manual visual comparison against rendered source image; embedded text used only as scaffold" for row in rows):
        raise ValueError("Review method is not the approved manual visual comparison")
    if any(row["Confidence"] != "high" for row in rows):
        raise ValueError("An unresolved or non-high-confidence response is present")
    if {int(row["PDF_Page"]) for row in rows} != set(range(80, 128)):
        raise ValueError("Expected reviewed Appendix B.5 pages 80-127")
    return rows


def dialect_tag(site: str) -> str:
    dialect, display, _ = SITES[site]
    return f"dialect:AdivasiOriya:{dialect}:{quote(display)}"


def build(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    by_cell: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    glosses: dict[int, str] = {}
    for row in rows:
        item = int(row["Item"])
        if row["Site"] not in SITES:
            raise ValueError(f"Unknown site {row['Site']!r}")
        glosses.setdefault(item, row["Gloss"])
        if glosses[item] != row["Gloss"]:
            raise ValueError(f"Gloss mismatch for item {item}")
        by_cell[(item, row["Site"])].append(row)
    glosses.update({23: "urine", 24: "feces"})

    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for item in range(1, 211):
        if item not in glosses:
            raise ValueError(f"Missing gloss for item {item}")
        for site in SITE_ORDER:
            dialect, display, community = SITES[site]
            source_rows = by_cell.get((item, site), [])
            no_entry_rows = [row for row in source_rows if row["Manual_Form"] == "no entry"]
            attested_rows = [row for row in source_rows if row["Manual_Form"] != "no entry"]
            if not attested_rows:
                if item not in {23, 24} or len(no_entry_rows) != 1:
                    raise ValueError(f"Unaccounted nonblank cell: item {item}, site {site}")
                row = no_entry_rows[0]
                citation = f"{SOURCE_KEY}[Appendix B.5, printed p. {row['Printed_Page']}, item {item}, site {site}]"
                audit.append(dict(zip(AUDIT_FIELDS, [
                    SOURCE_KEY, row["PDF_Page"], row["Printed_Page"], f"p{row['PDF_Page']}:{row['Line']}", str(item), glosses[item],
                    "", site, community, "", "high", "manual visual comparison against rendered source image; embedded text used only as scaffold",
                    "confirmed-blank", "missing", "source explicitly prints no entry for this site/item",
                    "AdivasiOriya", dialect, citation, "",
                ])))
                continue

            # Preserve source order and merge only literally identical readings.
            unique: dict[str, list[dict[str, str]]] = {}
            for row in attested_rows:
                unique.setdefault(unicodedata.normalize("NFC", row["Manual_Form"]), []).append(row)
            for form_index, (form, occurrences) in enumerate(unique.items(), 1):
                groups = list(dict.fromkeys(row["Similarity_Group"] for row in occurrences))
                pages = list(dict.fromkeys(row["PDF_Page"] for row in occurrences))
                printed = list(dict.fromkeys(row["Printed_Page"] for row in occurrences))
                locators = [f"p{row['PDF_Page']}:{row['Line']}" for row in occurrences]
                citation = f"{SOURCE_KEY}[Appendix B.5, printed p. {printed[0]}, item {item}, site {site}]"
                entry_key = f"sildesia2021:i{item:03d}:{site.lower().replace(' ', '-')}:f{form_index}"
                group_note = ",".join(group if group else "[blank]" for group in groups)
                notes = [
                    "manually verified against typeset source image",
                    f"lexical-similarity group(s) {group_note} (non-etymological)",
                ]
                if len(unique) > 1:
                    notes.append(f"source cell variant {form_index}/{len(unique)}")
                if len(occurrences) > 1:
                    notes.append("identical response printed under multiple group labels")
                forms.append([
                    "AdivasiOriya", "", form, glosses[item], "", form, "; ".join(notes),
                    citation, "", "", entry_key, "", "", "", dialect_tag(site),
                ])
                audit.append(dict(zip(AUDIT_FIELDS, [
                    SOURCE_KEY, ";".join(pages), ";".join(printed), ";".join(locators),
                    str(item), glosses[item], ";".join(groups), site, community, form,
                    "high", "manual visual comparison against rendered source image; embedded text used only as scaffold",
                    "complete", "installed", "", "AdivasiOriya", dialect, citation, entry_key,
                ])))
    return forms, audit


def validate(forms: list[list[str]], audit: list[dict[str, str]]) -> None:
    cells = {(int(row["Item"]), row["Site"]) for row in audit}
    if len(cells) != 210 * 19:
        raise ValueError(f"Expected 3,990 accounted cells, got {len(cells)}")
    if len(forms) != 4655:
        raise ValueError(f"Expected 4,655 installed forms, got {len(forms)}")
    counts = Counter(row["Status"] for row in audit)
    if counts != {"installed": 4655, "missing": 38}:
        raise ValueError(f"Unexpected audit status counts: {counts}")
    keys = [row[10] for row in forms]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate Entry_Key")
    if any(len(row) != 15 for row in forms):
        raise ValueError("Installed row does not have 15 columns")
    if any(unicodedata.normalize("NFC", row[2]) != row[2] for row in forms):
        raise ValueError("Installed form is not NFC")
    if any(row[8] or row[9] for row in forms):
        raise ValueError("Similarity groups leaked into etymological fields")


def write(forms: list[list[str]], audit: list[dict[str, str]]) -> None:
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    page_rows = list(csv.DictReader(PAGE_REVIEW.open(encoding="utf-8", newline=""), delimiter="\t"))
    unresolved_rows = list(csv.DictReader(UNRESOLVED.open(encoding="utf-8", newline=""), delimiter="\t"))
    correction_rows = list(csv.DictReader(GLYPH_CORRECTIONS.open(encoding="utf-8", newline=""), delimiter="\t"))
    status_counts = Counter(row["Status"] for row in audit)
    manifest = {
        "source_key": SOURCE_KEY,
        "title": "A Sociolinguistic Survey among Desia-Speaking People Groups in South Orissa, India",
        "author": "Gangadhar Behera",
        "series": "Journal of Language Survey Reports 2021-056",
        "survey_created": 2007,
        "publication_year": 2021,
        "publisher": "SIL International",
        "license": "not stated on the SIL archive/publication record; source PDF is not redistributed in this package",
        "record_url": RECORD_URL,
        "archive_url": ARCHIVE_URL,
        "canonical_pdf_url": PDF_URL,
        "retrieval_url": WAYBACK_URL,
        "pdf_sha256": PDF_SHA256,
        "pdf_bytes": PDF.stat().st_size if PDF.exists() else 0,
        "pdf_pages": 158,
        "wordlist_pdf_pages": [80, 127],
        "wordlist_printed_pages": [71, 118],
        "items": 210,
        "target_sites": 19,
        "comparison_controls": 0,
        "conceptual_cells": 3990,
        "manually_reviewed_response_lines": 4696,
        "manually_reviewed_attested_response_lines": 4658,
        "manually_reviewed_blank_cells": 38,
        "manually_reviewed_conceptual_cells": 3990,
        "installed_forms": len(forms),
        "audit_rows": len(audit),
        "audit_status_counts": dict(sorted(status_counts.items())),
        "unresolved_readings": len(unresolved_rows),
        "text_layer_glyph_order_corrections": len(correction_rows),
        "page_reviews": len(page_rows),
        "page_reviews_complete": all(row["Review_Status"] == "complete" for row in page_rows),
        "review_authority": "manual visual comparison against every rendered typeset response; embedded text is scaffold only",
        "ocr_heavy_addendum": "not applicable: Appendix B.5 is typeset with an embedded Unicode text layer; no handwritten or image-only IPA",
        "similarity_groups": "preserved in Notes/audit only; non-etymological Wordsurv judgments",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pdf", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.verify_pdf:
        if not PDF.exists():
            raise SystemExit(f"Missing canonical PDF: {PDF}")
        actual = sha256(PDF)
        if actual != PDF_SHA256:
            raise SystemExit(f"PDF checksum mismatch: {actual}")
    rows = reviewed_rows()
    forms, audit = build(rows)
    validate(forms, audit)
    if args.install:
        write(forms, audit)
    print(f"reviewed_lines={len(rows)} forms={len(forms)} audit_rows={len(audit)}")


if __name__ == "__main__":
    main()
