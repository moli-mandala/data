#!/usr/bin/env python3
"""Extract and install Appendix C.3 of SIL ESR 2018-011.

The public 197-page PDF has a usable Unicode Charis/Doulos SIL text layer.  Its
landscape wordlist appendix is laid out as three independent columns, but long
responses can continue into the next column or page.  Extraction therefore
walks page columns in reading order and treats a numbered heading as the only
concept boundary.  This avoids both OCR and the common error of assigning an
overflow line to the concept which happens to occupy its physical column.

Run with ``--extract`` (and ``pdfplumber`` available) to rebuild the checked-in
TSV snapshot from ``tmp/pdfs/bareli/silesr2018_011.pdf``.  With no flag, the
script rebuilds the installed CSV, audit and manifest from that snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
PDF = WORKSPACE / "tmp/pdfs/bareli/silesr2018_011.pdf"
SNAPSHOT = HERE / "wordlist_snapshot.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-bareli-pauri.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-bareli-pauri-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-bareli-pauri-manifest.json"

SOURCE_KEY = "varkey-vunnamatla2018bareli"
PDF_SHA256 = "02128358a61e175ba2a07b2862f6072167a3609cf71264e235ae21284fe2ceea"
PDF_URL = (
    "https://www.sil.org/system/files/reapdata/21/77/99/"
    "21779931578427577212724772957525485566/silesr2018_011.pdf"
)
ARCHIVE_CAPTURE = "20240627030145"

LECTS = [
    "Rathwi Pauri-Amalwadi",
    "Rathwi Pauri-Segwi",
    "Rathwi Bareli-Tharadpura",
    "Rathwi Bareli-Udainagar",
    "Rathwi Bareli-Chiklia",
    "Rathwi-Chenpur",
    "Rathwi-Dongargaon",
    "Bhilali-Bodugam",
    "Bhili-Punyawat",
    "Bhili-Anjhera",
    "Bhilali-Anjhera",
    "Bhilali-Mandwi",
    "Bhilali-Navalpura",
    "Bhilali-Agar",
    "Bhilali-Udaigadh",
    "Bhilali-Kattivada",
    "Parya Bhilali-Bhorwada",
    "Bhili-Piplia",
    "Bhili-Kharod",
    "Bhilali-Aspai",
    "Rathawi-Mankodi",
    "Palya-Choutharya",
    "Palya-Natvada",
    "Bareli Pauri-Shahana",
    "Bareli Pauri-Mandvi",
    "Bareli Pauri-Khadki",
    "Nimadi-Khargone",
    "Nimadi-Awlia",
    "Nimadi-Ashapur",
    "Ahirani-Dhule",
    "Hindi",
    "Gujarati",
    "Marathi",
]
CONTROLS = {"Hindi", "Gujarati", "Marathi"}
LABEL_ALIASES = {
    # Two isolated text-layer/printing inconsistencies; the table identity is
    # unambiguous from its fixed position in the 33-list sequence.
    "Rathwi-ðongargaon": "Rathwi-Dongargaon",
    "Parya Bhilali- Bhorwada": "Parya Bhilali-Bhorwada",
}

LANGUAGE_BY_LECT = {
    **{lect: "RathwiBareli" for lect in LECTS[:7]},
    **{
        lect: "Bhilali"
        for lect in (
            "Bhilali-Bodugam",
            "Bhilali-Anjhera",
            "Bhilali-Mandwi",
            "Bhilali-Navalpura",
            "Bhilali-Agar",
            "Bhilali-Udaigadh",
            "Bhilali-Kattivada",
            "Parya Bhilali-Bhorwada",
            "Bhilali-Aspai",
        )
    },
    **{lect: "Bhili" for lect in ("Bhili-Punyawat", "Bhili-Anjhera", "Bhili-Piplia", "Bhili-Kharod")},
    "Rathawi-Mankodi": "Rathawi",
    "Palya-Choutharya": "PalyaBareli",
    "Palya-Natvada": "PalyaBareli",
    "Bareli Pauri-Shahana": "PauriBareli",
    "Bareli Pauri-Mandvi": "PauriBareli",
    "Bareli Pauri-Khadki": "PauriBareli",
    "Nimadi-Khargone": "Nimadi",
    "Nimadi-Awlia": "Nimadi",
    "Nimadi-Ashapur": "Nimadi",
    "Ahirani-Dhule": "Khandesi",
}

SNAPSHOT_FIELDS = [
    "PDF_Page",
    "Printed_Page",
    "Concept",
    "Gloss",
    "Lect",
    "Category",
    "Response_Index",
    "Form",
    "Notes",
    "Continuation",
    "Raw_Fragments",
    "Source_Status",
]
AUDIT_FIELDS = [
    "Record_Type",
    "Source_Key",
    "PDF_Page",
    "Printed_Page",
    "Concept",
    "Gloss",
    "Lect",
    "Scope",
    "Category",
    "Response_Index",
    "Raw_Form",
    "Form",
    "Notes",
    "Status",
    "Reason",
    "Language_ID",
    "Dialect_ID",
    "Citation",
    "Entry_Key",
]
FORM_FIELDS = [
    "Language_ID",
    "Parameter_ID",
    "Form",
    "Gloss",
    "Native",
    "Phonemic",
    "Notes",
    "Source",
    "Cognateset",
    "Etymology",
    "Entry_Key",
    "Variant_Of_Key",
    "Borrowed_From_Key",
    "Derivation_Parent_Keys",
    "Tags",
]

HEADER_RE = re.compile(r"^(\d{1,3})\.\s*(.+)$")
RESPONSE_RE = re.compile(r"^(\d+(?:,\d+)*)\s+(.+)$")
TRAILING_NOTE_RE = re.compile(r"^(.*?)\s*(?:\[([^\]]+)\]|\(([^()]+)\))$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def dialect_id(lect: str) -> str:
    # Several language lists were elicited in the same locality (most notably
    # Bhili-Anjhera and Bhilali-Anjhera).  Keep the full printed list label in
    # the identifier so dialect IDs and immutable entry keys stay globally
    # unique while the display tag can still show just the locality.
    return f"sil-bareli-2018-{slug(lect)}"


def normalize_form(value: str) -> str:
    # A few non-spacing dental marks are positioned far enough from their
    # base glyph that PDF extraction inserts whitespace.  Preserve real word
    # boundaries while reattaching only Unicode combining marks.
    value = re.sub(r"\s+([\u0300-\u036f])", r"\1", value.strip())
    return unicodedata.normalize("NFC", value)


def separate_editorial_note(form: str, notes: str) -> tuple[str, str]:
    """Move explicit English size/location glosses out of the transcription."""
    match = TRAILING_NOTE_RE.fullmatch(form)
    if match and (match.group(2) or match.group(3)):
        annotation = match.group(2) or match.group(3)
        # A bare unmatched '[' is source transcription evidence, not a note.
        if annotation in {"big", "small", "on ground", "mango tree"}:
            form = match.group(1).rstrip()
            notes = "; ".join(part for part in (notes, f"source annotation: {annotation}") if part)
    if form.endswith("["):
        notes = "; ".join(
            part
            for part in (
                notes,
                "uncertain transcription: source prints a literal unmatched open bracket",
            )
            if part
        )
    return normalize_form(form), notes


def extract_pdf(path: Path) -> list[dict[str, str]]:
    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover - exercised only for snapshot refreshes
        raise SystemExit("--extract requires pdfplumber") from error

    if sha256(path) != PDF_SHA256:
        raise ValueError(f"Unexpected PDF SHA-256 for {path}")

    labels = sorted(LECTS + list(LABEL_ALIASES), key=len, reverse=True)
    records: list[dict[str, object]] = []
    concepts: dict[int, str] = {}
    current_concept: int | None = None
    current_gloss = ""
    current_lect: str | None = None
    current_record: dict[str, object] | None = None
    seen_lects: list[str] = []
    disqualified: set[int] = set()
    concept_page: dict[int, int] = {}

    def finish_concept() -> None:
        nonlocal seen_lects
        if current_concept is None:
            return
        if current_concept in disqualified:
            if seen_lects:
                raise ValueError(f"Disqualified item {current_concept} unexpectedly has responses")
            for lect in LECTS:
                records.append(
                    {
                        "PDF_Page": concept_page[current_concept],
                        "Printed_Page": concept_page[current_concept] - 7,
                        "Concept": current_concept,
                        "Gloss": current_gloss,
                        "Lect": lect,
                        "Category": "",
                        "Form": "DISQUALIFIED",
                        "Notes": "",
                        "Continuation": False,
                        "Raw_Fragments": "70. millet | DISQUALIFIED",
                        "Source_Status": "disqualified",
                    }
                )
        elif seen_lects != LECTS:
            raise ValueError(
                f"Item {current_concept} lect sequence mismatch: "
                f"expected {LECTS!r}, got {seen_lects!r}"
            )
        seen_lects = []

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 197:
            raise ValueError(f"Expected 197 PDF pages, got {len(pdf.pages)}")
        # Physical PDF pages 87--156, printed pp. 80--149.
        for page_index in range(86, 156):
            page = pdf.pages[page_index]
            if (page.width, page.height) != (792.0, 612.0):
                raise ValueError(f"Unexpected appendix page geometry on PDF page {page_index + 1}")
            for column, (x0, x1) in enumerate(((50, 278), (280, 508), (510, 750))):
                text = page.crop((x0, 55, x1, 590)).extract_text(
                    x_tolerance=4, y_tolerance=2
                ) or ""
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    heading = HEADER_RE.fullmatch(line)
                    if heading:
                        finish_concept()
                        current_concept = int(heading.group(1))
                        current_gloss = heading.group(2).strip()
                        concepts[current_concept] = current_gloss
                        concept_page[current_concept] = page_index + 1
                        current_lect = None
                        current_record = None
                        seen_lects = []
                        continue
                    if line == "DISQUALIFIED" and current_concept is not None:
                        disqualified.add(current_concept)
                        continue

                    label = next(
                        (
                            candidate
                            for candidate in labels
                            if line == candidate or line.startswith(candidate + " ")
                        ),
                        None,
                    )
                    if label:
                        lect = LABEL_ALIASES.get(label, label)
                        response = RESPONSE_RE.fullmatch(line[len(label) :].strip())
                        if not response:
                            raise ValueError(f"Unparsed response line on PDF page {page_index + 1}: {line}")
                        category, form = response.groups()
                        current_lect = lect
                        seen_lects.append(lect)
                        current_record = {
                            "PDF_Page": page_index + 1,
                            "Printed_Page": page_index - 6,
                            "Concept": current_concept,
                            "Gloss": current_gloss,
                            "Lect": lect,
                            "Category": category,
                            "Form": form,
                            "Notes": "",
                            "Continuation": False,
                            "Raw_Fragments": line,
                            "Source_Status": "response",
                        }
                        records.append(current_record)
                        continue

                    continuation = RESPONSE_RE.fullmatch(line)
                    if continuation and current_concept is not None and current_lect is not None:
                        category, form = continuation.groups()
                        current_record = {
                            "PDF_Page": page_index + 1,
                            "Printed_Page": page_index - 6,
                            "Concept": current_concept,
                            "Gloss": current_gloss,
                            "Lect": current_lect,
                            "Category": category,
                            "Form": form,
                            "Notes": "",
                            "Continuation": True,
                            "Raw_Fragments": line,
                            "Source_Status": "response",
                        }
                        records.append(current_record)
                        continue

                    if current_record is not None:
                        current_record["Raw_Fragments"] += " | " + line
                        if line.startswith("[") or line.startswith("\\ ["):
                            note = line.lstrip("\\ ").strip("[]")
                            current_record["Notes"] = "; ".join(
                                part for part in (str(current_record["Notes"]), note) if part
                            )
                        else:
                            current_record["Form"] += " " + line
                    elif line != "C.3 Phonetic transcription of wordlists":
                        raise ValueError(
                            f"Orphan table text on PDF page {page_index + 1}, column {column}: {line}"
                        )
    finish_concept()

    if set(concepts) != set(range(1, 211)) or disqualified != {70}:
        raise ValueError(f"Unexpected concept inventory/disqualifications: {set(concepts)}, {disqualified}")
    if len(records) != 7_247:
        raise ValueError(f"Expected 7,247 audited records, got {len(records)}")

    occurrence: Counter[tuple[int, str]] = Counter()
    result: list[dict[str, str]] = []
    for record in records:
        key = (int(record["Concept"]), str(record["Lect"]))
        occurrence[key] += 1
        form, notes = separate_editorial_note(str(record["Form"]), str(record["Notes"]))
        result.append(
            {
                **{field: str(record[field]) for field in SNAPSHOT_FIELDS if field not in {"Response_Index", "Form", "Notes", "Continuation"}},
                "Response_Index": str(occurrence[key]),
                "Form": form,
                "Notes": notes,
                "Continuation": "1" if record["Continuation"] else "0",
            }
        )
    return result


def write_snapshot(records: list[dict[str, str]]) -> None:
    with SNAPSHOT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def load_snapshot() -> list[dict[str, str]]:
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 7_247:
        raise ValueError(f"Expected 7,247 snapshot records, got {len(rows)}")
    return rows


def install(records: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in records:
        lect = row["Lect"]
        control = lect in CONTROLS
        disqualified = row["Source_Status"] == "disqualified"
        no_entry = row["Category"] == "0" or row["Form"] == "NO ENTRY"
        status = "installed"
        reason = ""
        if disqualified:
            status, reason = "excluded", "source marks item 70 DISQUALIFIED for every list"
        elif control:
            status, reason = "excluded", "standard-language comparison control"
        elif no_entry:
            status, reason = "excluded", "source explicitly says NO ENTRY"

        language = LANGUAGE_BY_LECT.get(lect, "")
        site_id = dialect_id(lect) if language else ""
        concept = int(row["Concept"])
        response_index = int(row["Response_Index"])
        citation = (
            f"{SOURCE_KEY}[Appendix C.3, printed p. {row['Printed_Page']}, "
            f"item {concept}, {lect}]"
        )
        entry_key = (
            f"silbareli2018:g{concept:03d}:{site_id}:i{response_index}"
            if status == "installed"
            else ""
        )
        notes = "; ".join(
            part
            for part in (
                f"Appendix C.3 lexical-similarity category {row['Category']}"
                if row["Category"]
                else "",
                row["Notes"],
            )
            if part
        )
        if status == "installed":
            tag = f"dialect:{language}:{site_id}:{quote(lect.rsplit('-', 1)[-1])}"
            forms.append(
                [
                    language,
                    "",
                    row["Form"],
                    row["Gloss"],
                    "",
                    row["Form"],
                    notes,
                    citation,
                    "",
                    "",
                    entry_key,
                    "",
                    "",
                    "",
                    tag,
                ]
            )
        audit.append(
            dict(
                zip(
                    AUDIT_FIELDS,
                    [
                        "wordlist response" if not disqualified else "disqualified concept cell",
                        SOURCE_KEY,
                        row["PDF_Page"],
                        row["Printed_Page"],
                        row["Concept"],
                        row["Gloss"],
                        lect,
                        "standard control" if control else "regional list",
                        row["Category"],
                        row["Response_Index"],
                        row["Raw_Fragments"],
                        row["Form"],
                        row["Notes"],
                        status,
                        reason,
                        language,
                        site_id,
                        citation,
                        entry_key,
                    ],
                )
            )
        )
    if len(forms) != 6_320 or len(audit) != 7_247:
        raise ValueError(f"Unexpected install/audit counts: {len(forms)}, {len(audit)}")
    return forms, audit


def write_outputs(records: list[dict[str, str]]) -> None:
    forms, audit = install(records)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    statuses = Counter(row["Status"] for row in audit)
    exclusion_reasons = Counter(row["Reason"] for row in audit if row["Status"] == "excluded")
    manifest = {
        "source": SOURCE_KEY,
        "title": "A Sociolinguistic Study of Bareli/Pauri and Related Languages",
        "source_pdf_url": PDF_URL,
        "source_pdf_archive_capture": ARCHIVE_CAPTURE,
        "source_pdf_sha256": PDF_SHA256,
        "source_pdf_pages": 197,
        "appendix": "C.3, physical PDF pp. 87-156, printed pp. 80-149",
        "extraction": "Unicode Charis/Doulos SIL text layer; three landscape columns parsed in page reading order; no OCR",
        "snapshot_sha256": sha256(SNAPSHOT),
        "counts": {
            "concepts": 210,
            "regional_lists": 30,
            "standard_controls": 3,
            "snapshot_and_audit_records": len(audit),
            "printed_response_records": 7_214,
            "disqualified_concept_cells": 33,
            "installed_regional_forms": len(forms),
            "excluded_records": statuses["excluded"],
            "explicit_no_entry_records": exclusion_reasons["source explicitly says NO ENTRY"],
            "standard_control_records": exclusion_reasons["standard-language comparison control"],
            "additional_response_lines": sum(row["Continuation"] == "1" for row in records),
        },
        "language_counts": dict(sorted(Counter(row[0] for row in forms).items())),
        "editorial_policy": {
            "controls": "Hindi, Gujarati and Marathi remain in the audit and are not installed",
            "category_codes": "retained in Notes as source lexical-similarity categories; not treated as etymology",
            "literal_open_bracket": "two unmatched source glyphs are preserved and marked uncertain rather than emended",
            "english_annotations": "big, small, on ground and mango tree are moved from Form to Notes",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"installed={len(forms)} controls={exclusion_reasons['standard-language comparison control']} "
        f"no_entry={exclusion_reasons['source explicitly says NO ENTRY']} "
        f"disqualified={exclusion_reasons['source marks item 70 DISQUALIFIED for every list']} "
        f"audit={len(audit)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="rebuild the snapshot from the public PDF")
    args = parser.parse_args()
    if args.extract:
        write_snapshot(extract_pdf(PDF))
    write_outputs(load_snapshot())


if __name__ == "__main__":
    main()
