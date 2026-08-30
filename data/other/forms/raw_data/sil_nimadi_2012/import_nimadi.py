#!/usr/bin/env python3
"""Extract and install Appendix A of SIL ESR 2012-002.

The archived official 176-page PDF has a usable Unicode Doulos SIL text layer.
Appendix A prints 18 parallel lists in three portrait columns (two wider columns
for the long predicate prompts on physical pages 95--99).  The checked-in TSV
is the reproducible extraction snapshot; ordinary runs rebuild the installed
CSV, complete audit, and manifest without requiring the PDF or pdfplumber.

Use ``--extract`` to refresh the snapshot from
``tmp/pdfs/nimadi/silesr2012_002.pdf`` after checking its pinned SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
PDF = WORKSPACE / "tmp/pdfs/nimadi/silesr2012_002.pdf"
SNAPSHOT = HERE / "wordlist_snapshot.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-nimadi.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-nimadi-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-nimadi-manifest.json"

SOURCE_KEY = "vunnamatla-john-samuvel2012nimadi"
PDF_SHA256 = "1a7e8daaeb2b967e2f9490292689e33a188caf47dc262c942a47136bb270d0d8"
PDF_URL = "http://www-01.sil.org/silesr/2012/silesr2012-002.pdf"
ARCHIVE_URL = (
    "https://web.archive.org/web/20170810011221id_/"
    "http://www-01.sil.org/silesr/2012/silesr2012-002.pdf"
)

TARGET_LECTS = [
    "N-Son-Bal", "N-Son-Pat", "N-Bal-Br", "N-Jaj-OBC", "N-Bhi-Bhi",
    "N-Dhar-Bhi", "N-Khj-Bhi", "N-Mah-Bhi", "N-Rup-Br", "N-Khr-Gen",
    "N-Awl-Bal", "N-Sir-OBC", "N-Kup-Dar",
]
CONTROL_LECTS = ["Par Bhi", "Malvi", "Hindi", "Gujarati", "Marathi"]
LECTS = TARGET_LECTS + CONTROL_LECTS
LABEL_ALIASES = {"N-Son-Paty": "N-Son-Pat", "M arathi": "Marathi"}
OMITTED_PROMPTS = {11: "breast", 23: "urine", 24: "feces", 70: "millet"}

LECT_METADATA = {
    "N-Son-Bal": ("Sonipura-Balai", "Sonipura, Khargone tahsil and district, Madhya Pradesh; Balai community; source wordlist code e"),
    "N-Son-Pat": ("Sonipura-Patidar", "Sonipura, Khargone tahsil and district, Madhya Pradesh; Patidar community; source wordlist code f"),
    "N-Bal-Br": ("Balkhad-Brahmin", "Balkhad, Kasarawad tahsil, Khargone district, Madhya Pradesh; Brahmin community; source wordlist code b"),
    "N-Jaj-OBC": ("Jajamkhedi-OBC", "Jajamkhedi, Manawar tahsil, Dhar district, Madhya Pradesh; OBC community; source wordlist code o"),
    "N-Bhi-Bhi": ("Bhilkheda-Bhilala", "Bhilkheda, Barwani tahsil and district, Madhya Pradesh; Bhilala community; source wordlist code l"),
    "N-Dhar-Bhi": ("Awlia-Dhar-Bhilala", "Awlia, Nalcha tahsil, Dhar district, Madhya Pradesh; Bhilala community; source wordlist code n"),
    "N-Khj-Bhi": ("Khajuri-Bhilala", "Khajuri, Thikri tahsil, Rajpur district, Madhya Pradesh; Bhilala community; source wordlist code j"),
    "N-Mah-Bhi": ("Maheshwar-Bhilala", "Maheshwar, Maheshwar tahsil, Khargone district, Madhya Pradesh; Bhilala community; source wordlist code c"),
    "N-Rup-Br": ("Rupkheda-Brahmin", "Rupkheda, Barwa tahsil, Khargone district, Madhya Pradesh; Brahmin community; source wordlist code r"),
    "N-Khr-Gen": ("Khargone-General", "Khargone town, Khargone tahsil and district, Madhya Pradesh; general sample; source wordlist code k"),
    "N-Awl-Bal": ("Awlia-Khandwa-Balai", "Awlia, Khandwa tahsil and district, Madhya Pradesh; Balai community; source wordlist code a"),
    "N-Sir-OBC": ("Sirpur-Melgav-OBC", "Sirpur and Melgav, Khalwa tahsil, Harsood subdivision, Khandwa district, Madhya Pradesh; OBC community; source wordlist code s"),
    "N-Kup-Dar": ("Kupdol-Badgav-Darbar", "Kupdol and Badgav, Khargone tahsil and district, Madhya Pradesh; Darbar community; source wordlist code d"),
}

SNAPSHOT_FIELDS = [
    "PDF_Page", "Printed_Page", "Concept", "Gloss", "Lect", "Category",
    "Response_Index", "Form", "Notes", "Raw_Fragments", "Source_Status",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Concept",
    "Gloss", "Lect", "Scope", "Category", "Response_Index", "Raw_Form",
    "Form", "Notes", "Status", "Reason", "Language_ID", "Dialect_ID",
    "Citation", "Entry_Key",
]
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]

HEADER_RE = re.compile(r"^(\d{1,3})\.\s*(.*)$")
CATEGORY_RE = re.compile(r"^(\d+)\s+(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def dialect_id(lect: str) -> str:
    return f"sil-nimadi-2012-{slug(LECT_METADATA[lect][0])}"


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+([\u0300-\u036f])", r"\1", value.strip())
    return unicodedata.normalize("NFC", value)


def reconstruct_line(words: list[dict[str, object]]) -> str:
    """Reconstruct one PDF line, retaining true spaces but rejoining glyph runs."""
    parts: list[str] = []
    previous_x1: float | None = None
    for word in sorted(words, key=lambda item: float(item["x0"])):
        x0, x1, token = float(word["x0"]), float(word["x1"]), str(word["text"])
        if previous_x1 is not None and x0 - previous_x1 > 2.2:
            parts.append(" ")
        parts.append(token)
        previous_x1 = x1
    return normalize_text("".join(parts))


def column_lines(page, bounds: list[tuple[float, float]]) -> list[list[str]]:
    words = [
        word for word in page.extract_words(x_tolerance=2, y_tolerance=2)
        if 65 <= float(word["top"]) <= 735
    ]
    result: list[list[str]] = []
    for x0, x1 in bounds:
        selected = [word for word in words if x0 <= float(word["x0"]) < x1]
        groups: list[tuple[float, list[dict[str, object]]]] = []
        for word in sorted(selected, key=lambda item: (float(item["top"]), float(item["x0"]))):
            top = float(word["top"])
            if not groups or abs(groups[-1][0] - top) > 1.8:
                groups.append((top, []))
            groups[-1][1].append(word)
        result.append([reconstruct_line(group) for _, group in groups])
    return result


def extract_pdf(path: Path) -> list[dict[str, str]]:
    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover
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

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 176:
            raise ValueError(f"Expected 176 PDF pages, got {len(pdf.pages)}")
        for physical_page in range(58, 102):
            page = pdf.pages[physical_page - 1]
            if (round(page.width, 1), round(page.height, 1)) != (612.0, 792.0):
                raise ValueError(f"Unexpected page geometry on physical page {physical_page}")
            shift = -6.7 if physical_page % 2 else 0.0
            bounds = (
                [(35 + shift, 300 + shift), (300 + shift, 612)]
                if 95 <= physical_page <= 99
                else [(35 + shift, 232 + shift), (232 + shift, 385 + shift), (385 + shift, 612)]
            )
            for lines in column_lines(page, bounds):
                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    heading = HEADER_RE.fullmatch(line)
                    if heading:
                        current_concept = int(heading.group(1))
                        current_gloss = heading.group(2).strip()
                        concepts[current_concept] = current_gloss
                        current_lect = None
                        current_record = None
                        continue

                    label = next(
                        (candidate for candidate in labels if line == candidate or line.startswith(candidate + " ")),
                        None,
                    )
                    if label:
                        lect = LABEL_ALIASES.get(label, label)
                        remainder = line[len(label):].strip()
                        # The source's ToUnicode map renders category "1" as lowercase l
                        # for N-Rup-Br item 40. The rendered page shows a blank category-1
                        # primary followed by the printed category-2 alternate.
                        if current_concept == 40 and lect == "N-Rup-Br" and remainder == "l":
                            category, form = "1", ""
                            notes = "image-verified blank primary; PDF text layer maps printed category 1 to lowercase l"
                        elif current_concept == 13 and lect == "N-Son-Bal" and remainder == "1ct̪":
                            category, form = "1", "ct̪"
                            notes = "diplomatic image-verified transcription; category digit was fused to the source glyph run"
                        else:
                            response = CATEGORY_RE.fullmatch(remainder)
                            if not response:
                                raise ValueError(
                                    f"Unparsed response on PDF page {physical_page}: {line!r}"
                                )
                            category, form = response.groups()
                            notes = ""
                        current_lect = lect
                        current_record = {
                            "PDF_Page": physical_page,
                            "Printed_Page": physical_page - 1,
                            "Concept": current_concept,
                            "Gloss": current_gloss,
                            "Lect": lect,
                            "Category": category,
                            "Form": form,
                            "Notes": notes,
                            "Raw_Fragments": line,
                            "Source_Status": "blank" if not form else "response",
                        }
                        records.append(current_record)
                        continue

                    alternate = CATEGORY_RE.fullmatch(line)
                    if alternate and current_concept is not None and current_lect is not None:
                        category, form = alternate.groups()
                        current_record = {
                            "PDF_Page": physical_page,
                            "Printed_Page": physical_page - 1,
                            "Concept": current_concept,
                            "Gloss": current_gloss,
                            "Lect": current_lect,
                            "Category": category,
                            "Form": form,
                            "Notes": "",
                            "Raw_Fragments": line,
                            "Source_Status": "response",
                        }
                        records.append(current_record)
                        continue

                    if current_record is None:
                        # Wrapped concept headings precede the first response.
                        if current_concept is not None:
                            current_gloss = " ".join(part for part in (current_gloss, line) if part)
                            concepts[current_concept] = current_gloss
                            continue
                        raise ValueError(f"Orphan text on PDF page {physical_page}: {line!r}")

                    # Wrapped forms follow the record whose line immediately precedes them.
                    current_record["Form"] = f"{current_record['Form']} {line}".strip()
                    current_record["Raw_Fragments"] = f"{current_record['Raw_Fragments']} | {line}"

    if set(concepts) != set(range(1, 211)) - set(OMITTED_PROMPTS):
        raise ValueError(f"Unexpected printed concept inventory: {sorted(concepts)}")

    # Repair two documented text-layer artifacts without changing source content.
    for record in records:
        if record["Lect"] == "Gujarati" and record["Concept"] == 98:
            if str(record["Form"]).endswith(" (cid:1)"):
                record["Form"] = str(record["Form"]).removesuffix(" (cid:1)")
                record["Notes"] = "image-verified removal of spurious PDF text-layer (cid:1)"
        if record["Lect"] == "N-Son-Pat" and record["Form"]:
            # One printed label is misspelled N-Son-Paty; it is normalized at
            # the label level below if encountered by future extraction engines.
            pass

    # Complete the source matrix explicitly. Four prompts are absent from the
    # published appendix, and N-Son-Bal item 6 has no printed response row.
    for concept, gloss in OMITTED_PROMPTS.items():
        for lect in LECTS:
            records.append({
                "PDF_Page": "", "Printed_Page": "", "Concept": concept,
                "Gloss": gloss, "Lect": lect, "Category": "", "Form": "",
                "Notes": "prompt absent from the published Appendix A wordlist",
                "Raw_Fragments": "", "Source_Status": "omitted_prompt",
            })
    if not any(record["Concept"] == 6 and record["Lect"] == "N-Son-Bal" for record in records):
        records.append({
            "PDF_Page": 59, "Printed_Page": 58, "Concept": 6, "Gloss": concepts[6],
            "Lect": "N-Son-Bal", "Category": "", "Form": "",
            "Notes": "no response row printed for this lect/concept cell",
            "Raw_Fragments": "", "Source_Status": "implicit_missing",
        })

    occurrence: Counter[tuple[int, str]] = Counter()
    result: list[dict[str, str]] = []
    for record in sorted(
        records,
        key=lambda item: (
            int(item["Concept"]),
            LECTS.index(str(item["Lect"])),
            int(item["PDF_Page"] or 0),
            len(result),
        ),
    ):
        key = (int(record["Concept"]), str(record["Lect"]))
        occurrence[key] += 1
        form = normalize_text(str(record["Form"]))
        # A single category digit is fused to the first source glyph on item 13.
        if record["Concept"] == 13 and record["Lect"] == "N-Son-Bal" and form == "1ct̪":
            record["Category"], form = "1", "ct̪"
            record["Notes"] = "diplomatic image-verified transcription; category digit was fused to the source glyph run"
        result.append({
            "PDF_Page": str(record["PDF_Page"]),
            "Printed_Page": str(record["Printed_Page"]),
            "Concept": str(record["Concept"]),
            "Gloss": normalize_text(str(record["Gloss"])),
            "Lect": str(record["Lect"]),
            "Category": str(record["Category"]),
            "Response_Index": str(occurrence[key]),
            "Form": form,
            "Notes": str(record["Notes"]),
            "Raw_Fragments": str(record["Raw_Fragments"]),
            "Source_Status": str(record["Source_Status"]),
        })
    if len(result) != 4_092:
        raise ValueError(f"Expected 4,092 audited cells/responses, got {len(result)}")
    return result


def write_snapshot(records: list[dict[str, str]]) -> None:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def load_snapshot() -> list[dict[str, str]]:
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 4_092:
        raise ValueError(f"Expected 4,092 snapshot rows, got {len(rows)}")
    return rows


def install(records: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in records:
        lect = row["Lect"]
        target = lect in TARGET_LECTS
        no_entry = row["Category"] == "0" or row["Form"].casefold() == "no entry"
        missing = row["Source_Status"] in {"blank", "omitted_prompt", "implicit_missing"}
        status, reason = "installed", ""
        if not target:
            status, reason = "excluded", "borrowed or standard comparison list"
        elif no_entry:
            status, reason = "excluded", "source explicitly says no entry"
        elif missing:
            status = "excluded"
            reason = (
                "prompt absent from the published appendix"
                if row["Source_Status"] == "omitted_prompt"
                else "no primary form printed for this lect/concept cell"
            )

        concept = int(row["Concept"])
        response_index = int(row["Response_Index"])
        site_id = dialect_id(lect) if target else ""
        page = f"printed p. {row['Printed_Page']}, " if row["Printed_Page"] else ""
        citation = f"{SOURCE_KEY}[Appendix A, {page}item {concept}, {lect}]"
        entry_key = (
            f"silnimadi2012:g{concept:03d}:{site_id}:i{response_index}"
            if status == "installed"
            else ""
        )
        notes = "; ".join(part for part in (
            f"Appendix A lexical-similarity category {row['Category']}" if row["Category"] else "",
            row["Notes"],
        ) if part)
        if status == "installed":
            display = LECT_METADATA[lect][0]
            tag = f"dialect:Nimadi:{site_id}:{quote(display)}"
            forms.append([
                "Nimadi", "", row["Form"], row["Gloss"], "", row["Form"], notes,
                citation, "", "", entry_key, "", "", "", tag,
            ])
        audit.append(dict(zip(AUDIT_FIELDS, [
            "wordlist response" if row["Source_Status"] == "response" else "wordlist matrix cell",
            SOURCE_KEY, row["PDF_Page"], row["Printed_Page"], row["Concept"],
            row["Gloss"], lect, "Nimadi target list" if target else "comparison list",
            row["Category"], row["Response_Index"], row["Raw_Fragments"], row["Form"],
            row["Notes"], status, reason, "Nimadi" if target else "", site_id,
            citation, entry_key,
        ])))
    if len(audit) != 4_092:
        raise ValueError(f"Unexpected audit count: {len(audit)}")
    return forms, audit


def write_outputs(records: list[dict[str, str]]) -> None:
    forms, audit = install(records)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    excluded = Counter(row["Reason"] for row in audit if row["Status"] == "excluded")
    manifest = {
        "source": SOURCE_KEY,
        "title": "The Nimadi-speaking people of Madhya Pradesh: A sociolinguistic profile",
        "source_pdf_url": PDF_URL,
        "source_pdf_archive_url": ARCHIVE_URL,
        "source_pdf_sha256": PDF_SHA256,
        "source_pdf_pages": 176,
        "appendix": "Appendix A, physical PDF pp. 49-101; wordlist data pp. 58-101 (printed pp. 57-100)",
        "extraction": "Unicode Doulos SIL text layer; parity-adjusted three-column parser, with two-column handling for physical pp. 95-99; no OCR",
        "snapshot_sha256": sha256(SNAPSHOT),
        "counts": {
            "standard_prompts": 210,
            "printed_prompts": 206,
            "target_lists": 13,
            "comparison_lists": 5,
            "snapshot_and_audit_records": len(audit),
            "printed_response_records": sum(row["Source_Status"] in {"response", "blank"} for row in records),
            "additional_response_lines": sum(int(row["Response_Index"]) > 1 for row in records),
            "synthetic_omitted_prompt_cells": sum(row["Source_Status"] == "omitted_prompt" for row in records),
            "implicit_missing_cells": sum(row["Source_Status"] == "implicit_missing" for row in records),
            "installed_nimadi_forms": len(forms),
            "excluded_records": sum(row["Status"] == "excluded" for row in audit),
            "explicit_no_entry_records": sum(row["Category"] == "0" or row["Form"].casefold() == "no entry" for row in records),
            "comparison_records": sum(row["Lect"] in CONTROL_LECTS for row in records),
        },
        "lect_counts": dict(sorted(Counter(row[14].split(":", 3)[3] for row in forms).items())),
        "editorial_policy": {
            "scope": "thirteen newly elicited Nimadi lists installed; Parya Bhilali, Malvi, Hindi, Gujarati and Marathi comparison lists retained only in audit",
            "categories": "source lexical-similarity category numbers retained in Notes, never interpreted as etymologies",
            "missing_prompts": "items 11, 23, 24 and 70 are absent from the published appendix and represented as explicit audit-only matrix cells",
            "transcription": "source IPA preserved in Form and Phonemic; only documented extraction artifacts corrected, with image-verification notes",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"installed={len(forms)} comparisons={excluded['borrowed or standard comparison list']} "
        f"no_entry={excluded['source explicitly says no entry']} "
        f"missing={excluded['no primary form printed for this lect/concept cell']} "
        f"omitted={excluded['prompt absent from the published appendix']} audit={len(audit)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="rebuild snapshot from pinned PDF")
    args = parser.parse_args()
    if args.extract:
        write_snapshot(extract_pdf(PDF))
    write_outputs(load_snapshot())


if __name__ == "__main__":
    main()
