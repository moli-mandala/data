#!/usr/bin/env python3
"""Install the manually image-checked wordlists in JLSR 2022-015.

The report's Appendix B.4 is raster-only.  ``manual_transcription.txt`` is the
authoritative diplomatic transcription; ``tesseract_scaffold.txt`` is retained
only as non-authoritative comparison evidence and is never parsed here.
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
MANUAL = HERE / "manual_transcription.txt"
OCR = HERE / "tesseract_scaffold.txt"
IMAGE_MANIFEST = HERE / "image_manifest.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-bagheli.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-bagheli-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-bagheli-manifest.json"

SOURCE_KEY = "koshy2022bagheli"
PDF_URL = "https://www.sil.org/system/files/reapdata/74/43/06/74430698133294939627853202970587913444/JLSR2022_015.pdf"
ARCHIVE_URL = "https://www.sil.org/resources/archives/94596"
PDF_SHA256 = "d1424f317dc12fe01d99d33abd917201575487f4de44529678ecce1c282a4627"
ALL_CODES = set("DKPSabcdehjklmnprst")
CONTROL_CODES = {"h"}
TARGET_CODES = ALL_CODES - CONTROL_CODES

# code -> (dialect id, display label, source locality description)
SITES = {
    "D": ("sil-bagheli-2022-dabhaura", "Dabhaura", "Dabhaura, Theothar tahsil, Rewa district, Madhya Pradesh; source code D"),
    "K": ("sil-bagheli-2022-katkon", "Katkon (Khadi-Hindi)", "Katkon, Nagod tahsil, Satna district, Madhya Pradesh; source label Khadi-Hindi; code K"),
    "P": ("sil-bagheli-2022-amarkantak", "Amarkantak (Pindra-Zamindari)", "Amarkantak, Pushparajgarh tahsil, Anuppur district, Madhya Pradesh; source label Pindra-Zamindari; code P"),
    "S": ("sil-bagheli-2022-sunwari", "Sunwari", "Sunwari, Maihar tahsil, Satna district, Madhya Pradesh; source code S"),
    "a": ("sil-bagheli-2022-karchana", "Karchana (Allahabadi)", "Karchana, Allahabad district, Uttar Pradesh; source label Allahabadi; code a"),
    "b": ("sil-bagheli-2022-baikanthpur", "Baikanthpur", "Baikanthpur, Sirmour tahsil, Rewa district, Madhya Pradesh; source code b"),
    "c": ("sil-bagheli-2022-chawari", "Chawari", "Chawari, Sidhi tahsil and district, Madhya Pradesh; source code c"),
    "d": ("sil-bagheli-2022-dewara", "Dewara", "Dewara, Hanumana tahsil, Rewa district, Madhya Pradesh; source code d"),
    "e": ("sil-bagheli-2022-domahai", "Domahai", "Domahai, Majgama tahsil, Satna district, Madhya Pradesh; source code e"),
    "h": ("", "Standard Hindi", "standard Hindi comparison list; source code h"),
    "j": ("sil-bagheli-2022-janakpur", "Janakpur (Bakhari Boli)", "Janakpur, Bharatpur tahsil, Koriya district, Chhattisgarh; source label Bakhari Boli; code j"),
    "k": ("sil-bagheli-2022-keoti", "Keoti", "Keoti, Sirmour tahsil, Rewa district, Madhya Pradesh; source code k"),
    "l": ("sil-bagheli-2022-lodha", "Lodha (Rimahi Bagheli)", "Lodha, Umaria tahsil and district, Madhya Pradesh; source label Rimahi Bagheli; code l"),
    "m": ("sil-bagheli-2022-kotasiv-prathapsing", "Kotasiv Prathapsing (Mirzapuri)", "Kotasiv Prathapsing, Lalganj tahsil, Mirzapur district, Uttar Pradesh; source label Mirzapuri; code m"),
    "n": ("sil-bagheli-2022-singpur", "Singpur (Sohagpuri)", "Singpur, Sohagpur tahsil, Shahdol district, Madhya Pradesh; source label Sohagpuri; code n"),
    "p": ("sil-bagheli-2022-parasawar", "Parasawar", "Parasawar, Devsar tahsil, Sidhi district, Madhya Pradesh; source code p"),
    "r": ("sil-bagheli-2022-semara", "Semara", "Semara, Jaisingh-Nagar tahsil, Shahdol district, Madhya Pradesh; source code r"),
    "s": ("sil-bagheli-2022-silpari", "Silpari", "Silpari, Rewa tahsil and district, Madhya Pradesh; source code s"),
    "t": ("sil-bagheli-2022-mahdeiya", "Mahdeiya (Singraulihi)", "Mahdeiya, Singrauli tahsil, Sidhi district, Madhya Pradesh; source also prints Thurua/Dabhaura inconsistently; code t"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "PDF_Page", "Printed_Page", "Column", "Line", "Item", "Gloss",
    "Similarity_Group", "Site_Code", "Site", "Scope", "OCR_Evidence", "Manual_Form",
    "Qualifier", "Review_Method", "Review_Status", "Confidence", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Citation", "Entry_Key",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manual() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pdf_page = printed_page = column = item = None
    gloss = ""
    line_number: Counter[tuple[int, int]] = Counter()
    for source_line, raw in enumerate(MANUAL.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("P "):
            _, pdf, printed = line.split()
            pdf_page, printed_page = int(pdf), int(printed)
            column = item = None
            continue
        if line.startswith("C "):
            column = int(line[2:])
            if not 1 <= column <= 4:
                raise ValueError(f"manual line {source_line}: invalid column")
            continue
        if line.startswith("I "):
            match = re.fullmatch(r"I (\d{1,3})\|(.+)", line)
            if not match:
                raise ValueError(f"manual line {source_line}: malformed item line {line!r}")
            item, gloss = int(match.group(1)), match.group(2).strip()
            continue
        if line.startswith(("R ", "B ", "U ")):
            if None in (pdf_page, printed_page, column, item) or not gloss:
                raise ValueError(f"manual line {source_line}: record before page/column/item")
            kind = {"R": "response", "B": "nonlexical", "U": "unassigned"}[line[0]]
            fields = [unicodedata.normalize("NFC", value.strip()) for value in line[2:].split("|")]
            if kind == "response":
                if len(fields) != 4:
                    raise ValueError(f"manual line {source_line}: expected four response fields")
                group, form, codes, qualifier = fields
                # The table normally numbers similarity groups, but item 70
                # also uses the labels A and B; retain the source label.
                if not re.fullmatch(r"(?:\d+|[A-Z])", group) or not form or not codes:
                    raise ValueError(f"manual line {source_line}: incomplete response")
            elif kind == "nonlexical":
                if len(fields) != 2:
                    raise ValueError(f"manual line {source_line}: expected non-lexical codes and reason")
                codes, qualifier = fields
                group = form = ""
                if not codes or not qualifier:
                    raise ValueError(f"manual line {source_line}: incomplete non-lexical record")
            else:
                if len(fields) != 3:
                    raise ValueError(f"manual line {source_line}: expected group, form, and reason")
                group, form, qualifier = fields
                codes = ""
                if not re.fullmatch(r"(?:\d+|[A-Z])", group) or not form or not qualifier:
                    raise ValueError(f"manual line {source_line}: incomplete unassigned response")
            if codes:
                unknown = set(codes) - ALL_CODES
                if unknown:
                    raise ValueError(f"manual line {source_line}: unknown site codes {sorted(unknown)}")
                if len(codes) != len(set(codes)):
                    raise ValueError(f"manual line {source_line}: duplicate site code in {codes!r}")
            line_number[(pdf_page, column)] += 1
            records.append({
                "kind": kind,
                "pdf_page": pdf_page, "printed_page": printed_page, "column": column,
                "line": line_number[(pdf_page, column)], "item": item, "gloss": gloss,
                "group": group, "form": form, "codes": codes, "qualifier": qualifier,
                "source_line": source_line,
            })
            continue
        raise ValueError(f"manual line {source_line}: unknown directive {line!r}")
    return records


def dialect_tag(code: str) -> str:
    dialect_id, display, _ = SITES[code]
    return f"dialect:bagheli_lakshman:{dialect_id}:{quote(display)}"


def build(records: list[dict[str, object]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    accounted: dict[int, set[str]] = defaultdict(set)
    item_locator: dict[int, tuple[int, int, int, str]] = {}
    for record in records:
        item_locator.setdefault(int(record["item"]), (
            int(record["pdf_page"]), int(record["printed_page"]),
            int(record["column"]), str(record["gloss"]),
        ))
        if record["kind"] == "unassigned":
            audit.append(dict(zip(AUDIT_FIELDS, [
                SOURCE_KEY, str(record["pdf_page"]), str(record["printed_page"]),
                str(record["column"]), str(record["line"]), str(record["item"]),
                str(record["gloss"]), str(record["group"]), "", "",
                "unassigned source response", f"tesseract_scaffold.txt: PDF_PAGE {record['pdf_page']} COLUMN {record['column']}",
                str(record["form"]), str(record["qualifier"]),
                "manual transcription from embedded source image; OCR comparison only",
                "unresolved", "high transcription / unresolved assignment", "excluded", "source response has no printed site code",
                "", "", _citation(record, "unassigned"), "",
            ])))
            continue
        for code in str(record["codes"]):
            accounted[int(record["item"])].add(code)
            target = code in TARGET_CODES
            dialect_id, display, _ = SITES[code]
            entry_key = (
                f"silbagheli2022:p{record['pdf_page']}:c{record['column']}:"
                f"l{record['line']:03d}:{code}"
                if target and record["kind"] == "response" else ""
            )
            if entry_key and entry_key in seen_keys:
                raise ValueError(f"Duplicate entry key {entry_key}")
            seen_keys.add(entry_key)
            citation = _citation(record, code)
            notes = "; ".join(part for part in (
                f"lexical-similarity group {record['group']}",
                f"source qualifier: {record['qualifier']}" if record["qualifier"] else "",
            ) if part)
            if target and record["kind"] == "response":
                forms.append([
                    "bagheli_lakshman", "", str(record["form"]), str(record["gloss"]), "",
                    str(record["form"]), notes, citation, "", "", entry_key, "", "", "",
                    dialect_tag(code),
                ])
            source_uncertain = "trailing question mark" in str(record["qualifier"])
            is_nonlexical = record["kind"] == "nonlexical"
            status = "installed" if target and not is_nonlexical else "excluded"
            reason = ""
            if is_nonlexical:
                reason = str(record["qualifier"])
            elif not target:
                reason = "standard Hindi comparison list"
            audit.append(dict(zip(AUDIT_FIELDS, [
                SOURCE_KEY, str(record["pdf_page"]), str(record["printed_page"]),
                str(record["column"]), str(record["line"]), str(record["item"]),
                str(record["gloss"]), str(record["group"]), code, display,
                "Bagheli survey lect" if target else "standard Hindi control",
                f"tesseract_scaffold.txt: PDF_PAGE {record['pdf_page']} COLUMN {record['column']}",
                str(record["form"]), str(record["qualifier"]),
                "manual transcription from embedded source image; OCR comparison only",
                "source-marked-uncertain" if source_uncertain else "complete",
                "high (source uncertainty retained)" if source_uncertain else (
                    "medium (site-code case interpreted)"
                    if code == "l" and "uppercase L" in str(record["qualifier"]) else "high"
                ), status,
                reason, "bagheli_lakshman" if target and not is_nonlexical else "", dialect_id, citation,
                entry_key,
            ])))
    # Appendix B.4 omits prompts 23-24, and has sporadic cells with no line.
    # Add one explicit audit row for every such conceptual source cell.  These
    # rows make the 210 x 19 review denominator independently checkable.
    for item in range(1, 211):
        missing = ALL_CODES - accounted[item]
        if not missing:
            continue
        if item in item_locator:
            pdf_page, printed_page, column, gloss = item_locator[item]
            reason = "no response printed for this site/item"
        else:
            # Items 23 and 24 are absent between items 22 and 25 on printed p.52.
            pdf_page, printed_page, column, gloss = 61, 52, 1, ""
            reason = "prompt absent from the printed Appendix B.4 table"
        for code in sorted(missing):
            dialect_id, display, _ = SITES[code]
            synthetic = {
                "pdf_page": pdf_page, "printed_page": printed_page, "column": column,
                "line": "", "item": item, "gloss": gloss,
            }
            audit.append(dict(zip(AUDIT_FIELDS, [
                SOURCE_KEY, str(pdf_page), str(printed_page), str(column), "",
                str(item), gloss, "", code, display,
                "Bagheli survey lect" if code in TARGET_CODES else "standard Hindi control",
                f"tesseract_scaffold.txt: PDF_PAGE {pdf_page} COLUMN {column}", "", "",
                "manual visual confirmation of absent table cell; OCR comparison only",
                "complete", "high", "excluded", reason, "", dialect_id,
                _citation(synthetic, code), "",
            ])))
    return forms, audit


def _citation(record: dict[str, object], code: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B.4, printed p. {record['printed_page']}, "
        f"item {record['item']}, site {code}]"
    )


def validate_topology(records: list[dict[str, object]], complete: bool = True) -> None:
    items = defaultdict(set)
    for record in records:
        items[int(record["item"])].update(str(record["codes"]))
    if complete:
        prompt_items = {
            int(match.group(1))
            for line in MANUAL.read_text(encoding="utf-8").splitlines()
            if (match := re.fullmatch(r"I (\d{1,3})\|.+", line.strip()))
        }
        expected_prompts = set(range(1, 211)) - {23, 24}
        if prompt_items != expected_prompts:
            raise ValueError(f"Prompt topology differs: missing={sorted(expected_prompts-prompt_items)} extra={sorted(prompt_items-expected_prompts)}")
        if {int(r["pdf_page"]) for r in records} != set(range(59, 82)):
            raise ValueError("Expected manual records on every PDF page 59-81")
    response_cells = sum(len(str(r["codes"])) for r in records if r["kind"] == "response")
    nonlexical_cells = sum(len(str(r["codes"])) for r in records if r["kind"] == "nonlexical")
    print(
        f"manual_records={len(records)} represented_items={len(items)} "
        f"expanded_response_cells={response_cells} nonlexical_response_cells={nonlexical_cells}"
    )


def write_outputs(records: list[dict[str, object]]) -> None:
    forms, audit = build(records)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    response_records = [r for r in records if r["kind"] == "response"]
    nonlexical_records = [r for r in records if r["kind"] == "nonlexical"]
    unassigned_records = [r for r in records if r["kind"] == "unassigned"]
    installed_audit = [row for row in audit if row["Status"] == "installed"]
    nonlexical_audit = [row for row in audit if row["Reason"].startswith("source prints “by name”")]
    blank_audit = [
        row for row in audit
        if row["Reason"] in {
            "no response printed for this site/item",
            "prompt absent from the printed Appendix B.4 table",
        }
    ]
    control_audit = [row for row in audit if row["Reason"] == "standard Hindi comparison list"]
    conceptual_cells = {(row["Item"], row["Site_Code"]) for row in audit if row["Site_Code"]}
    attested_conceptual_cells = {
        (row["Item"], row["Site_Code"])
        for row in audit if row["Site_Code"] and row["Manual_Form"]
    }
    nonlexical_only_cells = {
        (row["Item"], row["Site_Code"]) for row in nonlexical_audit
    } - attested_conceptual_cells
    if len(conceptual_cells) != 210 * 19:
        raise ValueError(f"Expected 3,990 audited conceptual cells, got {len(conceptual_cells)}")
    if len(forms) != len(installed_audit):
        raise ValueError("Installed-form/audit count mismatch")
    manifest = {
        "source": SOURCE_KEY,
        "title": "A Sociolinguistic Study of Bagheli Speakers in Madhya Pradesh",
        "archive_url": ARCHIVE_URL,
        "pdf_url": PDF_URL,
        "pdf_sha256": PDF_SHA256,
        "pdf_pages": 161,
        "wordlist_appendix": "Appendix B.4, physical PDF pp. 59-81, printed pp. 50-72",
        "counts": {
            "prompts": 210,
            "lists": 19,
            "target_bagheli_lists": 18,
            "standard_hindi_controls": 1,
            "manual_response_lines": len(response_records),
            "manual_nonlexical_directives": len(nonlexical_records),
            "expanded_nonlexical_response_cells": len(nonlexical_audit),
            "manual_unassigned_response_lines": len(unassigned_records),
            "expanded_assigned_response_cells": sum(len(str(r["codes"])) for r in response_records),
            "conceptual_source_cells_reviewed": len(conceptual_cells),
            "conceptual_attested_cells": len(attested_conceptual_cells),
            "conceptual_attested_target_cells": sum(
                code in TARGET_CODES for _, code in attested_conceptual_cells
            ),
            "conceptual_attested_control_cells": sum(
                code in CONTROL_CODES for _, code in attested_conceptual_cells
            ),
            "audit_rows_including_alternatives_and_unassigned": len(audit),
            "installed_bagheli_forms": len(forms),
            "excluded_hindi_control_forms": len(control_audit),
            "confirmed_blank_cells": len(blank_audit),
            "confirmed_blank_target_cells": sum(row["Scope"] == "Bagheli survey lect" for row in blank_audit),
            "confirmed_blank_control_cells": sum(row["Scope"] == "standard Hindi control" for row in blank_audit),
            "conceptual_nonlexical_only_cells": len(nonlexical_only_cells),
            "unresolved_unassigned_response_lines": len(unassigned_records),
            "interpreted_site_code_cells": sum(row["Confidence"].startswith("medium") for row in audit),
            "source_marked_uncertain_installed_forms": sum(row["Review_Status"] == "source-marked-uncertain" for row in installed_audit),
        },
        "transcription": {
            "authority": "manual_transcription.txt; every response line visually checked against its embedded Appendix B.4 image",
            "ocr": "tesseract_scaffold.txt is comparison evidence only and never feeds installation",
            "normalization": "NFC only; source IPA is otherwise diplomatic and copied to Form and Phonemic",
            "similarity_groups": "retained as notes; not interpreted as cognacy or etymology",
            "unresolved": [
                "item 191 berəʈɛ: response line has no printed site code; excluded",
                "item 195 reŋeʈe: response line has no printed site code; excluded",
                "item 189 site a bejtʰe: source itself prints a trailing question mark; installed without punctuation and annotated",
            ],
        },
        "artifact_sha256": {
            "manual_transcription": sha256(MANUAL),
            "tesseract_scaffold": sha256(OCR),
            "image_manifest": sha256(IMAGE_MANIFEST),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed={len(forms)} audit={len(audit)} controls={len(control_audit)} blanks={len(blank_audit)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="validate the current review prefix only")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    records = parse_manual()
    validate_topology(records, complete=not args.partial)
    if args.install:
        if args.partial:
            raise SystemExit("Refusing to install a partial manual transcription")
        write_outputs(records)


if __name__ == "__main__":
    main()
