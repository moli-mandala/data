#!/usr/bin/env python3
"""Install the visually reviewed Appendix B.5 Korwa and Kodaku wordlists.

Only ``manual_review.tsv`` is parsed.  The embedded text and Tesseract files
are retained as comparison scaffolds and never feed installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE_ROOT = HERE.parents[5]
MANUAL = HERE / "manual_review.tsv"
PAGE_REVIEW = HERE / "page_review.tsv"
TEXT_SCAFFOLD = HERE / "text_layer_scaffold.txt"
OCR = HERE / "tesseract_scaffold.txt"
PDF = WORKSPACE_ROOT / "tmp/pdfs/korwa-kodaku-2022/source.pdf"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-korwa-kodaku.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-korwa-kodaku-manifest.json"
UNRESOLVED = HERE / "unresolved_source_codes.tsv"

SOURCE_KEY = "behera2022korwakodaku"
RECORD_URL = "https://www.sil.org/resources/publications/entry/94564"
PDF_URL = "https://www.sil.org/system/files/reapdata/13/03/86/13038659512317049919318473837327540493/JLSR2022_014.pdf"
WAYBACK_URL = "https://web.archive.org/web/20240617131527id_/https://www.sil.org/system/files/reapdata/13/03/86/13038659512317049919318473837327540493/JLSR2022_014.pdf"
PDF_SHA256 = "a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c"

ALL_CODES = set("ABCDGHJKLMRSTUVZbcdjkmptw")
KORWA_CODES = set("CDGHKLMRZ")
KODAKU_CODES = set("Sbcdjmptw")
TARGET_CODES = KORWA_CODES | KODAKU_CODES
CONTROL_CODES = ALL_CODES - TARGET_CODES

# code -> language id, dialect id, display label, scope
SITES = {
    "C": ("kw", "sil-korwa-2004-chilma", "Chilma Korwa", "Korwa target"),
    "D": ("kw", "sil-korwa-2004-dhaneshpur", "Dhaneshpur Korwa", "Korwa target"),
    "G": ("kw", "sil-korwa-2005-gaseband", "Gaseband Korwa", "Korwa target"),
    "H": ("kw", "sil-korwa-2004-harrapat", "Harrapat Korwa", "Korwa target"),
    "K": ("kw", "sil-korwa-2004-bladerpat", "Bladerpat Korwa", "Korwa target"),
    "L": ("kw", "sil-korwa-2004-kirkima", "Kirkima Korwa", "Korwa target"),
    "M": ("kw", "sil-korwa-2004-musakhoel", "Musakhoel Korwa", "Korwa target"),
    "R": ("kw", "sil-korwa-2004-rakkaya", "Rakkaya Korwa", "Korwa target"),
    "Z": ("kw", "sil-korwa-2005-sardih", "Sardih Korwa", "Korwa target"),
    "S": ("Kodaku", "sil-kodaku-2004-sagardinwa", "Sagardinwa Kodaku", "Kodaku target"),
    "b": ("Kodaku", "sil-kodaku-2005-jamuniatanr", "Jamuniatanr Kodaku", "Kodaku target"),
    "c": ("Kodaku", "sil-kodaku-2005-chainpur", "Chainpur Kodaku", "Kodaku target"),
    "d": ("Kodaku", "sil-kodaku-2005-dhengura", "Dhengura Kodaku", "Kodaku target"),
    "j": ("Kodaku", "sil-kodaku-2005-jhaleria", "Jhaleria Kodaku", "Kodaku target"),
    "m": ("Kodaku", "sil-kodaku-2005-chilma", "Chilma Kodaku", "Kodaku target"),
    "p": ("Kodaku", "sil-kodaku-2005-kodakupara", "Kodakupara Kodaku", "Kodaku target"),
    "t": ("Kodaku", "sil-kodaku-2005-tharki", "Tharki Kodaku", "Kodaku target"),
    "w": ("Kodaku", "sil-kodaku-2005-baikanthpur", "Baikanthpur Kodaku", "Kodaku target"),
    "A": ("", "", "Husambu Asuri", "Asuri comparison control"),
    "B": ("", "", "Dumortoli Sadri", "Sadri comparison control"),
    "J": ("", "", "Arahas Birjia", "Birjia comparison control"),
    "T": ("", "", "Tanmai mixed variety", "Tanmai comparison control"),
    "U": ("", "", "Mahuabathan Mundari", "Mundari comparison control"),
    "V": ("", "", "Kotgohna Sadri", "Sadri comparison control"),
    "k": ("", "", "Kesrelmal Standard Sadri", "Sadri comparison control"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "PDF_Page", "Printed_Page", "Source_Line", "Item", "Gloss",
    "Similarity_Groups", "Site_Code", "Site", "Scope", "Manual_Form",
    "Confidence", "Review_Method", "Review_Status", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Citation", "Entry_Keys",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual_rows() -> list[dict[str, str]]:
    with MANUAL.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 2900:
        raise ValueError(f"Expected 2,900 reviewed response lines, got {len(rows)}")
    if any(row["Review_Status"] != "complete" for row in rows):
        raise ValueError("Manual response-line review is incomplete")
    if {int(row["PDF_Page"]) for row in rows} != set(range(66, 91)):
        raise ValueError("Expected reviewed Appendix B.5 pages 66-90")
    return rows


def dialect_tag(code: str) -> str:
    language, dialect, display, _ = SITES[code]
    return f"dialect:{language}:{dialect}:{quote(display)}"


def slash_variants(form: str) -> list[str]:
    # The source has exactly one slash construction, item 104/c:
    # koda/koɖi hɔpoɲ. Expand the shared following material diplomatically.
    if form == "koda/koɖi hɔpoɲ":
        return ["koda hɔpoɲ", "koɖi hɔpoɲ"]
    if "/" in form:
        raise ValueError(f"Unreviewed slash construction: {form!r}")
    return [form]


def build(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], list[dict[str, str]]]:
    by_cell: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    unresolved: list[dict[str, str]] = []
    glosses: dict[int, str] = {}
    item_pages: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        item = int(row["Item"])
        glosses[item] = row["Gloss"]
        item_pages[item].add(int(row["PDF_Page"]))
        for code in row["Site_Codes"]:
            if code in ALL_CODES:
                by_cell[(item, code)].append(row)
            else:
                unresolved.append({
                    "PDF_Page": row["PDF_Page"], "Printed_Page": row["Printed_Page"],
                    "Source_Line": row["Line"], "Item": row["Item"], "Gloss": row["Gloss"],
                    "Similarity_Group": row["Similarity_Group"], "Manual_Form": row["Manual_Form"],
                    "Unknown_Site_Code": code, "Confidence": "high transcription / unresolved assignment",
                    "Resolution": "excluded; source code has no wordlist metadata and was not reassigned",
                })
    # Items 23 and 24 have page-level NO ENTRY statements rather than rows.
    glosses.update({23: "urine", 24: "feces"})
    item_pages.update({23: {68}, 24: {68}})

    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for item in range(1, 211):
        for code in sorted(ALL_CODES, key="ABCDGHJKLMRSTUVZbcdjkmptw".index):
            language, dialect, display, scope = SITES[code]
            source_rows = by_cell.get((item, code), [])
            explicit_no_entry = any("entry" in row["Manual_Form"].lower() for row in source_rows)
            attested_rows = [row for row in source_rows if "entry" not in row["Manual_Form"].lower()]
            # Preserve first source occurrence and merge group labels when an
            # identical form is repeated in multiple similarity groups.
            unique: dict[str, list[dict[str, str]]] = {}
            for row in attested_rows:
                unique.setdefault(unicodedata.normalize("NFC", row["Manual_Form"]), []).append(row)
            if not unique:
                pages = sorted(item_pages[item])
                if item in {23, 24}:
                    reason = "source prints NO ENTRY for the entire item"
                elif explicit_no_entry:
                    reason = "source explicitly assigns NO ENTRY to this site"
                else:
                    reason = "no response printed for this site/item in the compressed table"
                status = "missing" if code in TARGET_CODES else "excluded"
                citation = f"{SOURCE_KEY}[Appendix B.5, printed p. {pages[0] - 10}, item {item}, site {code}]"
                audit.append(dict(zip(AUDIT_FIELDS, [
                    SOURCE_KEY, ";".join(map(str, pages)), ";".join(str(page - 10) for page in pages),
                    "", str(item), glosses[item], "", code, display, scope, "", "high",
                    "manual visual review of typeset source; text/OCR comparison only",
                    "confirmed-blank", status, reason, language if code in TARGET_CODES else "",
                    dialect if code in TARGET_CODES else "", citation, "",
                ])))
                continue
            for form_index, (source_form, occurrences) in enumerate(unique.items(), 1):
                groups = list(dict.fromkeys(row["Similarity_Group"] for row in occurrences))
                pages = list(dict.fromkeys(int(row["PDF_Page"]) for row in occurrences))
                lines = [f"p{row['PDF_Page']}:l{row['Line']}" for row in occurrences]
                citation = (
                    f"{SOURCE_KEY}[Appendix B.5, printed p. {pages[0] - 10}, "
                    f"item {item}, site {code}]"
                )
                entry_keys: list[str] = []
                if code in TARGET_CODES:
                    expanded = slash_variants(source_form)
                    for variant_index, installed_form in enumerate(expanded, 1):
                        suffix = f"v{variant_index}" if len(expanded) > 1 else f"f{form_index}"
                        entry_key = f"silkorwakodaku2022:i{item:03d}:{code}:{suffix}"
                        entry_keys.append(entry_key)
                        note_parts = [
                            "manually verified against typeset source image",
                            f"lexical-similarity group(s) {','.join(groups)} (non-etymological)",
                        ]
                        if len(unique) > 1:
                            note_parts.append(f"source cell variant {form_index}/{len(unique)}")
                        if len(expanded) > 1:
                            note_parts.append(f"expanded source slash alternative {variant_index}/{len(expanded)}")
                        forms.append([
                            language, "", installed_form, glosses[item], "", installed_form,
                            "; ".join(note_parts), citation, "", "", entry_key, "", "", "",
                            dialect_tag(code),
                        ])
                audit.append(dict(zip(AUDIT_FIELDS, [
                    SOURCE_KEY, ";".join(map(str, pages)),
                    ";".join(str(page - 10) for page in pages), ";".join(lines), str(item),
                    glosses[item], ";".join(groups), code, display, scope, source_form, "high",
                    "manual visual review of typeset source; text/OCR comparison only", "complete",
                    "installed" if code in TARGET_CODES else "excluded",
                    "" if code in TARGET_CODES else "comparison control not installed",
                    language if code in TARGET_CODES else "", dialect if code in TARGET_CODES else "",
                    citation, "|".join(entry_keys),
                ])))
    return forms, audit, unresolved


def validate(forms: list[list[str]], audit: list[dict[str, str]], unresolved: list[dict[str, str]]) -> None:
    cells = {(int(row["Item"]), row["Site_Code"]) for row in audit}
    if len(cells) != 210 * 25:
        raise ValueError(f"Expected 5,250 accounted cells, got {len(cells)}")
    if len(forms) != 4458:
        raise ValueError(f"Expected 4,458 installed target rows, got {len(forms)}")
    if len(unresolved) != 2:
        raise ValueError(f"Expected two unidentified source-code assignments, got {len(unresolved)}")
    keys = [row[10] for row in forms]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate installed Entry_Key")
    if any(unicodedata.normalize("NFC", row[2]) != row[2] for row in forms):
        raise ValueError("Installed form is not NFC")


def write(forms: list[list[str]], audit: list[dict[str, str]], unresolved: list[dict[str, str]]) -> None:
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    with UNRESOLVED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unresolved[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(unresolved)
    manifest = {
        "source": SOURCE_KEY,
        "title": "A Sociolinguistic Profile of Korwa and Kodaku Tribes in Chhattisgarh and Jharkhand, India",
        "record_url": RECORD_URL,
        "pdf_url": PDF_URL,
        "archived_pdf_url": WAYBACK_URL,
        "pdf_sha256": PDF_SHA256,
        "pdf_bytes": 2198621,
        "pdf_pages": 115,
        "wordlist_appendix": "Appendix B.5, physical PDF pp. 66-90, printed pp. 56-80",
        "counts": {
            "prompts": 210,
            "lists": 25,
            "target_lists": 18,
            "comparison_controls": 7,
            "printed_response_lines_visually_reviewed": 2900,
            "conceptual_cells_manually_audited": 5250,
            "target_cells_manually_audited": 3780,
            "target_attested_cells": 3730,
            "target_blank_or_unlisted_cells": 50,
            "control_attested_cells": 1453,
            "control_blank_or_unlisted_cells": 17,
            "unique_source_form_assignments": 6068,
            "installed_target_rows_after_slash_expansion": len(forms),
            "unidentified_source_code_assignments": len(unresolved),
            "ambiguous_or_illegible_installed_forms": 0,
        },
        "policy": {
            "authority": "manual_review.tsv after visual comparison with all 25 rendered Appendix B.5 pages",
            "image_only_or_handwritten_cells": "none; Appendix B.5 is typeset and has an embedded text layer",
            "text_layer": "navigation/scaffold only; every printed response line and bracket was visually checked",
            "ocr": "comparison scaffold only; never parsed by the importer",
            "normalization": "NFC only; source phonetic transcription otherwise retained diplomatically",
            "similarity_groups": "retained in Notes only; never treated as cognacy or etymology",
        },
        "unresolved": [
            "PDF p.73 / printed p.63, item 83, buluŋg: bracket contains unidentified lowercase site code u",
            "PDF p.84 / printed p.74, item 173, nɐʔa: bracket contains unidentified lowercase site code n",
            "13 known site/item combinations have no response printed in the compressed table; audited as unlisted blanks",
        ],
        "artifact_sha256": {
            "manual_review": sha256(MANUAL),
            "page_review": sha256(PAGE_REVIEW),
            "text_layer_scaffold": sha256(TEXT_SCAFFOLD),
            "tesseract_scaffold": sha256(OCR),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify-pdf", action="store_true")
    args = parser.parse_args()
    if args.verify_pdf:
        if sha256(PDF) != PDF_SHA256:
            raise SystemExit("Canonical PDF SHA-256 mismatch")
    rows = manual_rows()
    forms, audit, unresolved = build(rows)
    validate(forms, audit, unresolved)
    if args.install:
        write(forms, audit, unresolved)
    print(
        f"response_lines={len(rows)} cells=5250 installed={len(forms)} "
        f"audit={len(audit)} unresolved_codes={len(unresolved)}"
    )


if __name__ == "__main__":
    main()
