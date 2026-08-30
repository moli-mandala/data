#!/usr/bin/env python3
"""Guard and stage the manually reviewed Bonda 2022-005 checkpoint.

The importer rejects OCR-bearing ledgers and never reads source PDF text. It
uses the prior JLSR 2022-004 reviewed ledger only after current manual rows are
loaded, solely to build a separate comparison audit.
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
CHUNKS = HERE / "manual_chunks"
REGISTRY = HERE / "list_registry.tsv"
PROFILE = HERE / "conversion_profile.tsv"
MANIFEST = HERE / "source_manifest.json"
FORMS = HERE / "checkpoint_forms.csv"
AUDIT = HERE / "checkpoint_audit.tsv"
RECONCILIATION = HERE / "comparison_reconciliation.tsv"
DUM_RECONCILIATION = HERE / "dumripada_replacement_reconciliation.tsv"
PRIOR_CELLS = HERE.parent / "sil_bonda_didayi_2022" / "extracted_cells.tsv"

SOURCE_KEY = "mathew2022bonda-further"
PDF_SHA256 = "9c4457aa6e73906b34e8c69e790e9d205a9b95cfc2a94ccae054bcb1537dfcfa"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REQUIRED = {
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Similarity_Groups",
    "Source_Qualification", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
}
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    *[field for field in [
        "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
        "Column", "Manual_Transcription", "Similarity_Groups",
        "Source_Qualification", "Review_Status", "Confidence", "Uncertainty",
        "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
    ]],
    "Scope", "Disposition", "Language_ID", "Dialect_ID", "Citation",
    "Installed_Count", "Entry_Keys",
]
RECON_FIELDS = [
    "Item", "Site_Code", "Site_Name", "Current_Reviewed_Transcription",
    "Prior_Source_Key", "Prior_Site_Code", "Prior_Reviewed_Transcription",
    "Match_Status", "Notes",
]
DUM_RECON_FIELDS = [
    "Item", "Gloss", "Current_Site_Code", "Current_Review_Status",
    "Current_Reviewed_Transcription", "Prior_Source_Key", "Prior_Site_Code",
    "Prior_Reviewed_Transcription", "Match_Status", "Integration_Disposition",
    "Notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 11 and len({row["Site_Code"] for row in rows}) == 11
    assert Counter(row["Scope"] for row in rows) == Counter(
        new_target=2, checked_replacement_target=1, republished_comparison=8
    )
    assert sum(row["Install"] == "yes" for row in rows) == 3
    assert all(row["Prior_Site_Code"] for row in rows if row["Scope"] != "new_target")
    return {row["Site_Code"]: row for row in rows}


def load_manual_cells(path: Path | None = None) -> list[dict[str, str]]:
    paths = [path] if path else sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))
    rows: list[dict[str, str]] = []
    for ledger in paths:
        with ledger.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            assert fields == REQUIRED, f"unexpected ledger schema: {sorted(fields)}"
            assert not any("ocr" in field.casefold() for field in fields)
            rows.extend(reader)
    registry = load_registry()
    keys = [(row["Item"], row["Site_Code"]) for row in rows]
    assert len(keys) == len(set(keys))
    if path is None:
        expected = {(str(item), code) for item in range(1, 211) for code in registry}
        assert len(rows) == 2310 and set(keys) == expected
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Review_Status"] in {"attested", "source_blank_no_entry", "excluded_disqualified"}
        if row["Review_Status"] == "attested":
            assert row["Manual_Transcription"]
        elif row["Review_Status"] == "source_blank_no_entry":
            assert not row["Manual_Transcription"] and row["Similarity_Groups"] == "0"
            assert "no entry" in row["Source_Qualification"]
        else:
            assert not row["Manual_Transcription"] and not row["Similarity_Groups"]
        assert row["Confidence"] == "high"
        item = int(row["Item"])
        if item <= 5:
            expected_page = ("15", "10")
        elif item <= 12 or (item == 13 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
            expected_page = ("16", "11")
        elif item <= 19:
            expected_page = ("17", "12")
        elif item <= 27 or (item == 28 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN"}):
            expected_page = ("18", "13")
        elif item <= 33 or (item == 34 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT"}):
            expected_page = ("19", "14")
        elif item <= 40 or (item == 41 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN"}):
            expected_page = ("20", "15")
        elif item <= 46 or (item == 47 and row["Site_Code"] not in {"RON", "ODI"}):
            expected_page = ("21", "16")
        elif item <= 53 or (item == 54 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD"}):
            expected_page = ("22", "17")
        elif item <= 59 or (item == 60 and row["Site_Code"] not in {"RON", "ODI"}):
            expected_page = ("23", "18")
        elif item <= 66 or (item == 67 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD"}):
            expected_page = ("24", "19")
        elif item <= 73 or (item == 74 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}):
            expected_page = ("25", "20")
        elif item <= 80:
            expected_page = ("26", "21")
        elif item <= 85:
            expected_page = ("27", "22")
        elif item == 86 or (item == 87 and row["Site_Code"] not in {"PAR", "RON", "ODI"}):
            expected_page = ("27", "22")
        elif item <= 90:
            expected_page = ("28", "23")
        elif item <= 92:
            expected_page = ("28", "23")
        elif item <= 95:
            expected_page = ("29", "24")
        elif item <= 98 or (item == 99 and row["Site_Code"] not in {"PAR", "RON", "ODI"}):
            expected_page = ("29", "24")
        elif item <= 105:
            expected_page = ("30", "25")
        elif item <= 111 or (item == 112 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
            expected_page = ("31", "26")
        elif item <= 118:
            expected_page = ("32", "27")
        elif item <= 124:
            expected_page = ("33", "28")
        elif item == 125 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}:
            expected_page = ("33", "28")
        elif item == 125:
            expected_page = ("34", "29")
        elif item <= 130:
            expected_page = ("34", "29")
        elif item == 131 and row["Site_Code"] not in {"RON", "ODI"}:
            expected_page = ("34", "29")
        elif item <= 135:
            expected_page = ("35", "30")
        elif item <= 137:
            expected_page = ("35", "30")
        elif item <= 140:
            expected_page = ("36", "31")
        elif item <= 143:
            expected_page = ("36", "31")
        elif item == 144 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN"}:
            expected_page = ("36", "31")
        elif item <= 145:
            expected_page = ("37", "32")
        elif item <= 150:
            expected_page = ("37", "32")
        elif item <= 155:
            expected_page = ("38", "33")
        elif item == 156 or (item == 157 and row["Site_Code"] not in {"PAR", "RON", "ODI"}):
            expected_page = ("38", "33")
        elif item <= 160:
            expected_page = ("39", "34")
        elif item <= 163 or (item == 164 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN"}):
            expected_page = ("39", "34")
        elif item <= 170:
            expected_page = ("40", "35")
        elif item <= 175:
            expected_page = ("41", "36")
        elif item == 176 or (item == 177 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
            expected_page = ("41", "36")
        elif item <= 180:
            expected_page = ("42", "37")
        elif item <= 182 or (item == 183 and row["Site_Code"] not in {"RON", "ODI"}):
            expected_page = ("42", "37")
        elif item <= 185:
            expected_page = ("43", "38")
        elif item <= 189 or (item == 190 and row["Site_Code"] in {"POD", "BON"}):
            expected_page = ("43", "38")
        elif item == 190:
            expected_page = ("44", "39")
        elif item <= 195:
            expected_page = ("44", "39")
        elif item <= 200:
            expected_page = ("45", "40")
        elif item == 201 or (item == 202 and row["Site_Code"] in {"POD", "BON", "DUM", "KAD", "KEN"}):
            expected_page = ("45", "40")
        elif item <= 205:
            expected_page = ("46", "41")
        elif item <= 208:
            expected_page = ("46", "41")
        elif item <= 210:
            expected_page = ("47", "42")
        else:
            raise AssertionError(f"unaccounted source page for item {item}")
        assert (row["PDF_Page"], row["Printed_Page"]) == expected_page
        assert row["Site_Name"] == registry[row["Site_Code"]]["Display_Name"]
    return rows


def expand_cell(value: str) -> list[str]:
    return [part.strip() for part in value.split(" | ") if part.strip()]


def build_checkpoint(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], dict[str, int]]:
    registry = load_registry()
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        meta = registry[row["Site_Code"]]
        citation = (
            f"{SOURCE_KEY}[Appendix A, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, {row['Site_Name']}]"
        )
        keys: list[str] = []
        if meta["Install"] == "yes" and row["Review_Status"] == "attested":
            groups = row["Similarity_Groups"].split("|")
            variants = expand_cell(row["Manual_Transcription"])
            assert len(groups) == len(variants)
            for index, (form, group) in enumerate(zip(variants, groups, strict=True), 1):
                key = f"bonda-further-2022:p{int(row['PDF_Page']):03d}:i{int(row['Item']):03d}:{row['Site_Code']}:a{index}"
                keys.append(key)
                notes = "Appendix A; manually transcribed from rendered source"
                if group:
                    notes += f"; source similarity group {group} (descriptive only)"
                if row["Source_Qualification"] and index == 2:
                    notes += f"; {row['Source_Qualification']}"
                if meta["Scope"] == "checked_replacement_target":
                    notes += "; checked Dumripada list explicitly replaces prior report's list"
                tag = (
                    f"dialect:{meta['Language_ID']}:"
                    f"{quote(meta['Dialect_ID'], safe='')}:"
                    f"{quote(meta['Display_Name'], safe='')}"
                )
                forms.append([
                    meta["Language_ID"], "", form, row["Gloss"], "", form,
                    notes, citation, "", "", key, "", "", "", tag,
                ])
        if row["Review_Status"] == "excluded_disqualified":
            disposition = "excluded: source prompt DISQUALIFIED"
        elif row["Review_Status"] == "source_blank_no_entry":
            disposition = "source blank: printed no entry"
        elif meta["Install"] == "yes":
            disposition = "source-local target staging"
        else:
            disposition = "audit-only: republished comparison list from JLSR 2022-004"
        audit.append({
            **row, "Scope": meta["Scope"], "Disposition": disposition,
            "Language_ID": meta["Language_ID"], "Dialect_ID": meta["Dialect_ID"],
            "Citation": citation, "Installed_Count": str(len(keys)),
            "Entry_Keys": " | ".join(keys),
        })
    counts = {
        "reviewed_cells": len(rows),
        "attested_cells": sum(row["Review_Status"] == "attested" for row in rows),
        "source_blank_cells": sum(row["Review_Status"] == "source_blank_no_entry" for row in rows),
        "excluded_cells": sum(row["Review_Status"] == "excluded_disqualified" for row in rows),
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": sum(len(expand_cell(row["Manual_Transcription"])) for row in rows),
        "target_cells": sum(registry[row["Site_Code"]]["Install"] == "yes" for row in rows),
        "target_forms": len(forms),
        "comparison_cells": sum(registry[row["Site_Code"]]["Install"] == "no" for row in rows),
        "comparison_responses": sum(
            len(expand_cell(row["Manual_Transcription"]))
            for row in rows if registry[row["Site_Code"]]["Install"] == "no"
        ),
    }
    assert tuple(counts.values()) == (2310, 2259, 7, 44, 0, 0, 2394, 630, 644, 1680, 1750)
    assert len({row[10] for row in forms}) == len(forms)
    return forms, audit, counts


def parse_prior_response(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw or raw in {"DISQUALIFIED", "---"}:
        return []
    out: list[str] = []
    inherited = ""
    for piece in [part.strip() for part in raw.rstrip(",").split(",")]:
        match = re.match(r"^(\d+)\s*(.*)$", piece)
        if match:
            inherited, form = match.groups()
        else:
            form = piece
        if inherited != "0" and form.casefold() != "no entry" and form:
            out.append(unicodedata.normalize("NFC", form))
    return out


def build_reconciliation(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    registry = load_registry()
    controls = [row for row in rows if registry[row["Site_Code"]]["Install"] == "no"]
    with PRIOR_CELLS.open(encoding="utf-8", newline="") as handle:
        prior_rows = list(csv.DictReader(handle, delimiter="\t"))
    prior = {
        (row["Item"], row["Site_Code"]): parse_prior_response(row["Raw_Response"])
        for row in prior_rows if row["Site_Code"] in {registry[c["Site_Code"]]["Prior_Site_Code"] for c in controls}
    }
    out: list[dict[str, str]] = []
    for row in controls:
        meta = registry[row["Site_Code"]]
        current = expand_cell(row["Manual_Transcription"])
        old = prior[(row["Item"], meta["Prior_Site_Code"])]
        if current == old:
            status, notes = "exact-diplomatic-match", "current and prior reviewed strings are identical"
        elif [value.replace(":", "ː") for value in current] == old:
            status, notes = "length-mark-rendering-equivalent", "current prints colon where prior source prints IPA length mark"
        elif list(dict.fromkeys(current)) == old:
            status, notes = "repeated-current-response", "current prints the same lexical response under multiple similarity groups"
        else:
            status, notes = "different-current-printing", "retain both reviewed source readings in audit; do not overwrite prior source"
        out.append({
            "Item": row["Item"], "Site_Code": row["Site_Code"], "Site_Name": row["Site_Name"],
            "Current_Reviewed_Transcription": " | ".join(current),
            "Prior_Source_Key": meta["Prior_Source_Key"],
            "Prior_Site_Code": meta["Prior_Site_Code"],
            "Prior_Reviewed_Transcription": " | ".join(old),
            "Match_Status": status, "Notes": notes,
        })
    counts = Counter(row["Match_Status"] for row in out)
    assert len(out) == 1680
    return out, counts


def build_dumripada_reconciliation(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Relate every checked 2002 Dumripada cell to the superseded 1997 list.

    Both ledgers are already frozen before this comparison is built.  The old
    strings are evidence about the replacement relation only and never supply
    or verify a reading in the current manual ledger.
    """
    registry = load_registry()
    meta = registry["DUM"]
    current_rows = {row["Item"]: row for row in rows if row["Site_Code"] == "DUM"}
    with PRIOR_CELLS.open(encoding="utf-8", newline="") as handle:
        prior_rows = list(csv.DictReader(handle, delimiter="\t"))
    prior = {
        row["Item"]: parse_prior_response(row["Raw_Response"])
        for row in prior_rows if row["Site_Code"] == meta["Prior_Site_Code"]
    }
    assert set(current_rows) == set(prior) == {str(item) for item in range(1, 211)}

    out: list[dict[str, str]] = []
    for item in map(str, range(1, 211)):
        row = current_rows[item]
        current = expand_cell(row["Manual_Transcription"])
        old = prior[item]
        if not current and not old:
            status = "both-unattested"
            notes = "neither reviewed source supplies a lexical response"
        elif not current:
            status = f"current-{row['Review_Status']}-prior-attested"
            notes = "the checked current list has no installable response; the prior reading remains audit evidence only"
        elif not old:
            status = "current-attested-prior-unattested"
            notes = "the checked current list supplies a response where the prior list does not"
        elif current == old:
            status = "exact-diplomatic-match"
            notes = "checked current and prior reviewed strings are identical"
        elif [value.replace(":", "ː") for value in current] == old:
            status = "length-mark-rendering-equivalent"
            notes = "checked current list prints colon where the prior source prints IPA length mark"
        elif list(dict.fromkeys(current)) == old:
            status = "repeated-current-response"
            notes = "checked current list repeats a response under multiple similarity groups"
        else:
            status = "different-checked-current-printing"
            notes = "install the checked current reading; preserve the superseded prior reading and citation in this audit"
        out.append({
            "Item": item,
            "Gloss": row["Gloss"],
            "Current_Site_Code": "DUM",
            "Current_Review_Status": row["Review_Status"],
            "Current_Reviewed_Transcription": " | ".join(current),
            "Prior_Source_Key": meta["Prior_Source_Key"],
            "Prior_Site_Code": meta["Prior_Site_Code"],
            "Prior_Reviewed_Transcription": " | ".join(old),
            "Match_Status": status,
            "Integration_Disposition": "checked-2002-list-current; 1997-list-superseded-audit-only",
            "Notes": notes,
        })
    counts = Counter(row["Match_Status"] for row in out)
    assert len(out) == 210
    return out, counts


def load_profile() -> list[tuple[str, str]]:
    with PROFILE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == ["Grapheme", "IPA"]
        rows = [(row["Grapheme"], row["IPA"]) for row in reader]
    return sorted(rows, key=lambda pair: len(pair[0]), reverse=True)


def convert(form: str, profile: list[tuple[str, str]]) -> str:
    output: list[str] = []
    position = 0
    while position < len(form):
        for source, target in profile:
            if form.startswith(source, position):
                output.append(target); position += len(source); break
        else:
            raise AssertionError(f"uncovered profile input at {form!r}[{position}]: {form[position]!r}")
    return "".join(output)


def write_outputs(
    forms, audit, counts, reconciliation, reconciliation_counts,
    dum_reconciliation, dum_reconciliation_counts,
) -> None:
    with FORMS.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    with RECONCILIATION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECON_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(reconciliation)
    with DUM_RECONCILIATION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUM_RECON_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(dum_reconciliation)
    manifest = {
        "source_id": "sil-jlsr-2022-005-bonda-further",
        "source_key": SOURCE_KEY,
        "title": "The Bonda: Further Sociolinguistic Survey",
        "author": "Chacko Mathew", "year": 2022, "survey_completed": 2002,
        "official_archive_id": 92609,
        "official_archive_url": "https://www.sil.org/resources/archives/92609",
        "canonical_pdf_url": "https://www.sil.org/system/files/reapdata/14/59/18/145918519553516788025406164230420247696/JLSR2022_005.pdf",
        "pinned_workspace_path": "tmp/pdfs/bonda_further_2022/JLSR2022_005.pdf",
        "bytes": 1247227, "physical_pages": 74, "sha256": PDF_SHA256,
        "lexical_appendix": {
            "section": "Appendix A: Wordlists", "physical_pdf_pages": "15-47",
            "printed_pages": "10-42", "prompts": 210, "response_lists": 11,
            "conceptual_cells": 2310, "target_lists": 3, "target_cells": 630,
            "comparison_lists": 8, "comparison_cells": 1680,
            "topology_correction": "three two-line locality/language labels were previously miscounted as six lists",
        },
        "manual_review_checkpoint": {
            "completed_items": "1-210", "remaining_items": "none",
            "remaining_cells": 0, **counts,
            "method": "100% manual visual review at 600 dpi with every checkpoint cell rechecked in 1200-dpi crops; OCR/PDF text not accepted",
            "unresolved_transcriptions": [],
        },
        "comparison_reconciliation": {
            "prior_source_key": "mathew-chamberlain2022bonda-didayi",
            "reviewed_comparison_cells": len(reconciliation),
            "status": dict(reconciliation_counts),
            "policy": "eight same-list comparanda audit-only; Dumripada target requires complete checked-replacement reconciliation",
        },
        "dumripada_replacement_reconciliation": {
            "prior_source_key": "mathew-chamberlain2022bonda-didayi",
            "reviewed_replacement_cells": len(dum_reconciliation),
            "status": dict(dum_reconciliation_counts),
            "policy": "checked 2002 Dumripada list is current; superseded 1997 readings remain source-local audit evidence",
        },
        "artifacts": {
            "manual_ledgers": [{"path": str(path.relative_to(HERE)), "sha256": sha256(path)} for path in sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))],
            "checkpoint_forms": {"path": FORMS.name, "sha256": sha256(FORMS)},
            "checkpoint_audit": {"path": AUDIT.name, "sha256": sha256(AUDIT)},
            "comparison_reconciliation": {"path": RECONCILIATION.name, "sha256": sha256(RECONCILIATION)},
            "dumripada_replacement_reconciliation": {"path": DUM_RECONCILIATION.name, "sha256": sha256(DUM_RECONCILIATION)},
            "list_registry": {"path": REGISTRY.name, "sha256": sha256(REGISTRY)},
            "conversion_profile": {"path": PROFILE.name, "sha256": sha256(PROFILE)},
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.pdf:
        assert sha256(args.pdf) == PDF_SHA256, "canonical PDF checksum mismatch"
    rows = load_manual_cells()
    forms, audit, counts = build_checkpoint(rows)
    reconciliation, reconciliation_counts = build_reconciliation(rows)
    dum_reconciliation, dum_reconciliation_counts = build_dumripada_reconciliation(rows)
    profile = load_profile()
    assert all("�" not in convert(row[2], profile) for row in forms)
    if args.write:
        write_outputs(
            forms, audit, counts, reconciliation, reconciliation_counts,
            dum_reconciliation, dum_reconciliation_counts,
        )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    print("reconciliation=" + json.dumps(dict(reconciliation_counts), sort_keys=True))
    print("dumripada_replacement=" + json.dumps(dict(dum_reconciliation_counts), sort_keys=True))


if __name__ == "__main__":
    main()
