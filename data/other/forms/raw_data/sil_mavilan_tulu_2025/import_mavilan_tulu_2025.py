#!/usr/bin/env python3
"""Guard the manual Mavilan Tulu ledger; stage only after full review."""

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PDF = ROOT.parents[4] / "tmp" / "pdfs" / "JLSR2025-005.pdf"
PDF_SHA256 = "d7675b86c9f083eb2389d268078325643979db4a520da776246cf8ecb5fdc629"
EXPECTED_ITEMS = 208
SITE_ORDER = ("MTP", "MTV", "MTE", "MAL", "TUL", "KOD")
TARGETS = {"MTP", "MTV", "MTE"}
EXPECTED_CELLS = EXPECTED_ITEMS * len(SITE_ORDER)
SOURCE_KEY = "canvin2025"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "900/1200-dpi crops used for dense glyphs; text scaffold not accepted"
)
STATUSES = {"attested", "source_blank", "ambiguous", "illegible"}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Page_Row", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict[str, dict[str, str]]:
    with (ROOT / "list_registry.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["Site_Code"] for row in rows] == list(SITE_ORDER)
    assert Counter(row["Scope"] for row in rows) == Counter(target=3, control=3)
    assert all((row["Install"] == "yes") == (row["Site_Code"] in TARGETS) for row in rows)
    return {row["Site_Code"]: row for row in rows}


def load_cells() -> list[dict[str, str]]:
    paths = sorted((ROOT / "manual_chunks").glob("items_*_hand_keyed.tsv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            assert list(reader.fieldnames or []) == FIELDS, f"unexpected schema: {path}"
            assert not any("ocr" in field.casefold() for field in reader.fieldnames or [])
            rows.extend(reader)
    seen = set()
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        item = int(row["Item"])
        assert 1 <= item <= EXPECTED_ITEMS
        assert row["Site_Code"] in SITE_ORDER
        key = (item, row["Site_Code"])
        assert key not in seen, f"duplicate cell: {key}"
        seen.add(key)
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Reviewer_Method"] == METHOD
        assert row["Review_Status"] in STATUSES
        assert row["PDF_Page"].isdigit() and 28 <= int(row["PDF_Page"]) <= 38
        assert row["Printed_Page"].isdigit() and 22 <= int(row["Printed_Page"]) <= 32
        assert row["Column"] in {"left", "middle", "right"}
        assert row["Page_Row"].isdigit()
        if row["Review_Status"] == "attested":
            assert row["Manual_Transcription"]
            assert row["Confidence"] in {"high", "medium"}
        else:
            assert not row["Manual_Transcription"]
            assert row["Uncertainty"]
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["Review_Status"] for row in rows)
    return {
        "reviewed_cells": len(rows),
        "attested_cells": counts["attested"],
        "source_blank_cells": counts["source_blank"],
        "ambiguous_cells": counts["ambiguous"],
        "illegible_cells": counts["illegible"],
        "target_reviewed_cells": sum(row["Site_Code"] in TARGETS for row in rows),
        "control_reviewed_cells": sum(row["Site_Code"] not in TARGETS for row in rows),
        "pending_cells": EXPECTED_CELLS - len(rows),
    }


def stage(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assert len(rows) == EXPECTED_CELLS, (
        f"refusing staging: only {len(rows)}/{EXPECTED_CELLS} cells have manual decisions"
    )
    assert all(row["Review_Status"] in {"attested", "source_blank"} for row in rows), (
        "refusing staging: ambiguous or illegible cells remain unresolved"
    )
    registry = load_registry()
    forms = []
    audit = []
    for row in rows:
        disposition = "target-form" if row["Site_Code"] in TARGETS else "control-excluded"
        if row["Review_Status"] == "source_blank":
            disposition = "source-blank"
        audit.append({**row, "Scope": registry[row["Site_Code"]]["Scope"], "Disposition": disposition})
        if row["Site_Code"] not in TARGETS or row["Review_Status"] != "attested":
            continue
        forms.append({
            "Language_ID": registry[row["Site_Code"]]["Language_ID"],
            "Parameter_ID": "", "Form": row["Manual_Transcription"],
            "Gloss": row["Gloss"], "Native": "",
            "Phonemic": row["Manual_Transcription"],
            "Notes": (
                f"JLSR 2025-005 physical p.{row['PDF_Page']} / printed p.{row['Printed_Page']} / "
                f"item {row['Item']} / {row['Site_Code']} / {row['Column']} column"
            ),
            "Source": (
                f"{SOURCE_KEY}[p. {row['Printed_Page']}, item {row['Item']}, "
                f"{row['Site_Code']}]"
            ),
            "Cognateset": "", "Etymology": "",
            "Entry_Key": f"sil-mavilan-2025-{int(row['Item']):03d}-{row['Site_Code'].lower()}",
            "Variant_Of_Key": "",
            "Borrowed_From_Key": "", "Derivation_Parent_Keys": "", "Tags": "",
        })
    return forms, audit


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def profile_inventory(forms: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Inventory source characters and propose any additions to the existing profile."""
    profile_path = ROOT.parents[4] / "conversion" / "markodi.txt"
    with profile_path.open(encoding="utf-8", newline="") as handle:
        profile_rows = list(csv.DictReader(handle, delimiter="\t"))
    graphemes = {row["Grapheme"] for row in profile_rows}
    ordered = sorted(graphemes, key=len, reverse=True)
    character_counts = Counter(char for row in forms for char in row["Form"])
    inventory = []
    for char, count in sorted(character_counts.items(), key=lambda pair: ord(pair[0])):
        inventory.append({
            "Character": char,
            "Codepoint": f"U+{ord(char):04X}",
            "Unicode_Name": unicodedata.name(char, "UNNAMED"),
            "Occurrences": str(count),
            "Covered_By_Existing_Profile": "yes" if any(char in token for token in graphemes) else "no",
        })

    unmatched = Counter()
    for row in forms:
        form = row["Form"]
        position = 0
        while position < len(form):
            match = next((token for token in ordered if form.startswith(token, position)), None)
            if match is None:
                unmatched[form[position]] += 1
                position += 1
            else:
                position += len(match)
    additions = [
        {
            "Grapheme": char,
            "Proposed_IPA": char,
            "Codepoint": f"U+{ord(char):04X}",
            "Unicode_Name": unicodedata.name(char, "UNNAMED"),
            "Occurrences": str(count),
            "Rationale": "literal source character absent from existing markodi profile",
        }
        for char, count in sorted(unmatched.items(), key=lambda pair: ord(pair[0]))
    ]
    return inventory, additions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    assert sha256(PDF) == PDF_SHA256, "canonical PDF hash mismatch"
    load_registry()
    rows = load_cells()
    summary = summarize(rows)
    if args.stage:
        forms, audit = stage(rows)
        write_tsv(ROOT / "staged_forms.tsv", forms)
        write_tsv(ROOT / "staged_audit.tsv", audit)
        inventory, additions = profile_inventory(forms)
        write_tsv(ROOT / "profile_inventory.tsv", inventory)
        if additions:
            write_tsv(ROOT / "profile_additions.tsv", additions)
        else:
            (ROOT / "profile_additions.tsv").write_text(
                "Grapheme\tProposed_IPA\tCodepoint\tUnicode_Name\tOccurrences\tRationale\n",
                encoding="utf-8",
            )
        summary["staged_target_forms"] = len(forms)
        summary["profile_additions_required"] = len(additions)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
