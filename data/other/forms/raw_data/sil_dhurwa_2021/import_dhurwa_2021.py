#!/usr/bin/env python3
"""Guard and stage the exhaustive, manually reviewed Dhurwa 2021 source.

This importer refuses OCR-bearing schemas and never reads PDF text.  It writes
only source-local artifacts; shared registries and CLDF outputs are
outside its scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_KEY = "josephmichael2021dhurwa"
PDF_SHA256 = "92965cbf77b88685a3f46e59053ce027b4a600037c8043d518c477ac7eac341e"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REQUIRED = {
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
}
ALLOWED_STATUSES = {"attested", "source_blank", "ambiguous", "illegible"}
CHUNKS = HERE / "manual_chunks"
CHUNK = CHUNKS / "items_001_041_hand_keyed.tsv"
REGISTRY = HERE / "list_registry.tsv"
PROFILE = HERE / "conversion_profile.tsv"
MANIFEST = HERE / "source_manifest.json"
CHECKPOINT_FORMS = HERE / "checkpoint_forms.csv"
CHECKPOINT_AUDIT = HERE / "checkpoint_audit.tsv"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
    "Scope", "Disposition", "Citation", "Installed_Count", "Entry_Keys",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manual_cells(path: Path = CHUNK) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        assert fields == REQUIRED, f"unexpected ledger schema: {sorted(fields)}"
        assert not any("ocr" in field.casefold() for field in fields)
        rows = list(reader)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        key = (row["Item"], row["Site_Code"])
        assert key not in seen, f"duplicate conceptual cell: {key}"
        seen.add(key)
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Review_Status"] in ALLOWED_STATUSES
        assert row["Column"] in {"1", "2", "3", "4", "5"}
        if row["Review_Status"] == "attested":
            assert row["Manual_Transcription"]
            assert row["Confidence"] in {"high", "medium"}
        else:
            assert not row["Manual_Transcription"]
            assert row["Uncertainty"]
    item_set = frozenset(int(row["Item"]) for row in rows)
    assert item_set in {
        frozenset(range(1, 42)),
        frozenset(range(42, 83)),
        frozenset(range(83, 125)),
        frozenset(range(125, 168)),
        frozenset(range(168, 201)),
    }
    start, end = min(item_set), max(item_set)
    pdf_page, printed_page = {
        1: ("17", "12"),
        42: ("18", "13"),
        83: ("19", "14"),
        125: ("20", "15"),
        168: ("21", "16"),
    }[start]
    assert all(row["PDF_Page"] == pdf_page and row["Printed_Page"] == printed_page for row in rows)
    expected = {(str(item), site) for item in range(start, end + 1) for site in ["TIR", "NET", "DHA", "KUK", "U5"]}
    assert len(rows) == len(expected) and {(row["Item"], row["Site_Code"]) for row in rows} == expected
    return rows


def load_all_manual_cells() -> list[dict[str, str]]:
    ledgers = sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))
    rows = [row for ledger in ledgers for row in load_manual_cells(ledger)]
    expected = {(str(item), site) for item in range(1, 201) for site in ["TIR", "NET", "DHA", "KUK", "U5"]}
    keys = [(row["Item"], row["Site_Code"]) for row in rows]
    assert len(rows) == 1000
    assert len(keys) == len(set(keys)) and set(keys) == expected
    return rows


def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 5 and len({row["Site_Code"] for row in rows}) == 5
    assert Counter(row["Scope"] for row in rows) == Counter(
        known_dhurwa_target=4, unresolved_list_identity=1
    )
    for row in rows:
        assert row["Install"] == ("yes" if row["Scope"] == "known_dhurwa_target" else "no")
        if row["Install"] == "yes":
            assert row["Language_ID"] and row["Dialect_ID"]
    return {row["Site_Code"]: row for row in rows}


def expand_cell(form: str) -> list[str]:
    parts: list[str] = []
    for printed_line in form.split(" | "):
        parts.extend(part.strip() for part in printed_line.split("/") if part.strip())
    return parts


def build_checkpoint(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], dict[str, int]]:
    registry = load_registry()
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        meta = registry[row["Site_Code"]]
        citation = (
            f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {row['Site_Code']}]"
        )
        entry_keys: list[str] = []
        if row["Review_Status"] == "attested" and meta["Install"] == "yes":
            for variant, form in enumerate(expand_cell(row["Manual_Transcription"]), 1):
                entry_key = (
                    f"dhurwa2021:p{int(row['PDF_Page']):03d}:"
                    f"i{int(row['Item']):03d}:{row['Site_Code']}:a{variant}"
                )
                entry_keys.append(entry_key)
                forms.append([
                    meta["Language_ID"], "", form, row["Gloss"], "", form,
                    f"source list {meta['Display_Name']}", citation, "", "",
                    entry_key, "", "", "",
                    f"dialect:pci:{meta['Dialect_ID']}:{meta['Display_Name']}",
                ])
        if meta["Scope"] == "unresolved_list_identity":
            disposition = "audit-only: fifth printed header is blank; identity unresolved"
        elif row["Review_Status"] == "source_blank":
            disposition = "source-explicit blank"
        else:
            disposition = "source-local exhaustive staging"
        audit.append({
            **row,
            "Scope": meta["Scope"],
            "Disposition": disposition,
            "Citation": citation,
            "Installed_Count": str(len(entry_keys)),
            "Entry_Keys": " | ".join(entry_keys),
        })
    counts = {
        "reviewed_cells": len(rows),
        "attested_cells": sum(row["Review_Status"] == "attested" for row in rows),
        "source_blank_cells": sum(row["Review_Status"] == "source_blank" for row in rows),
        "ambiguous_cells": sum(row["Review_Status"] == "ambiguous" for row in rows),
        "illegible_cells": sum(row["Review_Status"] == "illegible" for row in rows),
        "expanded_responses": sum(
            len(expand_cell(row["Manual_Transcription"]))
            for row in rows if row["Review_Status"] == "attested"
        ),
        "known_target_cells": sum(row["Site_Code"] != "U5" for row in rows),
        "known_target_forms": len(forms),
        "unresolved_identity_cells": sum(row["Site_Code"] == "U5" for row in rows),
        "unresolved_identity_responses": sum(
            len(expand_cell(row["Manual_Transcription"]))
            for row in rows if row["Site_Code"] == "U5" and row["Review_Status"] == "attested"
        ),
    }
    expected_by_items = {
        (1, 41): (205, 203, 2, 204, 164, 164, 41, 40),
        (42, 82): (205, 205, 0, 210, 164, 169, 41, 41),
        (83, 124): (210, 210, 0, 211, 168, 169, 42, 42),
        (125, 167): (215, 212, 3, 215, 172, 172, 43, 43),
        (168, 200): (165, 165, 0, 168, 132, 135, 33, 33),
        (1, 200): (1000, 995, 5, 1008, 800, 809, 200, 199),
    }
    item_range = (min(int(row["Item"]) for row in rows), max(int(row["Item"]) for row in rows))
    expected = expected_by_items[item_range]
    observed = tuple(counts[key] for key in [
        "reviewed_cells", "attested_cells", "source_blank_cells",
        "expanded_responses", "known_target_cells", "known_target_forms",
        "unresolved_identity_cells", "unresolved_identity_responses",
    ])
    assert observed == expected
    assert counts["ambiguous_cells"] == counts["illegible_cells"] == 0
    assert len({row[10] for row in forms}) == len(forms)
    return forms, audit, counts


def load_profile() -> list[tuple[str, str]]:
    with PROFILE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == ["Grapheme", "IPA"]
        rows = [(row["Grapheme"], row["IPA"]) for row in reader]
    assert len({source for source, _ in rows}) == len(rows)
    return sorted(rows, key=lambda pair: len(pair[0]), reverse=True)


def convert(form: str, profile: list[tuple[str, str]]) -> str:
    output: list[str] = []
    position = 0
    while position < len(form):
        for source, target in profile:
            if form.startswith(source, position):
                output.append(target)
                position += len(source)
                break
        else:
            raise AssertionError(f"uncovered profile input at {form!r}[{position}]: {form[position]!r}")
    return "".join(output)


def write_checkpoint(forms: list[list[str]], audit: list[dict[str, str]], counts: dict[str, int]) -> None:
    with CHECKPOINT_FORMS.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(forms)
    with CHECKPOINT_AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["manual_review_checkpoint"] = {
        "completed_items": "1-200",
        "remaining_items": "none",
        "remaining_cells": 0,
        **counts,
        "method": "manual visual inspection at 600 dpi with targeted 1200-dpi rechecks; PDF text/OCR not accepted",
        "unresolved_transcriptions": [],
        "unresolved_list_identity": "fifth printed response column has no header",
    }
    manifest["artifacts"] = {
        "manual_ledgers": [
            {"path": str(path.relative_to(HERE)), "sha256": sha256(path)}
            for path in sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))
        ],
        "checkpoint_forms": {"path": CHECKPOINT_FORMS.name, "sha256": sha256(CHECKPOINT_FORMS)},
        "checkpoint_audit": {"path": CHECKPOINT_AUDIT.name, "sha256": sha256(CHECKPOINT_AUDIT)},
        "list_registry": {"path": REGISTRY.name, "sha256": sha256(REGISTRY)},
        "conversion_profile": {"path": PROFILE.name, "sha256": sha256(PROFILE)},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.pdf:
        assert sha256(args.pdf) == PDF_SHA256, "canonical PDF checksum mismatch"
    rows = load_all_manual_cells()
    forms, audit, counts = build_checkpoint(rows)
    profile = load_profile()
    for form_row in forms:
        assert "�" not in convert(form_row[2], profile)
    if args.write:
        write_checkpoint(forms, audit, counts)
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
