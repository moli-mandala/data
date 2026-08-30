#!/usr/bin/env python3
"""Guard and stage only completely reviewed Adi manual ledgers."""

import argparse
import csv
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path


PDF_SHA256 = "8e1500383a02445252a3eb6973a1b011fabea71eb25ad79fc43ba5b78bd1135c"
EXPECTED_CELLS = 307 * 9
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "text scaffold not accepted without cell visual match"
)
SOURCE_KEY = "padung-sako2015adi"
ALLOWED_STATUSES = {"attested", "source_blank", "ambiguous", "illegible"}
LEDGER_FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]
REQUIRED = set(LEDGER_FIELDS)
REGISTRY_FIELDS = [
    "Site_Code", "Scope", "Install", "Language_ID", "Source_Type", "Label",
    "Dialect_ID", "Dialect_Tag", "Glottocode", "Source_Village", "District",
    "Location",
]
RAW_FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = LEDGER_FIELDS + [
    "Scope", "Disposition", "Citation", "Staged_Entry_Keys",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manual_cells(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        assert fields == REQUIRED, f"unexpected ledger schema: {sorted(fields)}"
        assert not any("ocr" in field.casefold() for field in fields)
        rows = list(reader)
    seen = set()
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        key = (row["Item"], row["Site_Code"])
        assert key not in seen, f"duplicate conceptual cell: {key}"
        seen.add(key)
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Reviewer_Method"] == METHOD
        assert row["Review_Status"] in ALLOWED_STATUSES
        if row["Review_Status"] == "attested":
            assert row["Manual_Transcription"]
            assert row["Confidence"] in {"high", "medium"}
            assert len(row["Manual_Transcription"].split(" | ")) == len(
                row["Source_Cognate_Labels"].split(" | ")
            )
        else:
            assert not row["Manual_Transcription"]
            assert row["Uncertainty"]
    return rows


def load_manual_ledgers(paths: list[Path]) -> list[dict[str, str]]:
    rows = [row for path in paths for row in load_manual_cells(path)]
    keys = [(row["Item"], row["Site_Code"]) for row in rows]
    assert len(keys) == len(set(keys)), "duplicate conceptual cells across ledgers"
    return rows


def stage_forms(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    staged = []
    for row in rows:
        if row["Review_Status"] != "attested":
            continue
        forms = row["Manual_Transcription"].split(" | ")
        labels = row["Source_Cognate_Labels"].split(" | ")
        for variant, (form, label) in enumerate(zip(forms, labels), 1):
            staged.append({
                "Item": row["Item"], "Gloss": row["Gloss"],
                "Site_Code": row["Site_Code"], "Form": form,
                "Source_Cognate_Label": label, "Variant": str(variant),
                "Source_Coordinates": (
                    f"PDF p.{row['PDF_Page']} / printed p.{row['Printed_Page']} / "
                    f"item {row['Item']} / {row['Site_Code']} / {row['Column']} column"
                ),
            })
    return staged


def load_registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert list(reader.fieldnames or []) == REGISTRY_FIELDS
        rows = list(reader)
    assert [row["Site_Code"] for row in rows] == [
        "MN", "BR", "RM", "ML", "PL", "AS", "PD", "SM", "BK"
    ]
    assert Counter(row["Scope"] for row in rows) == Counter(target=9)
    assert all(row["Install"] == "yes" and row["Language_ID"] for row in rows)
    assert Counter(row["Language_ID"] for row in rows) == Counter({
        "MisingPadamMiriMinyong": 2, "BoriKarko": 2,
        "BokarRamo": 4, "Milang": 1,
    })
    assert len({row["Dialect_ID"] for row in rows}) == 9
    assert len({row["Dialect_Tag"] for row in rows}) == 9
    assert all(row["Source_Type"] == "elicitation_site" for row in rows)
    assert all(row["Source_Village"] and row["District"] and row["Location"] for row in rows)
    return rows


def build_source_package(
    rows: list[dict[str, str]], registry: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build lossless target staging and one audit row per conceptual cell."""
    require_full_review(rows)
    by_site = {row["Site_Code"]: row for row in registry}
    forms = []
    audit = []
    for row in rows:
        spec = by_site[row["Site_Code"]]
        citation = (
            f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {row['Site_Code']}]"
        )
        entry_keys = []
        disposition = "blank-excluded"
        if row["Review_Status"] == "attested":
            source_forms = row["Manual_Transcription"].split(" | ")
            labels = row["Source_Cognate_Labels"].split(" | ")
            for response, (form, _label) in enumerate(zip(source_forms, labels), 1):
                key = (
                    f"{SOURCE_KEY}:item:{int(row['Item']):03d}:"
                    f"site:{row['Site_Code']}:response:{response}"
                )
                entry_keys.append(key)
                forms.append({
                    "Language_ID": spec["Language_ID"], "Parameter_ID": "",
                    "Form": form, "Gloss": row["Gloss"], "Native": "",
                    "Phonemic": "", "Notes": "", "Source": citation,
                    "Cognateset": "", "Etymology": "", "Entry_Key": key,
                    "Variant_Of_Key": "", "Borrowed_From_Key": "",
                    "Derivation_Parent_Keys": "", "Tags": spec["Dialect_Tag"],
                })
            disposition = "staged"
        elif row["Review_Status"] in {"ambiguous", "illegible"}:
            disposition = "unresolved-excluded"
        audit.append({
            **row, "Scope": spec["Scope"], "Disposition": disposition,
            "Citation": citation, "Staged_Entry_Keys": " | ".join(entry_keys),
        })
    assert len(forms) == 2770
    assert len(audit) == EXPECTED_CELLS
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert all(unicodedata.is_normalized("NFC", value)
               for row in forms + audit for value in row.values())
    return forms, audit


def write_symbol_inventory(forms: list[dict[str, str]], path: Path) -> None:
    counts = Counter(char for row in forms for char in row["Form"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["Codepoint", "Symbol", "Unicode_Name", "Count", "Decision"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for char in sorted(counts, key=ord):
            punctuation = char.isspace() or unicodedata.category(char).startswith("P")
            writer.writerow({
                "Codepoint": f"U+{ord(char):04X}", "Symbol": char,
                "Unicode_Name": unicodedata.name(char, "UNNAMED"),
                "Count": str(counts[char]),
                "Decision": (
                    "preserve reviewed source punctuation or boundary"
                    if punctuation else "preserve NFC diplomatic transcription"
                ),
            })


def write_source_package(
    forms: list[dict[str, str]], audit: list[dict[str, str]], here: Path
) -> None:
    with (here / "staged_forms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FORM_FIELDS, lineterminator="\n")
        writer.writerows(forms)
    with (here / "staged_audit.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    unresolved = [row for row in audit if row["Review_Status"] in {"ambiguous", "illegible"}]
    with (here / "unresolved_readings.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(unresolved)
    write_symbol_inventory(forms, here / "symbol_inventory.tsv")


def require_full_review(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_CELLS:
        raise RuntimeError(
            f"manual visual review incomplete: {EXPECTED_CELLS - len(rows)} "
            f"of {EXPECTED_CELLS} cells unreviewed"
        )


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, action="append")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    if args.pdf:
        assert sha256(args.pdf) == PDF_SHA256, "canonical PDF checksum mismatch"
    ledger_paths = args.ledger or sorted(
        (here / "manual_chunks").glob("items_*_hand_keyed.tsv")
    )
    rows = load_manual_ledgers(ledger_paths)
    registry = load_registry(here / "list_registry.tsv")
    if args.stage:
        require_full_review(rows)
    staged = stage_forms(rows)
    counts = {status: sum(r["Review_Status"] == status for r in rows)
              for status in ALLOWED_STATUSES}
    print(
        f"guarded ledger OK: {len(rows)} reviewed cells; "
        f"{counts['attested']} attested; {counts['source_blank']} source blanks; "
        f"{counts['ambiguous']} ambiguous; {counts['illegible']} illegible; "
        f"{len(staged)} form candidates"
    )
    if args.stage:
        forms, audit = build_source_package(rows, registry)
        write_source_package(forms, audit, here)
        print(f"review complete; staged {len(forms)} forms and {len(audit)} audit rows")


if __name__ == "__main__":
    main()
