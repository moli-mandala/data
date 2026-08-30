#!/usr/bin/env python3
"""Guard manually reviewed Bhumij 2015 ledgers and stage only targets."""

import argparse
import csv
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


PDF_SHA256 = "1dadbe266842c5e07e4efc4d937f2d3f09daacbe10052db6a715635db018e395"
EXPECTED_CELLS = 210 * 18
SOURCE_KEY = "baileymaggard2015bhumij"
SITE_ORDER = "BAI CHA DIG DUM LAD MAD MOH MUN POD UDA MCH MDI MDH MJH HDI SDI SNA ORI".split()
TARGETS = set(SITE_ORDER[:10])
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REQUIRED = {
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
}
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
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
        assert row["Review_Status"] in {"attested", "source_blank", "ambiguous", "illegible"}
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


def load_registry(here: Path | None = None) -> dict[str, dict[str, str]]:
    root = here or Path(__file__).parent
    with (root / "list_registry.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["Site_Code"] for row in rows] == SITE_ORDER
    assert Counter(row["Scope"] for row in rows) == Counter(
        target=10, comparison_control=8
    )
    assert {row["Site_Code"] for row in rows if row["Install"] == "yes"} == TARGETS
    assert all(row["Language_ID"] and row["Dialect_ID"] for row in rows[:10])
    assert len({row["Dialect_ID"] for row in rows[:10]}) == 10
    assert all(not row["Dialect_ID"] for row in rows[10:])
    return {row["Site_Code"]: row for row in rows}


def stage_target_forms(
    rows: list[dict[str, str]], registry: dict[str, dict[str, str]] | None = None
) -> list[dict[str, str]]:
    registry = registry or load_registry()
    staged = []
    for row in rows:
        if row["Target"] != "yes" or row["Review_Status"] != "attested":
            continue
        forms = row["Manual_Transcription"].split(" | ")
        labels = row["Source_Cognate_Labels"].split(" | ")
        for variant, (form, label) in enumerate(zip(forms, labels), 1):
            spec = registry[row["Site_Code"]]
            base_key = f"{spec['Dialect_ID']}-i{int(row['Item']):03d}"
            entry_key = f"{base_key}-a{variant:02d}"
            staged.append({
                "Language_ID": spec["Language_ID"], "Parameter_ID": "",
                "Form": form, "Gloss": row["Gloss"], "Native": "",
                "Phonemic": form,
                "Notes": (
                    f"Manual rendered-source transcription; source similarity label {label}; "
                    f"physical p.{row['PDF_Page']} / printed p.{row['Printed_Page']} / "
                    f"item {row['Item']} / list {row['Site_Code']} / {row['Column']} column"
                ),
                "Source": (
                    f"{SOURCE_KEY}[Appendix B.3, printed p. {row['Printed_Page']}, "
                    f"item {row['Item']}, list {row['Site_Code']}]"
                ),
                "Cognateset": "", "Etymology": "", "Entry_Key": entry_key,
                "Variant_Of_Key": "" if variant == 1 else f"{base_key}-a01",
                "Borrowed_From_Key": "", "Derivation_Parent_Keys": "",
                "Tags": (
                    f"dialect:unr:{spec['Dialect_ID']}:"
                    f"{quote(spec['Display_Name'])}"
                ),
            })
    return staged


def build_audit(
    rows: list[dict[str, str]], registry: dict[str, dict[str, str]] | None = None
) -> list[dict[str, str]]:
    registry = registry or load_registry()
    audit = []
    for row in rows:
        spec = registry[row["Site_Code"]]
        if spec["Install"] == "yes" and row["Review_Status"] == "attested":
            disposition = "target-staged"
        elif spec["Install"] == "yes":
            disposition = "target-source-blank-excluded"
        elif row["Review_Status"] == "attested":
            disposition = "comparison-control-excluded"
        else:
            disposition = "comparison-control-blank-excluded"
        audit.append({
            **row,
            "Scope": spec["Scope"],
            "Dialect_ID": spec["Dialect_ID"],
            "Stable_Cell_ID": (
                f"{spec['Dialect_ID']}-i{int(row['Item']):03d}"
                if spec["Install"] == "yes" else ""
            ),
            "Disposition": disposition,
        })
    return audit


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def profile_inventory(forms: list[dict[str, str]], here: Path | None = None):
    root = here or Path(__file__).parent
    profile = root.parents[4] / "conversion" / "sil-bhumij.txt"
    with profile.open(encoding="utf-8", newline="") as handle:
        graphemes = [row["Grapheme"] for row in csv.DictReader(handle, delimiter="\t")]
    ordered = sorted(graphemes, key=len, reverse=True)
    char_counts = Counter(char for row in forms for char in row["Form"])
    inventory = [{
        "Character": char,
        "Codepoint": f"U+{ord(char):04X}",
        "Unicode_Name": unicodedata.name(char, "UNNAMED"),
        "Occurrences": str(count),
        "Covered_By_Profile": "yes" if any(char in token for token in graphemes) else "no",
    } for char, count in sorted(char_counts.items(), key=lambda pair: ord(pair[0]))]
    unmatched = Counter()
    for row in forms:
        position = 0
        while position < len(row["Form"]):
            match = next(
                (token for token in ordered if row["Form"].startswith(token, position)), None
            )
            if match is None:
                unmatched[row["Form"][position]] += 1
                position += 1
            else:
                position += len(match)
    return inventory, unmatched


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
    paths = args.ledger or sorted((here / "manual_chunks").glob("items_*_hand_keyed.tsv"))
    rows = load_manual_ledgers(paths)
    registry = load_registry(here)
    if args.stage:
        require_full_review(rows)
    staged = stage_target_forms(rows, registry)
    audit = build_audit(rows, registry)
    counts = {status: sum(row["Review_Status"] == status for row in rows)
              for status in {"attested", "source_blank", "ambiguous", "illegible"}}
    print(
        f"guarded ledger OK: {len(rows)} reviewed cells; "
        f"{counts['attested']} attested; {counts['source_blank']} source blanks; "
        f"{counts['ambiguous']} ambiguous; {counts['illegible']} illegible; "
        f"{len(staged)} target form candidates"
    )
    if args.stage:
        assert not counts["ambiguous"] and not counts["illegible"]
        assert len(staged) == 2100 and len(audit) == EXPECTED_CELLS
        assert len({row["Entry_Key"] for row in staged}) == 2100
        write_tsv(here / "staged_forms.tsv", staged, FORM_FIELDS)
        write_tsv(here / "staged_audit.tsv", audit)
        inventory, unmatched = profile_inventory(staged, here)
        assert not unmatched, f"uncovered profile characters: {dict(unmatched)}"
        write_tsv(here / "profile_inventory.tsv", inventory)
        print(
            "review_complete=1 staged_forms=2100 audit_rows=3780 "
            f"profile_characters={len(inventory)} profile_unmatched=0"
        )


if __name__ == "__main__":
    main()
