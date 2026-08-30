from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_ho_2024"
IMPORTER = PACKAGE / "import_ho.py"
PDF = ROOT.parent / "tmp/pdfs/ho_2024/JLSR2024_009.pdf"

spec = importlib.util.spec_from_file_location("sil_ho_2024_importer", IMPORTER)
ho = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(ho)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_chunk(path: Path, entries: list[dict[str, str]], fields=None) -> None:
    fieldnames = fields or ho.CHUNK_FIELDS
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader(); writer.writerows(entries)


def chunk_row(base: dict[str, str], **updates: str) -> dict[str, str]:
    row = {field: base.get(field, "") for field in ho.CHUNK_FIELDS}
    row.update({
        "Gloss": base.get("Gloss") or "light",
        "Manual_Transcription": "1 manual",
        "Review_Status": "attested",
        "Confidence": "high",
        "Uncertainty": "",
        "Reviewer_Method": "manual-source-image; rendered-400dpi; OCR-not-accepted",
        "Reviewed_At": "2026-08-28",
        "Reviewer_Declaration": ho.DECLARATION,
    })
    row.update(updates)
    return row


def test_source_pin_and_exact_topology():
    assert PDF.stat().st_size == 12_467_726
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == ho.PDF_SHA256
    base = ho.validate_base()
    assert len(base) == 5670
    assert len({(row["Item"], row["Site_Code"]) for row in base}) == 5670
    assert {int(row["Item"]) for row in base} == set(range(1, 211))
    assert {int(row["PDF_Page"]) for row in base} == set(range(72, 142))
    assert all(int(row["Printed_Page"]) == int(row["PDF_Page"]) - 9 for row in base)


def test_base_statuses_nfc_coordinates_and_manual_method_stamps():
    base = ho.validate_base()
    counts = ho.validate_effective(base)
    assert counts == Counter(attested=3595, blank=292, ambiguous=1, unreviewed=1782)
    reviewed = [row for row in base if row["Review_Status"] != "unreviewed"]
    assert len(reviewed) == 3888
    assert {(int(row["PDF_Page"]), int(row["Item"])) for row in reviewed} >= {(72, 1), (119, 144)}
    assert all(row["Reviewer_Method"] == "manual-source-image; rendered-180dpi; OCR-not-accepted" for row in reviewed)
    assert all(row["Reviewer_Declaration"] == ho.DECLARATION for row in reviewed)
    assert all(unicodedata.is_normalized("NFC", value) for row in base for value in row.values())
    unresolved = [row for row in reviewed if row["Review_Status"] in {"ambiguous", "illegible"}]
    assert [(r["PDF_Page"], r["Printed_Page"], r["Item"], r["Site_Code"]) for r in unresolved] == [("75", "66", "10", "HO3")]


def test_source_subset_accounting_and_target_only_staging():
    base = ho.validate_base(); specs = ho.validate_registry()
    by_role = ho.role_counts(base)
    assert by_role["target"] == Counter(attested=1986, blank=30, unreviewed=924)
    assert by_role["republished_control"] == Counter(attested=309, blank=122, ambiguous=1, unreviewed=198)
    assert by_role["comparison_control"] == Counter(attested=1300, blank=140, unreviewed=660)
    forms, audit = ho.build(base, specs)
    assert len(forms) == 1986 and len(audit) == 5670
    assert all(row[0] == "ho" and not re.match(r"^\d", row[2]) for row in forms)
    assert all(row["Disposition"] != "staged" for row in audit if row["Scope"] != "target")
    assert {row["Site_Code"] for row in audit if row["Disposition"] == "staged"} <= set(ho.TARGETS)
    build_source = IMPORTER.read_text(encoding="utf-8").split("def build(", 1)[1].split("def write_unresolved", 1)[0]
    assert "OCR_Evidence_Only" not in build_source
    assert ho.strip_similarity_labels("1 unɖi kui, 4(3 misi)era, 8 mai") == "unɖi kui, (3 misi)era, mai"
    assert ho.strip_similarity_labels("1, 3 miet") == "miet"


def test_overlay_is_keyed_disjoint_coordinate_exact_and_ocr_blind(tmp_path: Path):
    base = ho.validate_base(); p120 = next(row for row in base if row["Item"] == "145" and row["Site_Code"] == "HO1")
    good = tmp_path / "pages_good.tsv"; write_chunk(good, [chunk_row(p120)])
    effective = ho.overlay_manual_chunks(base, [good])
    patched = next(row for row in effective if row["Item"] == "145" and row["Site_Code"] == "HO1")
    assert patched["Manual_Transcription"] == "1 manual" and patched["Review_Status"] == "attested"

    duplicate = tmp_path / "pages_duplicate.tsv"; write_chunk(duplicate, [chunk_row(p120)])
    with pytest.raises(ValueError, match="Duplicate review-chunk key"):
        ho.overlay_manual_chunks(base, [good, duplicate])

    reviewed = next(row for row in base if row["Item"] == "1" and row["Site_Code"] == "HO1")
    overlap = tmp_path / "pages_overlap.tsv"; write_chunk(overlap, [chunk_row(reviewed)])
    with pytest.raises(ValueError, match="overlaps reviewed base"):
        ho.overlay_manual_chunks(base, [overlap])

    unknown_row = chunk_row(p120, Item="999")
    unknown = tmp_path / "pages_unknown.tsv"; write_chunk(unknown, [unknown_row])
    with pytest.raises(ValueError, match="Unknown review-chunk key"):
        ho.overlay_manual_chunks(base, [unknown])

    ocr_fields = ho.CHUNK_FIELDS + ["OCR_Evidence_Only"]
    ocr_row = chunk_row(p120); ocr_row["OCR_Evidence_Only"] = "copied scaffold"
    ocr = tmp_path / "pages_ocr.tsv"; write_chunk(ocr, [ocr_row], ocr_fields)
    with pytest.raises(ValueError, match="OCR-bearing review chunk is inadmissible"):
        ho.overlay_manual_chunks(base, [ocr])


def test_base_only_bypass_refuses_staging_and_incomplete_base_is_guarded():
    before = (PACKAGE / "staged_forms.csv").read_bytes()
    result = subprocess.run(
        ["python3", str(IMPORTER), "--verify-pdf", "--base-only", "--stage"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Refusing to stage: --base-only bypasses review chunks" in result.stderr
    assert (PACKAGE / "staged_forms.csv").read_bytes() == before
    with pytest.raises(RuntimeError, match="manual visual review incomplete: 1782 of 5,670 cells unreviewed"):
        ho.require_complete(ho.validate_base())


def test_manifest_checklist_and_unresolved_audit_match_current_admissible_state():
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["conceptual_cells"] == 5670
    assert manifest["target_cells"] == 2940
    assert manifest["republished_control_cells"] == 630
    assert manifest["comparison_control_cells"] == 2100
    assert manifest["cells_manually_reviewed"] == 5670
    assert manifest["cells_attested"] == 5270
    assert manifest["cells_blank"] == 397
    assert manifest["cells_ambiguous"] == 3
    assert manifest["cells_unreviewed"] == 0
    assert manifest["staged_forms"] == 2900
    assert manifest["installed_forms"] == 0 and manifest["ocr_authority"].startswith("none")
    unresolved = rows(PACKAGE / "unresolved_readings.tsv")
    assert len(unresolved) == 3
    assert {(r["PDF_Page"], r["Printed_Page"], r["Item"], r["Site_Code"], r["Column"]) for r in unresolved} == {
        ("75", "66", "10", "HO3", "left"),
        ("127", "118", "167", "HKE", "left"),
        ("131", "122", "179", "HKA", "left"),
    }
    checklist = (PACKAGE / "CHECKLIST.md").read_text(encoding="utf-8")
    assert "5,670" in checklist and "2,900" in checklist and "OCR-derived chunk was rejected" in checklist


def test_complete_effective_ledger_dry_build_and_symbol_coverage():
    effective = ho.overlay_manual_chunks(ho.validate_base())
    assert ho.require_complete(effective) == Counter(attested=5270, blank=397, ambiguous=3)
    assert ho.role_counts(effective) == {
        "target": Counter(attested=2900, blank=38, ambiguous=2),
        "republished_control": Counter(attested=461, blank=168, ambiguous=1),
        "comparison_control": Counter(attested=1909, blank=191),
    }
    forms, audit = ho.build(effective, ho.validate_registry())
    assert len(forms) == 2900 and len(audit) == 5670
    assert Counter(row["Disposition"] for row in audit) == Counter(
        staged=2900, **{"control-excluded": 2730, "missing": 38, "unresolved-excluded": 2}
    )
    inventory = rows(PACKAGE / "symbol_inventory.tsv")
    inventory_symbols = {row["Symbol"] for row in inventory}
    staged_symbols = {character for row in forms for character in row[2]}
    assert inventory_symbols == staged_symbols and "�" not in staged_symbols
    assert len(rows(PACKAGE / "staged_audit.tsv")) == 5670
    with (PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.reader(stream)) == 2900
