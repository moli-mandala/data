"""Focused checks for the manually transcribed SIL Korku survey appendix."""

import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_korku_2021"
INSTALLED = ROOT / "data/other/forms/20260828-sil-korku.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-korku-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-korku-manifest.json"
PROFILE = ROOT / "conversion/sil-korku.txt"
TARGET_SITES = {"CHI", "KHA", "BAG", "WAR", "MOR", "LAH", "AMD", "KHM"}

sys.path.insert(0, str(PACKAGE))
import import_korku  # noqa: E402

ROWS = import_korku.ROWS


def installed_rows():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audit_rows():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_manual_ledger_has_exact_source_topology_and_nfc():
    assert len(ROWS) == 99
    assert sum(len(row["Forms"]) for row in ROWS) == 1890
    assert all(row["Review"] == "manual-source-image" for row in ROWS)
    assert {(row["PDF_Page"], row["Site"]) for row in ROWS} == set(
        import_korku.expected_page_item_pairs()
    )
    for row in ROWS:
        assert len(row["Forms"]) == len(row["Uncertainties"])
        for form, uncertainty in zip(row["Forms"], row["Uncertainties"]):
            assert unicodedata.normalize("NFC", form) == form
            assert form or uncertainty


def test_installed_shape_counts_keys_and_source_identity():
    rows = installed_rows()
    assert len(rows) == 1521
    assert {len(row) for row in rows} == {15}
    assert all(row[0] == "ko" and row[1] == "" for row in rows)
    assert all(row[2] and row[2] == row[5] for row in rows)
    assert all(row[8] == row[9] == "" for row in rows)
    assert all(row[7].startswith("stahl2021korku[Appendix F,") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)


def test_rebuild_is_row_equivalent():
    rebuilt, rebuilt_audit, _ = import_korku.build()
    assert rebuilt == installed_rows()
    assert [{key: str(value) for key, value in row.items()} for row in rebuilt_audit] == audit_rows()


def test_every_cell_is_manually_reviewed_and_dispositioned():
    rows = audit_rows()
    assert len(rows) == 1890
    assert Counter(row["Site_Code"] for row in rows) == {code: 210 for code in import_korku.SITES}
    assert Counter(row["Status"] for row in rows) == {
        "installed": 1463, "missing": 217, "excluded": 210,
    }
    assert all(row["Manual_Review"] == "manual-source-image" for row in rows)
    assert all(row["OCR_Evidence"].startswith("tesseract_raw.txt#pdf") for row in rows)
    controls = [row for row in rows if row["Site_Code"] == "NIH"]
    assert len(controls) == 210 and not any(row["Entry_Keys"] for row in controls)
    assert all(row["Reason"] == "excluded comparison control" for row in controls)


def test_blanks_and_the_single_unresolved_cell_are_explicit():
    rows = audit_rows()
    target_blanks = [
        row for row in rows
        if row["Site_Code"] in TARGET_SITES and row["Uncertainty"] == "confirmed ruled blank"
    ]
    assert len(target_blanks) == 216
    unresolved = [row for row in rows if row["Status"] == "missing" and "illegible" in row["Reason"]]
    assert len(unresolved) == 1
    assert (unresolved[0]["PDF_Page"], unresolved[0]["Item"], unresolved[0]["Site_Code"]) == ("83", "93", "AMD")
    assert unresolved[0]["Manual_Transcription"] == ""


def test_no_ocr_only_control_or_unresolved_record_is_installed():
    audit = audit_rows()
    keys = {key for row in audit if row["Status"] == "installed" for key in row["Entry_Keys"].split("|")}
    installed = installed_rows()
    assert keys == {row[10] for row in installed}
    assert all("manual source-image transcription" in row[6] for row in installed)
    assert not any("OCR" in row[6] for row in installed)
    assert not any(":nih:" in row[10] for row in installed)


def test_slash_alternatives_split_and_representative_forms():
    rows = installed_rows()
    by_key = {row[10]: row for row in rows}
    assert by_key["silkorku1985:lah:i002:v1"][2] == "dẽi"
    assert by_key["silkorku1985:lah:i002:v2"][2] == "kapar"
    assert by_key["silkorku1985:amd:i001:v1"][2] == "kompɛr"
    assert by_key["silkorku1985:khm:i018:v1"][2] == "dʒaɖa"
    assert not any("/" in row[2] for row in rows)


def test_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed_rows():
        assert "�" not in tokenizer(row[2], column="IPA")


def test_manifest_pins_source_ocr_and_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == "d17426da3788d66c95f05824483941e7d5468e154c66d43c6354262fda00190d"
    assert manifest["source_pdf_pages"] == 102
    assert manifest["counts"]["source_image_cells_manually_reviewed"] == 1890
    assert manifest["counts"]["target_cells_manually_reviewed"] == 1680
    assert manifest["counts"]["comparison_cells_manually_reviewed"] == 210
    assert manifest["counts"]["installed_rows"] == 1521
    assert manifest["ocr_scaffold_sha256"] == hashlib.sha256(
        (PACKAGE / "tesseract_raw.txt").read_bytes()
    ).hexdigest()
    assert manifest["policy"]["ocr"].startswith("comparison scaffold only")
