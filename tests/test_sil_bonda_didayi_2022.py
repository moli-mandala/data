"""Focused checks for SIL JLSR 2022-004 Bonda/Didayi Appendix B."""

import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_bonda_didayi_2022"
INSTALLED = ROOT / "data/other/forms/20260828-sil-bonda-didayi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-bonda-didayi-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-bonda-didayi-manifest.json"
PROFILE = ROOT / "conversion/sil-bonda-didayi.txt"

sys.path.insert(0, str(PACKAGE))
import import_bonda_didayi as source  # noqa: E402


def installed_rows():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audit_rows():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_exact_cell_topology_review_and_nfc():
    rows = source.read_cells()
    assert len(rows) == 2730
    assert Counter(row["Site_Code"] for row in rows) == {code: 210 for code in source.SITES}
    assert all(unicodedata.normalize("NFC", row["Raw_Response"]) == row["Raw_Response"] for row in rows)
    assert Counter(row["Extraction_Status"] for row in rows) == {
        "text-layer": 2676, "disqualified": 52,
        "source-omitted": 1, "manual-visual-correction": 1,
    }
    assert not any("�" in row["Raw_Response"] for row in rows)


def test_installed_counts_shape_keys_and_source():
    rows = installed_rows()
    assert len(rows) == 1938
    assert {len(row) for row in rows} == {15}
    assert Counter(row[0] for row in rows) == {"gt": 1091, "re": 847}
    assert all(row[2] and row[2] == row[5] for row in rows)
    assert all(row[7].startswith("mathew-chamberlain2022bonda-didayi[Appendix B,") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(row[8] == row[9] == "" for row in rows)


def test_rebuild_is_row_equivalent():
    rebuilt, rebuilt_audit, _ = source.build()
    assert rebuilt == installed_rows()
    assert rebuilt_audit == audit_rows()


def test_every_cell_dispositioned_and_controls_excluded():
    rows = audit_rows()
    assert len(rows) == 2730
    assert Counter(row["Status"] for row in rows) == {
        "installed": 1836, "excluded": 840, "disqualified": 36, "missing": 18,
    }
    assert all(row["Manual_Review"] == "visual-source-page" for row in rows)
    controls = [row for row in rows if row["Site_Code"] not in source.TARGETS]
    assert len(controls) == 840 and not any(row["Entry_Keys"] for row in controls)
    assert all(row["Status"] == "excluded" for row in controls)


def test_missing_disqualified_and_omitted_cell_are_explicit():
    rows = audit_rows()
    disqualified = [row for row in rows if row["Status"] == "disqualified"]
    assert len(disqualified) == 36
    assert {int(row["Item"]) for row in disqualified} == {11, 23, 24, 70}
    omitted = [row for row in rows if row["Uncertainty"]]
    assert len(omitted) == 1
    assert (omitted[0]["PDF_Page"], omitted[0]["Item"], omitted[0]["Site_Code"]) == ("45", "174", "ORA")
    assert omitted[0]["Status"] == "missing" and not omitted[0]["Entry_Keys"]


def test_similarity_groups_are_notes_not_cognacy():
    rows = installed_rows()
    assert all("source similarity group" in row[6] for row in rows)
    assert all(not row[8] and not row[9] for row in rows)
    by_key = {row[10]: row for row in rows}
    assert by_key["silbondadidayi1997:chi:i050:v1"][2] == "bɾihumhaiʒã"
    assert by_key["silbondadidayi1997:bia:i190:v2"][2] == "mebike"
    assert by_key["silbondadidayi1997:ori:i180:v2"][2] == "anɖua"


def test_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed_rows():
        assert "�" not in tokenizer(row[2], column="IPA"), (row[10], row[2])


def test_manifest_pins_source_and_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == "bb0548b4324224260b9618786dfd3aa40377138d0fbf4ae14c796df82f6190ce"
    assert manifest["extracted_cells_sha256"] == hashlib.sha256((PACKAGE / "extracted_cells.tsv").read_bytes()).hexdigest()
    assert manifest["counts"]["conceptual_cells_visually_reviewed"] == 2730
    assert manifest["counts"]["target_cells_visually_reviewed"] == 1890
    assert manifest["counts"]["comparison_cells_visually_reviewed"] == 840
    assert manifest["counts"]["installed_rows"] == 1938
    assert manifest["policy"]["ocr"].startswith("not used")
