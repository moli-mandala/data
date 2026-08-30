"""Focused checks for the manually transcribed SIL Kullu survey wordlists."""

import csv
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_kullu_2021"
INSTALLED = ROOT / "data/other/forms/20260828-sil-kullu.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-kullu-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-kullu-manifest.json"
PROFILE = ROOT / "conversion/sil-kullu.txt"
SOURCE_KEY = "blair2021kullu"

sys.path.insert(0, str(PACKAGE))
import import_kullu  # noqa: E402

TARGET_SITES = set(import_kullu.SITES)


def installed_rows():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audit_rows():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def manual_rows():
    with (PACKAGE / "manual_pages.tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_manual_ledger_exact_topology_nfc_and_explicit_blanks():
    rows = manual_rows()
    assert len(rows) == 208
    assert all(row["Review"] == "manual-source-image" for row in rows)
    assert {row["Site"] for row in rows} == set(import_kullu.SITES)
    cells = blanks = 0
    by_site = {code: [] for code in import_kullu.SITES}
    for row in rows:
        first = int(row["First_Item"])
        forms = json.loads(row["Forms_JSON"])
        uncertainty = json.loads(row["Uncertainty_JSON"])
        expected = 6 if first == 193 else 16
        assert len(forms) == expected
        assert len(uncertainty) in (0, expected)
        for offset, form in enumerate(forms):
            cells += 1
            by_site[row["Site"]].append(first + offset)
            assert form == unicodedata.normalize("NFC", form)
            if not form:
                blanks += 1
                assert uncertainty[offset] == "blank"
    assert cells == 3168 and blanks == 415
    assert all(items == list(range(1, 199)) for items in by_site.values())


def test_installed_counts_shape_keys_and_no_claimed_cognacy():
    rows = installed_rows()
    assert len(rows) == 2963
    assert {len(row) for row in rows} == {15}
    assert all(row[0] == "kul" and row[1] == "" for row in rows)
    assert all(row[2] and row[2] == row[5] for row in rows)
    assert all(row[8] == row[9] == "" for row in rows)
    assert all(row[10].startswith("silkullu1985:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(unicodedata.normalize("NFC", row[2]) == row[2] for row in rows)


def test_importer_rebuild_is_row_equivalent():
    forms, audit, manifest = import_kullu.build()
    assert forms == installed_rows()
    assert [{k: str(v) for k, v in row.items()} for row in audit] == audit_rows()
    assert manifest["counts"] == json.loads(MANIFEST.read_text())["counts"]


def test_audit_accounts_for_every_cell_and_hindi_layout_label():
    rows = audit_rows()
    assert len(rows) == 3169
    lexical = [row for row in rows if row["Record_Type"] == "wordlist cell"]
    assert len(lexical) == 3168
    assert Counter(row["Site_Code"] for row in lexical) == {code: 198 for code in import_kullu.SITES}
    assert Counter(row["Status"] for row in rows) == {"installed": 2753, "missing": 415, "excluded": 1}
    assert all(row["Manual_Review"] == "manual-source-image" for row in rows)
    layout = [row for row in rows if row["Record_Type"] == "layout-header"]
    assert len(layout) == 1 and layout[0]["Source_Dialect_Label"] == "Hindi"
    assert "no Hindi lexical response column" in layout[0]["Reason"]


def test_ocr_is_only_evidence_and_blanks_uncertainties_are_explicit():
    audit = audit_rows()
    installed = installed_rows()
    keys = {key for row in audit if row["Status"] == "installed" for key in row["Entry_Keys"].split("|")}
    assert keys == {row[10] for row in installed}
    assert all("manual source-image transcription; OCR comparison only" in row[6] for row in installed)
    assert all(not row["Entry_Keys"] for row in audit if row["Status"] != "installed")
    assert all(row["Uncertainty"] == "blank" for row in audit if row["Status"] == "missing")
    assert Counter(row["Uncertainty"] for row in audit if row["Uncertainty"]) == {
        "blank": 415, "source-question-mark": 3,
        "ambiguous-faint-reading": 1, "ambiguous-final-word": 1,
    }


def test_slash_alternatives_are_split_and_linked():
    rows = installed_rows()
    assert not any("/" in row[2] for row in rows)
    by_key = {row[10]: row for row in rows}
    first = by_key["silkullu1985:jib:i193:v1"]
    second = by_key["silkullu1985:jib:i193:v2"]
    assert first[2] == "hã" and second[2] == "oː"
    assert second[11] == first[10]


def test_representative_sites_forms_and_dialect_tags():
    rows = installed_rows()
    by_key = {row[10]: row for row in rows}
    assert by_key["silkullu1985:chu:i001:v1"][2] == "dʒɪsəm"
    assert by_key["silkullu1985:ani:i198:v1"][2] == "t̪eɽə nʊ kidʒi əsə"
    assert len({row[14].split()[0] for row in rows}) == 16
    assert all(row[7].startswith("blair2021kullu[Appendix C") for row in rows)


def test_profile_covers_all_installed_forms():
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed_rows():
        assert "�" not in tokenizer(row[2], column="IPA")


def test_shared_profile_routing_and_metadata_registration():
    sys.path.insert(0, str(ROOT))
    from make_cldf import parse_file

    errors = io.StringIO()
    parsed, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 2963, "for_conversion": 2963}
    assert errors.getvalue() == ""
    assert len(parsed) == 2963
    assert all(row.ipa == row.old_form and "�" not in row.form for row in parsed)

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith("sil-kullu-1985-")
        ]
    assert len(dialects) == 16
    assert {row["Language_ID"] for row in dialects} == {"kul"}
    assert {row["Source_Language_ID"] for row in dialects} == TARGET_SITES
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all(row["Quality"] == "C" for row in dialects)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    assert "No installed form originates from OCR" in bib.split(
        f"@techreport{{{SOURCE_KEY},", 1
    )[1].split("\n}", 1)[0]


def test_manifest_pins_source_and_manual_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == import_kullu.SOURCE_SHA256
    assert manifest["source_pdf_pages"] == 126
    assert manifest["counts"]["source_cells"] == 3168
    assert manifest["counts"]["source_image_cells_manually_reviewed"] == 3168
    assert manifest["counts"]["missing_blank_cells"] == 415
    assert manifest["policy"]["ocr"].startswith("comparison scaffold only")
