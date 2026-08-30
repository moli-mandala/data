"""Regression tests for the raster-only JLSR 2024-011 Haryanvi wordlists."""

import csv
import io
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_haryanvi_2024"
IMPORTER = SOURCE_DIR / "import_haryanvi.py"
MANUAL = SOURCE_DIR / "manual_transcription.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-haryanvi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-haryanvi-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-haryanvi-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "webster2024haryanvi"
TARGET_SITES = ("HRT", "HJN", "HFT", "HNG", "HTR", "HLH")
sys.path.insert(0, str(ROOT))

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def test_importer_rebuilds_checked_artifacts():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == "installed=1553 source_cells=2100 excluded=862"


def test_manual_ledger_is_complete_cell_by_cell_and_unicode_clean():
    with MANUAL.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 210
    assert [int(row["Item"]) for row in rows] == list(range(1, 211))
    cells = [row[site] for row in rows for site in TARGET_SITES]
    assert len(cells) == 1260
    assert all(cells)
    assert all(row["Review"] == "manual-scan" for row in rows)
    assert sum(cell == "[blank]" for cell in cells) == 21
    assert sum(cell.startswith("[elicitation note:") for cell in cells) == 1
    assert sum("same as item" in cell for cell in cells) == 7
    assert all(
        value == unicodedata.normalize("NFC", value)
        for row in rows for value in row.values()
    )


def test_installed_variants_keys_scope_and_difficult_transcriptions():
    rows = forms()
    assert len(rows) == 1553
    assert Counter(row["Language_ID"] for row in rows) == Counter(kaithal=1553)
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(not row["Parameter_ID"] and not row["Cognateset"] and not row["Etymology"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A.3, printed p. ") for row in rows)
    assert all(row["Tags"].startswith("dialect:kaithal:sil-haryanvi-2024-") for row in rows)
    assert not any("OCR" in row["Notes"] or "Tesseract" in row["Notes"] for row in rows)
    observed = {(row["Entry_Key"], row["Gloss"], row["Form"]) for row in rows}
    assert ("silharyanvi2024:i001:hrt:v01", "body", "d̪eh") in observed
    assert ("silharyanvi2024:i004:hjn:v01", "face", "mu") in observed
    assert ("silharyanvi2024:i052:hft:v02", "stone", "d̪ʌg·ʌl") in observed
    assert ("silharyanvi2024:i209:hft:v01", "you (plural)", "t̪u") in observed
    assert not any("(" in row["Form"] or ")" in row["Form"] for row in rows)


def test_profile_covers_every_manual_form_and_preserves_source_phonemic():
    from make_cldf import parse_file

    errors = io.StringIO()
    rows, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 1553, "for_conversion": 1553}
    assert errors.getvalue() == ""
    assert len(rows) == 1553
    assert all(row.ipa == row.old_form for row in rows)
    assert not any("�" in row.form for row in rows)
    converted = {row.old_form: row.form for row in rows}
    assert converted["d̪eh"] == "deh"
    assert converted["tʃutʃi"] == "cuci"
    assert converted["dʒevli"] == "jevli"
    assert converted["rʌs·a"] == "rasːa"


def test_complete_audit_and_manifest_reconcile_every_source_cell():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2100
    assert Counter(row["Role"] for row in rows) == Counter(target=1260, comparison=840)
    assert Counter(row["Status"] for row in rows) == Counter(installed=1238, excluded=862)
    assert Counter(row["Source_Status"] for row in rows if row["Role"] == "target") == Counter({
        "response": 1231,
        "blank": 21,
        "cross-reference": 7,
        "elicitation-note": 1,
    })
    comparison = [row for row in rows if row["Role"] == "comparison"]
    assert all(row["Status"] == "excluded" and not row["Entry_Keys"] for row in comparison)
    assert all(row["Manual_Review"] == "excluded comparison; not manually transcribed" for row in comparison)
    assert all("non-authoritative OCR evidence" in row["Reason"] for row in comparison)
    assert sum(bool(row["Uncertainty_Type"]) for row in rows if row["Role"] == "target") == 43

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "53121a1b9803ba502092866080e3bdb35457bc6040dcc7f47da508eca1fef2e2"
    )
    assert manifest["source_cells"] == manifest["audit_records"] == 2100
    assert manifest["target_cells"] == manifest["manually_reviewed_target_cells"] == 1260
    assert manifest["installed_responses"] == 1553
    assert manifest["comparison_cells"] == 840
    assert manifest["comparison_cells_manually_transcribed"] == 0
    assert manifest["unparsed_target_cells"] == manifest["replacement_or_private_use_glyphs"] == 0
    assert manifest["ocr_used"] is True
    assert "No installed form originates from OCR" in manifest["ocr_policy"]
    assert manifest["etymology_edges"] == 0


def test_language_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["kaithal"]["Glottocode"] == "hary1238"
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith("sil-haryanvi-2024-")
        ]
    assert len(dialects) == 6
    assert {row["Language_ID"] for row in dialects} == {"kaithal"}
    assert {row["Source_Language_ID"] for row in dialects} == set(TARGET_SITES)
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all("no point coordinate" in row["Location"] for row in dialects)
    assert Counter(row["Quality"] for row in dialects) == Counter(C=4, A=1, B=1)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    assert "No installed form originates from OCR" in bib


def test_compiled_rows_preserve_source_locators_and_six_dialects_after_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [row for row in csv.DictReader(stream) if SOURCE_KEY in row["Source"]]
    # The user requested one consolidated build after all parallel surveys finish.
    if not compiled:
        return
    installed = forms()
    compiled_citations = {
        source
        for row in compiled
        for source in row["Source"].split(";")
        if source.startswith(f"{SOURCE_KEY}[")
    }
    assert compiled_citations == {row["Source"] for row in installed}
    compiled_dialects = {
        tag.split(":", 3)[2]
        for row in compiled for tag in row["Tags"].split()
        if tag.startswith("dialect:") and "sil-haryanvi-2024-" in tag
    }
    assert len(compiled) == len(installed) == 1553
    assert compiled_dialects == {f"sil-haryanvi-2024-{site.lower()}" for site in TARGET_SITES}
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
