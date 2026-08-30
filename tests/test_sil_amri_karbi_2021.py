"""Focused checks for the SIL JLSR 2021-050 Amri Karbi ingest."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_amri_karbi_2021"
IMPORTER = SOURCE_DIR / "import_amri.py"
INSTALLED = ROOT / "data/other/forms/20260828-sil-amri-karbi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-amri-karbi-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-amri-karbi-manifest.json"
PROFILE = ROOT / "conversion/sil-amri-karbi.txt"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reviewed():
    with (SOURCE_DIR / "reviewed_transcription.tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_source_local_importer_rebuilds_artifacts():
    result = subprocess.run(
        [sys.executable, str(IMPORTER), "--install"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "conceptual_source_cells_manually_reviewed=5219" in result.stdout
    assert "installed_forms=5092" in result.stdout
    assert "unresolved_transcriptions=0" in result.stdout


def test_manifest_pins_source_scope_and_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == "cd121ad102e96b43bf68a1cc5b44f1559c764bc4ae8d71988c6b292a1896ccb1"
    assert manifest["pdf_pages"] == 165
    assert manifest["counts"] == {
        "prompts": 307,
        "printed_lists": 17,
        "published_wordlists_reported": 21,
        "published_lists_absent_from_appendix_b3": 4,
        "conceptual_source_cells_manually_reviewed": 5219,
        "printed_response_occurrences_manually_reviewed": 5960,
        "confirmed_blank_cells": 6,
        "target_printed_response_occurrences": 5329,
        "excluded_control_response_occurrences": 631,
        "duplicate_target_occurrences_audit_only": 237,
        "installed_forms": 5092,
        "installed_amri_karbi_forms": 993,
        "installed_karbi_forms": 4099,
        "audit_rows": 5966,
        "source_marked_uncertain_readings": 1,
        "unresolved_transcriptions": 0,
    }
    assert manifest["review"]["unresolved"] == []
    assert manifest["review"]["ocr"].startswith("not used")


def test_every_source_record_and_cell_has_manual_visual_review():
    rows = reviewed()
    assert len(rows) == 5966
    assert {(row["Item"], row["Site"]) for row in rows}.__len__() == 307 * 17
    assert {int(row["PDF_Page"]) for row in rows} == set(range(37, 116))
    assert Counter(row["Review_Status"] for row in rows) == Counter({
        "complete": 5965, "source-marked-uncertain": 1,
    })
    assert {row["Confidence"] for row in rows} == {"high"}
    assert all("visually reviewed" in row["Review_Note"] or row["Review_Status"] == "source-marked-uncertain" for row in rows)
    assert all(row["Extracted_Form"] == row["Verified_Form"] for row in rows)


def test_blanks_controls_duplicates_and_absent_lists_are_explicit():
    rows = audited()
    assert len(rows) == 5966
    assert Counter(row["Status"] for row in rows) == Counter(installed=5092, excluded=874)
    blanks = [row for row in rows if row["Record_Type"] == "blank"]
    assert {(row["Item"], row["Source_Code"]) for row in blanks} == {
        ("36", "S"), ("37", "b"), ("41", "Z"), ("50", "P"),
        ("53", "Z"), ("127", "C"),
    }
    controls = [row for row in rows if row["Reason"] in {"Khasi control", "Assamese control"}]
    assert Counter(row["Source_Code"] for row in controls) == Counter(C=310, Z=321)
    duplicates = [row for row in rows if row["Reason"].startswith("exact repeated source occurrence")]
    assert len(duplicates) == 237
    # Table 1/B.2 report these lists, but Appendix B.3 publishes none of them.
    assert {row["Source_Code"] for row in rows}.isdisjoint({"e", "f", "g", "w"})


def test_installed_rows_are_stable_diplomatic_and_conservative():
    rows = forms()
    assert len(rows) == 5092
    assert Counter(row["Language_ID"] for row in rows) == Counter(karbi=4099, amri_karbi=993)
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert len({row["Tags"].split(":")[2] for row in rows}) == 15
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Parameter_ID"] == row["Native"] == "" for row in rows)
    assert all(row["Source"].startswith("abraham-daimary2021amrikarbi[Appendix B.3") for row in rows)
    assert all(row["Cognateset"] == row["Etymology"] == "" for row in rows)
    assert all(row["Variant_Of_Key"] == row["Borrowed_From_Key"] == row["Derivation_Parent_Keys"] == "" for row in rows)


def test_source_marked_uncertainty_is_exact_and_excluded():
    row = next(row for row in audited() if row["Review_Status"] == "source-marked-uncertain")
    assert (row["PDF_Page"], row["Printed_Page"], row["Item"], row["Source_Code"]) == ("59", "49", "91", "Z")
    assert row["Verified_Form"] == "soʌ̆ĭ??"
    assert row["Status"] == "excluded" and row["Reason"] == "Assamese control"


def test_source_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in forms():
        converted = tokenizer(row["Form"], column="IPA", segment_separator="", separator="")
        assert "�" not in converted
    assert tokenizer("kʌ̆tʃɾeŋ", column="IPA", segment_separator="", separator="") == "kăcreŋ"
    assert tokenizer("dʒʌ̆ŋ", column="IPA", segment_separator="", separator="") == "jăŋ"


def test_extractor_is_reproducible_and_never_uses_ocr():
    source = (SOURCE_DIR / "extract_amri.py").read_text(encoding="utf-8")
    assert "PdfReader" in source and "PDF_SHA256" in source
    assert "ocr" not in source.lower()
    scaffold = SOURCE_DIR / "extraction_scaffold.tsv"
    with scaffold.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 5966
    assert Counter(row["Record_Type"] for row in rows) == Counter(response=5960, blank=6)
