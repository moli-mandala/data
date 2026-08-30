"""Focused checks for the image-only SIL JLSR 2022-015 Bagheli ingest."""

import csv
import io
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_bagheli_2022"
IMPORTER = SOURCE_DIR / "import_bagheli.py"
INSTALLED = ROOT / "data/other/forms/20260828-sil-bagheli.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-bagheli-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-bagheli-manifest.json"
PROFILE = ROOT / "conversion/sil-bagheli.txt"
SOURCE_KEY = "koshy2022bagheli"
TARGET_CODES = {"D", "K", "P", "S", "a", "b", "c", "d", "e", "j", "k", "l", "m", "n", "p", "r", "s", "t"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_importer_rebuilds_the_checked_source_local_artifacts():
    result = subprocess.run(
        [sys.executable, str(IMPORTER), "--install"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "expanded_response_cells=6111 nonlexical_response_cells=24" in result.stdout
    assert "installed=5828 audit=6184 controls=283 blanks=47" in result.stdout


def test_manifest_pins_source_and_manual_review_denominator():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == (
        "d1424f317dc12fe01d99d33abd917201575487f4de44529678ecce1c282a4627"
    )
    assert manifest["counts"] == {
        "audit_rows_including_alternatives_and_unassigned": 6184,
        "conceptual_attested_cells": 3933,
        "conceptual_attested_control_cells": 208,
        "conceptual_attested_target_cells": 3725,
        "conceptual_nonlexical_only_cells": 10,
        "conceptual_source_cells_reviewed": 3990,
        "confirmed_blank_cells": 47,
        "confirmed_blank_control_cells": 2,
        "confirmed_blank_target_cells": 45,
        "excluded_hindi_control_forms": 283,
        "expanded_assigned_response_cells": 6111,
        "expanded_nonlexical_response_cells": 24,
        "installed_bagheli_forms": 5828,
        "interpreted_site_code_cells": 1,
        "lists": 19,
        "manual_nonlexical_directives": 2,
        "manual_response_lines": 2284,
        "manual_unassigned_response_lines": 2,
        "prompts": 210,
        "source_marked_uncertain_installed_forms": 1,
        "standard_hindi_controls": 1,
        "target_bagheli_lists": 18,
        "unresolved_unassigned_response_lines": 2,
    }
    assert "never feeds installation" in manifest["transcription"]["ocr"]


def test_installed_rows_have_stable_scope_and_diplomatic_ipa():
    rows = forms()
    assert len(rows) == 5828
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"bagheli_lakshman"}
    assert len({row["Tags"].split(":")[2] for row in rows}) == 18
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Parameter_ID"] == row["Native"] == "" for row in rows)
    assert all(row["Source"].startswith("koshy2022bagheli[Appendix B.4") for row in rows)
    assert all(row["Cognateset"] == row["Etymology"] == "" for row in rows)


def test_audit_accounts_for_every_cell_blank_control_and_unresolved_line():
    rows = audited()
    assert len(rows) == 6184
    assert Counter(row["Status"] for row in rows) == Counter(installed=5828, excluded=356)
    cells = {(int(row["Item"]), row["Site_Code"]) for row in rows if row["Site_Code"]}
    assert len(cells) == 210 * 19
    assert {item for item, _ in cells} == set(range(1, 211))
    formless = [row for row in rows if not row["Manual_Form"]]
    assert len(formless) == 71
    assert Counter(row["Reason"] for row in formless) == Counter({
        "prompt absent from the printed Appendix B.4 table": 38,
        "source prints “by name”; no lexical response is supplied": 24,
        "no response printed for this site/item": 9,
    })
    blanks = [row for row in formless if not row["Reason"].startswith("source prints “by name”")]
    assert len(blanks) == 47
    absent_prompts = {(row["Item"], row["Site_Code"]) for row in blanks if row["Item"] in {"23", "24"}}
    assert len(absent_prompts) == 38
    unresolved = [row for row in rows if row["Review_Status"] == "unresolved"]
    assert [(row["Item"], row["Manual_Form"]) for row in unresolved] == [
        ("191", "berəʈɛ"), ("195", "reŋeʈe"),
    ]
    assert all(row["Site_Code"] == "" and row["Status"] == "excluded" for row in unresolved)


def test_manual_review_confidence_and_source_uncertainty_are_explicit():
    rows = audited()
    assert Counter(row["Confidence"] for row in rows) == Counter({
        "high": 6180,
        "high transcription / unresolved assignment": 2,
        "medium (site-code case interpreted)": 1,
        "high (source uncertainty retained)": 1,
    })
    uncertain = next(row for row in rows if row["Review_Status"] == "source-marked-uncertain")
    assert (uncertain["Item"], uncertain["Site_Code"], uncertain["Manual_Form"]) == (
        "189", "a", "bejtʰe",
    )
    case = next(row for row in rows if row["Confidence"].startswith("medium"))
    assert (case["Item"], case["Site_Code"], case["Manual_Form"]) == ("121", "l", "ʃam")
    assert all("manual" in row["Review_Method"] for row in rows)
    assert all("OCR comparison only" in row["Review_Method"] for row in rows)


def test_representative_manual_transcriptions_and_qualifiers_survive():
    rows = audited()
    expected = {
        ("1", "D", "ɖeh"),
        ("16", "n", "eŭʈʰi"),
        ("73", "h", "bēgen"),
        ("104", "P", "beɕja"),
        ("189", "l", "bet"),
        ("194", "D", "uɖəʈʰ he"),
        ("210", "e", "ū patʃe"),
    }
    actual = {(row["Item"], row["Site_Code"], row["Manual_Form"]) for row in rows}
    assert expected <= actual
    assert next(row for row in rows if (row["Item"], row["Site_Code"], row["Manual_Form"]) == ("189", "l", "bet"))["Qualifier"] == "source footnote marker 1 follows e"
    comma = next(row for row in rows if row["Item"] == "173" and row["Site_Code"] == "m")
    assert comma["Manual_Form"] == "je,e"
    assert comma["Review_Status"] == "complete"


def test_source_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in forms():
        converted = tokenizer(
            unicodedata.normalize("NFC", row["Form"]), column="IPA",
            segment_separator="", separator="",
        )
        assert "�" not in converted
    assert tokenizer("beɕja", column="IPA", segment_separator="", separator="") == "beśya"
    assert tokenizer("pā̃tʃʰ", column="IPA", segment_separator="", separator="") == "pā̃ch"


def test_shared_profile_routing_and_metadata_registration():
    sys.path.insert(0, str(ROOT))
    from make_cldf import parse_file

    errors = io.StringIO()
    parsed, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 5829, "for_conversion": 5829}
    assert errors.getvalue() == ""
    assert len(parsed) == 5829
    assert all("�" not in row.form for row in parsed)
    item_173_m = [
        row for row in parsed
        if row.gloss == "these" and "item 173, site m" in row.source
    ]
    assert [(row.old_form, row.form) for row in item_173_m] == [("je", "ye"), ("e", "e")]
    assert {row.ipa for row in item_173_m} == {"je,e"}
    assert all(row.ipa == row.old_form for row in parsed if row not in item_173_m)

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith("sil-bagheli-2022-")
        ]
    assert len(dialects) == 18
    assert {row["Language_ID"] for row in dialects} == {"bagheli_lakshman"}
    assert {row["Source_Language_ID"] for row in dialects} == TARGET_CODES
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all(row["Quality"] == "C" for row in dialects)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    assert "No installed form originates from OCR" in bib.split(
        f"@techreport{{{SOURCE_KEY},", 1
    )[1].split("\n}", 1)[0]


def test_ocr_scaffold_is_evidence_not_an_import_input():
    source = IMPORTER.read_text(encoding="utf-8")
    assert "tesseract_scaffold.txt" in source
    assert "OCR.read_text" not in source
    assert "manual_transcription.txt" in source


def test_image_and_ocr_scaffolds_cover_every_appendix_column():
    with (SOURCE_DIR / "image_manifest.tsv").open(encoding="utf-8", newline="") as stream:
        images = list(csv.DictReader(stream, delimiter="\t"))
    assert len(images) == 23
    assert {int(row["PDF_Page"]) for row in images} == set(range(59, 82))
    scaffold = (SOURCE_DIR / "tesseract_scaffold.txt").read_text(encoding="utf-8")
    assert scaffold.count("=== PDF_PAGE ") == 23 * 4
