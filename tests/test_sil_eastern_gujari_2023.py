"""Focused checks for the JLSR 2023-002 Eastern Gujari ingest."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_eastern_gujari_2023"
IMPORTER = SOURCE_DIR / "import_eastern_gujari.py"
INSTALLED = ROOT / "data/other/forms/20260828-sil-eastern-gujari.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-eastern-gujari-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-eastern-gujari-manifest.json"
PROFILE = ROOT / "conversion/sil-eastern-gujari.txt"
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
    assert "conceptual_source_cells_manually_reviewed=3150" in result.stdout
    assert "installed_forms=1753" in result.stdout
    assert "unresolved_transcriptions=0" in result.stdout


def test_manifest_pins_source_scope_and_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == "41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e"
    assert manifest["pdf_pages"] == 121
    assert manifest["counts"] == {
        "prompts": 210, "printed_lists": 15,
        "conceptual_source_cells_manually_reviewed": 3150,
        "target_conceptual_cells": 1680,
        "republished_ssnp_conceptual_cells": 1260,
        "urdu_control_conceptual_cells": 210,
        "attested_cells": 3117, "confirmed_blank_cells": 33,
        "confirmed_target_blank_cells": 25,
        "confirmed_non_target_blank_cells": 8,
        "excluded_attested_republished_ssnp_cells": 1254,
        "excluded_attested_urdu_control_cells": 208,
        "target_attested_cells": 1655,
        "target_printed_alternative_occurrences": 1754,
        "duplicate_target_alternatives_audit_only": 1,
        "installed_forms": 1753, "audit_rows": 3150,
        "ambiguous_or_illegible_cells": 0, "unresolved_transcriptions": 0,
    }
    assert manifest["review"]["unresolved"] == []
    assert manifest["review"]["image_only_or_handwritten_cells"] == 0


def test_every_cell_has_complete_visual_review():
    rows = reviewed()
    assert len(rows) == 210 * 15
    assert {(row["Item"], row["List"]) for row in rows}.__len__() == 3150
    assert {int(row["PDF_Page"]) for row in rows} == set(range(42, 77))
    assert Counter(row["Review_Status"] for row in rows) == Counter(complete=3150)
    assert {row["Confidence"] for row in rows} == {"high"}
    assert all("visually reviewed" in row["Review_Note"] for row in rows)
    assert all(row["Source_Cell"] == row["Verified_Cell"] for row in rows)


def test_controls_reprints_blanks_and_duplicate_are_explicit():
    rows = audited()
    assert len(rows) == 3150
    assert Counter(row["Role"] for row in rows) == Counter({
        "new Indian target": 1680, "republished SSNP list": 1260,
        "Urdu control": 210,
    })
    assert Counter(row["Record_Type"] for row in rows) == Counter(response=3117, blank=33)
    assert Counter(row["Existing_SSNP_Dialect"] for row in rows if row["Role"] == "republished SSNP list") == Counter({
        "SSNP-gojri-CHT": 210, "SSNP-gojri-SSW": 210,
        "SSNP-gojri-GLT": 210, "SSNP-gojri-KGH": 210,
        "SSNP-gojri-NAK": 210, "SSNP-gojri-CAK": 210,
    })
    assert all(row["Status"] == "excluded" for row in rows if row["Role"] != "new Indian target")
    duplicate = next(row for row in rows if row["Reason"] == "one exact repeated alternative installed once")
    assert (duplicate["PDF_Page"], duplicate["Item"], duplicate["Source_Code"]) == ("63", "127", "JAM")
    assert duplicate["Source_Cell"] == "1  sɑl / 3  bʌɾo / 4  bʌɾo"
    assert duplicate["Installed_Count"] == "2"


def test_reprinted_lists_reconcile_to_primary_ssnp_install():
    with (ROOT / "data/other/forms/20260725-ssnp.csv").open(encoding="utf-8", newline="") as stream:
        primary_counts = Counter(row[0] for row in csv.reader(stream))
    assert {dialect: primary_counts[dialect] for dialect in [
        "SSNP-gojri-CHT", "SSNP-gojri-SSW", "SSNP-gojri-GLT",
        "SSNP-gojri-KGH", "SSNP-gojri-NAK", "SSNP-gojri-CAK",
    ]} == {
        "SSNP-gojri-CHT": 208, "SSNP-gojri-SSW": 210,
        "SSNP-gojri-GLT": 209, "SSNP-gojri-KGH": 209,
        "SSNP-gojri-NAK": 210, "SSNP-gojri-CAK": 208,
    }


def test_installed_rows_are_diplomatic_and_conservative():
    rows = forms()
    assert len(rows) == 1753
    assert Counter(row["Language_ID"] for row in rows) == Counter(Goj=1753)
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert len({row["Tags"].split(":")[2] for row in rows}) == 8
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Source"].startswith("hugoniot-polster-ahmad-rajan2023easterngujari[Appendix B") for row in rows)
    assert all(row["Cognateset"] == row["Etymology"] == "" for row in rows)
    assert all(row["Variant_Of_Key"] == row["Borrowed_From_Key"] == row["Derivation_Parent_Keys"] == "" for row in rows)
    representative = {(row["Entry_Key"], row["Form"]) for row in rows}
    assert ("sileasterngujari2023:p042:i001:HAL:a2", "dʒɪsʌm") in representative
    assert ("sileasterngujari2023:p071:i176:NAL:a3", "fʌɾək") in representative
    assert ("sileasterngujari2023:p076:i210:DEH:a1", "oʋe") in representative


def test_source_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in forms():
        converted = tokenizer(row["Form"], column="IPA", segment_separator="", separator="")
        assert "�" not in converted
    assert tokenizer("dʒɪsʌm", column="IPA", segment_separator="", separator="") == "jisam"
    assert tokenizer("kʌnɑɾe kʌnɑɾe", column="IPA", segment_separator="", separator="") == "kanārekanāre"


def test_extractor_is_reproducible_and_text_layer_is_scaffold_only():
    source = (SOURCE_DIR / "extract_eastern_gujari.py").read_text(encoding="utf-8")
    assert "PdfReader" in source and "PDF_SHA256" in source
    assert "ocr" not in source.lower()
    with (SOURCE_DIR / "extraction_scaffold.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 3150
    assert Counter(row["Record_Type"] for row in rows) == Counter(response=3117, blank=33)
    wrapped = next(row for row in rows if row["Item"] == "176" and row["List"] == "Nalagarh/H.P.")
    assert wrapped["Source_Cell"] == "2  eklo / 3 kʌnɑɾe kʌnɑɾe / 10 fʌɾək"
    assert "wraps" in wrapped["Extraction_Note"]
