import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/buddruss_shina_1996.py"
FORMS_PATH = ROOT / "data/other/forms/20260828-buddruss-shina-raetsel.csv"
AUDIT_PATH = ROOT / "data/other/forms/raw_data/20260828-buddruss-shina-raetsel-audit.csv"
MANIFEST_PATH = ROOT / "data/other/forms/raw_data/20260828-buddruss-shina-raetsel-manifest.json"


def load_source():
    spec = importlib.util.spec_from_file_location("buddruss_shina_1996", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dict_rows(path, fields=None):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        if fields:
            return list(csv.DictReader(handle, fieldnames=fields))
        return list(csv.DictReader(handle))


source = load_source()
RAW = source.records()
FORMS = dict_rows(FORMS_PATH, source.FORM_FIELDS)
BY_KEY = {row["Entry_Key"]: row for row in FORMS}


def test_complete_glossary_census_and_stable_keys():
    assert len(RAW) == len(FORMS) == 311
    assert len({(row["page"], row["unit"]) for row in RAW}) == 296
    assert {int(row["page"]) for row in RAW} == set(range(40, 51))
    assert Counter(row["page"] for row in RAW)["50"] == 13
    assert len(BY_KEY) == len(FORMS)
    assert all(key.startswith("buddruss-shina1996:p") for key in BY_KEY)


def test_language_dialect_locators_and_alternates():
    assert {row["Language_ID"] for row in FORMS} == {"Sh"}
    assert all("dialect:Sh:gil:Gilgit" in row["Tags"] for row in FORMS)
    assert all(row["Source"].startswith("buddruss-shina1996[p. ") for row in FORMS)
    assert BY_KEY["buddruss-shina1996:p41:e25:v2"]["Form"] == "čučóoro"
    assert BY_KEY["buddruss-shina1996:p41:e25:v2"]["Variant_Of_Key"] == (
        "buddruss-shina1996:p41:e25"
    )
    assert BY_KEY["buddruss-shina1996:p43:e29"]["Form"] == "hagúl"
    assert BY_KEY["buddruss-shina1996:p43:e29"]["Variant_Of_Key"] == (
        "buddruss-shina1996:p40:e04"
    )
    # Inflected gender examples are retained in prose, not promoted to rows.
    assert "buddruss-shina1996:p45:e04:v2" not in BY_KEY


def test_representative_transcription_and_glosses():
    mouth = BY_KEY["buddruss-shina1996:p40:e07"]
    assert (mouth["Form"], mouth["Parameter_ID"], mouth["Gloss"]) == (
        "áa~i", "1533", "mouth"
    )
    mountain = BY_KEY["buddruss-shina1996:p42:e02"]
    assert mountain["Form"] == "čhii~ṣ"
    assert mountain["Parameter_ID"] == ""
    army = BY_KEY["buddruss-shina1996:p48:e22"]
    assert (army["Form"], army["Parameter_ID"]) == ("síi~", "13587")
    unclear = BY_KEY["buddruss-shina1996:p50:e01"]
    assert unclear["Form"] == "wáaku"
    assert "uncertain" in unclear["Tags"].split()


def test_turner_link_policy_and_ids_exist():
    assert BY_KEY["buddruss-shina1996:p40:e06"]["Parameter_ID"] == ""  # T. 145, 887
    assert BY_KEY["buddruss-shina1996:p41:e12"]["Parameter_ID"] == "2245"  # not 11435
    assert BY_KEY["buddruss-shina1996:p47:e28"]["Parameter_ID"] == "7934"  # T. 7934.2
    assert BY_KEY["buddruss-shina1996:p50:e03"]["Parameter_ID"] == ""  # T. 884, 2207
    assert sum(bool(row["Parameter_ID"]) for row in FORMS) == 149
    cdial_ids = {
        row[1] for row in csv.reader((ROOT / "data/cdial/cdial.csv").open(encoding="utf-8"))
        if len(row) > 1
    }
    assert {row["Parameter_ID"] for row in FORMS if row["Parameter_ID"]} <= cdial_ids


def test_sound_profile_covers_every_form_and_key_contrasts():
    tokenizer = Tokenizer(str(ROOT / "conversion/buddruss-shina.txt"))
    for row in FORMS:
        out = tokenizer(unicodedata.normalize("NFC", row["Form"]),
                        column="IPA", segment_separator="", separator=" ")
        assert "�" not in out and "?" not in out, (row["Form"], out)
    assert tokenizer("aáji", column="IPA", segment_separator="", separator=" ") == "ā́ji"
    assert tokenizer("áa~i", column="IPA", segment_separator="", separator=" ") == "ā̀̃i"
    assert tokenizer("c̣akáai", column="IPA", segment_separator="", separator=" ") == "ʦ̣akā̀i"
    assert tokenizer("tu(r)mák", column="IPA", segment_separator="", separator=" ") == "tu(r)mák"


def test_audit_manifest_rights_and_scope():
    audit = dict_rows(AUDIT_PATH)
    assert len(audit) == 311
    assert all(row["Material_Error"] == "no" for row in audit)
    assert {row["Final_Status"] for row in audit} == {"installed_form"}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == source.PDF_SHA256
    assert manifest["pdf_pages"] == 31
    assert manifest["pdf_redistributed"] is False
    assert manifest["outputs"]["form_count"] == 311
    assert manifest["extraction"]["analytical_headword_units"] == 296
    assert manifest["scope"]["excluded_counts"]["running_riddles"] == 58
    assert manifest["outputs"]["sample_count"] == 25


def test_bibliography_and_compiled_rows_when_available():
    bibliography = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert "@incollection{buddruss-shina1996" in bibliography
    assert source.PDF_SHA256 in bibliography
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists() or forms_path.stat().st_mtime < FORMS_PATH.stat().st_mtime:
        return
    compiled = dict_rows(forms_path)
    installed = [row for row in compiled if "buddruss-shina1996" in row["Source"]]
    assert len(installed) == 311
    assert "�" not in "".join("|".join(row.values()) for row in installed)
