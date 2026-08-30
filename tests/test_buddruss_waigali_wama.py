import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_source(filename, module_name):
    path = ROOT / "data/other/forms/raw_data" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


waigali = load_source("buddruss_waigali_1992.py", "buddruss_waigali_1992")
wama = load_source("buddruss_wama_2006.py", "buddruss_wama_2006")


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def form_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_checked_in_transcriptions_cover_both_complete_glossaries():
    wgi = waigali.records()
    wma = wama.records()
    assert len(wgi) == 158
    assert len(wma) == 276
    assert {int(row["page"]) for row in wgi} == set(range(71, 79))
    assert {int(row["page"]) for row in wma} == set(range(184, 192))
    assert Counter(row["page"] for row in wgi) == {
        "71": 16, "72": 14, "73": 23, "74": 25,
        "75": 21, "76": 21, "77": 17, "78": 21,
    }
    assert Counter(row["page"] for row in wma) == {
        "184": 29, "185": 34, "186": 38, "187": 37,
        "188": 37, "189": 37, "190": 44, "191": 20,
    }


def test_audits_are_complete_and_every_install_has_a_stable_key():
    for source, count, basename in (
        (waigali, 158, "20260824-buddruss-waigali"),
        (wama, 276, "20260824-buddruss-wama"),
    ):
        audit = rows(ROOT / f"data/other/forms/raw_data/{basename}-audit.csv")
        assert len(audit) == count
        assert {row["Final_Status"] for row in audit} == {"installed_form"}
        assert {row["Collation_Date"] for row in audit} == {source.COLLATION_DATE}
        assert all(row["Material_Error"] == "no" for row in audit)
        keys = [row["Emitted_Key"] for row in audit]
        assert len(keys) == len(set(keys)) == count


def test_manifests_record_rights_ocr_scope_and_language_model():
    wm = json.loads((ROOT / "data/other/forms/raw_data/20260824-buddruss-waigali-manifest.json").read_text())
    am = json.loads((ROOT / "data/other/forms/raw_data/20260824-buddruss-wama-manifest.json").read_text())
    assert (wm["pdf_sha256"], wm["pdf_pages"], wm["pdf_redistributed"]) == (
        waigali.PDF_SHA256, 22, False,
    )
    assert (am["pdf_sha256"], am["pdf_pages"], am["pdf_redistributed"]) == (
        wama.PDF_SHA256, 29, False,
    )
    assert wm["outputs"]["form_count"] == 158
    assert am["outputs"]["form_count"] == 276
    assert "Nisheigram" in wm["scope"]["language_model"]
    assert "dialect of canonical Ashkun" in am["scope"]["language_model"]
    assert wm["extraction"]["transcription_uncertainties_remaining"] == 0
    assert am["extraction"]["transcription_uncertainties_remaining"] == 0


def test_source_rows_preserve_notation_dialects_and_conservative_links():
    wgi = form_rows(ROOT / "data/other/forms/20260824-buddruss-waigali.csv")
    wma = form_rows(ROOT / "data/other/forms/20260824-buddruss-wama.csv")
    assert {row[0] for row in wgi} == {"Wg"}
    assert {row[0] for row in wma} == {"Ash"}
    assert all("dialect:Wg:nis:Nisheigram" in row[14] for row in wgi)
    assert all("dialect:Ash:cdial-Ash-wama:Wama" in row[14] for row in wma)
    wgi_by_key = {row[10]: row for row in wgi}
    wma_by_key = {row[10]: row for row in wma}
    assert wgi_by_key["buddruss-waigali1992:p72:e13"][2] == "čipičipun'i"
    assert wgi_by_key["buddruss-waigali1992:p74:e06"][2] == "ǰentab'ār"
    assert wgi_by_key["buddruss-waigali1992:p77:e03"][2] == "šüwal'a"
    assert wma_by_key["buddruss-wama2006:p186:e05"][2] == "cima-karā"
    assert wma_by_key["buddruss-wama2006:p186:e10"][2] == "čital"
    assert wma_by_key["buddruss-wama2006:p191:e16"][2] == "žatə̄rə"
    assert wma_by_key["buddruss-wama2006:p184:e11"][1] == ""
    assert wma_by_key["buddruss-wama2006:p190:e12"][1] == ""
    assert wgi_by_key["buddruss-waigali1992:p71:e05:v2"][11] == "buddruss-waigali1992:p71:e05"


def test_compiled_rows_exist_when_cldf_has_been_built():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists():
        return
    compiled = rows(forms_path)
    for source, count in ((waigali, 158), (wama, 276)):
        installed = [row for row in compiled if source.SOURCE_ID in row["Source"]]
        if not installed or forms_path.stat().st_mtime < source.FORM_OUTPUT.stat().st_mtime:
            continue
        assert len(installed) == count
        assert "�" not in "".join("|".join(row.values()) for row in installed)
