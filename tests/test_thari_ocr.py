import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/thari.py"
AUDIT = ROOT / "data/other/forms/raw_data/20260817-thari-audit.csv"
CALIBRATION = ROOT / "data/other/forms/raw_data/20260817-thari-calibration.csv"


def load_module():
    raw_dir = str(SCRIPT.parent)
    if raw_dir not in sys.path:
        sys.path.insert(0, raw_dir)
    spec = importlib.util.spec_from_file_location("thari_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_thari_ocr_cleanup():
    module = load_module()
    assert module.split_form_pos("g&d(n.ffi)") == ("gad", "noun")
    assert module.split_form_pos("sfebhr&n(tr)") == ("sfebhran", "verb tr")
    assert module.clean_gloss("to hear ,  to listen") == "to hear, to listen"


def test_unreviewed_thari_ocr_cannot_be_installed(tmp_path):
    module = load_module()
    try:
        module.write_import(tmp_path / "unsafe.csv", [])
    except RuntimeError as error:
        assert "failed calibration" in str(error)
    else:
        raise AssertionError("unreviewed OCR was accepted as lexical data")


def test_reviewed_thari_correction_can_be_installed(tmp_path):
    module = load_module()
    entry = module.Entry(
        pdf_page=15,
        printed_page=200,
        column=1,
        column_entry=10,
        top=420.0,
        raw_form_pos="kiietri (n.f)",
        raw_gloss="field",
        form="knetri",
        pos="noun feminine",
        gloss="field",
    )
    correction = module.OcrCorrection(
        entry_key=entry.key,
        status="corrected",
        form="khetrī",
        pos="noun feminine",
        gloss="field",
        notes="checked against scan",
        audit_fingerprint="abc",
        updated_at="",
    )
    output = tmp_path / "reviewed.csv"
    assert module.write_import(output, [entry], {entry.key: correction}) == 1
    row = next(csv.reader(output.open(encoding="utf-8")))
    assert row[2] == "khetrī"
    assert row[3] == "field"
    assert row[10] == entry.key


def test_thari_audit_covers_both_pdf_copies():
    rows = list(csv.DictReader(AUDIT.open(encoding="utf-8")))
    assert len(rows) == 3305
    assert {row["Status"] for row in rows} == {
        "already_reviewed",
        "prior_section",
        "needs_review",
    }
    assert sum(bool(row["Duplicate_Raw_Form_POS"]) for row in rows) >= 1500


def test_thari_calibration_is_a_true_page_holdout():
    rows = list(csv.DictReader(CALIBRATION.open(encoding="utf-8")))
    assert len(rows) == 18
    assert {row["Printed_Page"] for row in rows} == {"200"}
    assert all(row["Gold_Form"] for row in rows)
    assert all(row["Embedded_ABBYY"] for row in rows)

    scored = [row for row in rows if row["Tuned_Base_OCR"]]
    assert len(scored) == 14
    assert sum(row["Tuned_Base_Exact"] == "yes" for row in scored) == 10

    kraken_scored = [row for row in rows if row["Kraken_Unicode_OCR"]]
    assert len(kraken_scored) == 14
    assert sum(row["Kraken_Unicode_Exact"] == "yes" for row in kraken_scored) == 7
