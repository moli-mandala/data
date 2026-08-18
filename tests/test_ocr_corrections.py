import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/ocr_corrections.py"
SPEC = importlib.util.spec_from_file_location("ocr_corrections", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
audit_fingerprint = MODULE.audit_fingerprint
load_corrections = MODULE.load_corrections


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def test_correction_overlay_matches_exact_audit_row(tmp_path):
    audit = tmp_path / "audit.csv"
    corrections = tmp_path / "corrections.csv"
    headers = ["Status", "Entry_Key", "Raw_OCR", "Form", "Gloss"]
    row = {
        "Status": "needs_review",
        "Entry_Key": "sample:p1:e1",
        "Raw_OCR": "ta'",
        "Form": "ta",
        "Gloss": "give",
    }
    write_csv(audit, headers, [[row[field] for field in headers]])
    fingerprint = audit_fingerprint(headers, row)
    write_csv(
        corrections,
        ["Entry_Key", "Status", "Form", "POS", "Gloss", "Notes", "Audit_Fingerprint", "Updated_At"],
        [["sample:p1:e1", "corrected", "ṭā", "verb", "give", "checked scan", fingerprint, "2026-08-17T00:00:00Z"]],
    )

    loaded = load_corrections(corrections, audit)
    assert loaded["sample:p1:e1"].form == "ṭā"
    assert loaded["sample:p1:e1"].status == "corrected"


def test_stale_correction_is_rejected(tmp_path):
    audit = tmp_path / "audit.csv"
    corrections = tmp_path / "corrections.csv"
    write_csv(audit, ["Entry_Key", "Form"], [["sample:p1:e1", "ta"]])
    write_csv(
        corrections,
        ["Entry_Key", "Status", "Form", "POS", "Gloss", "Notes", "Audit_Fingerprint", "Updated_At"],
        [["sample:p1:e1", "accepted", "ta", "", "", "", "obsolete", ""]],
    )

    with pytest.raises(ValueError, match="stale OCR correction"):
        load_corrections(corrections, audit)


def test_lowercase_entry_key_from_cached_ocr_is_supported(tmp_path):
    audit = tmp_path / "entries.csv"
    corrections = tmp_path / "corrections.csv"
    headers = ["entry_key", "form", "raw_entry"]
    row = {"entry_key": "cached-entry-1", "form": "ṭā", "raw_entry": "ta'"}
    write_csv(audit, headers, [[row[field] for field in headers]])
    write_csv(
        corrections,
        ["Entry_Key", "Status", "Form", "POS", "Gloss", "Notes", "Audit_Fingerprint", "Updated_At"],
        [["cached-entry-1", "accepted", "ṭā", "", "", "", audit_fingerprint(headers, row), ""]],
    )

    assert load_corrections(corrections, audit)["cached-entry-1"].form == "ṭā"
