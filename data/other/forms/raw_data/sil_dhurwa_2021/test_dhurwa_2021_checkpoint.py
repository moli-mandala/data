from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_chunks" / "items_001_041_hand_keyed.tsv"
spec = importlib.util.spec_from_file_location("dhurwa_guard", HERE / "import_dhurwa_2021.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(guard)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_checkpoint_is_exhaustive_ocr_blind_and_nfc():
    rows = guard.load_manual_cells()
    assert len(rows) == 41 * 5 == 205
    assert {int(row["Item"]) for row in rows} == set(range(1, 42))
    assert {row["Site_Code"] for row in rows} == {"TIR", "NET", "DHA", "KUK", "U5"}
    assert all(row["Reviewer_Declaration"] == guard.DECLARATION for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert not any("ocr" in field.casefold() for field in fields)


def test_exact_cell_status_and_response_accounting():
    rows = guard.load_manual_cells()
    forms, audit, counts = guard.build_checkpoint(rows)
    assert counts == {
        "reviewed_cells": 205,
        "attested_cells": 203,
        "source_blank_cells": 2,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 204,
        "known_target_cells": 164,
        "known_target_forms": 164,
        "unresolved_identity_cells": 41,
        "unresolved_identity_responses": 40,
    }
    assert len(forms) == 164 and len(audit) == 205
    assert len({row[10] for row in forms}) == 164


def test_source_blanks_and_printed_alternative_are_diplomatic():
    rows = guard.load_manual_cells()
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    for site in ["KUK", "U5"]:
        row = by_key[("21", site)]
        assert row["Review_Status"] == "source_blank"
        assert row["Manual_Transcription"] == ""
        assert row["Uncertainty"] == "source prints double hyphen"
    assert by_key[("4", "KUK")]["Manual_Transcription"] == "bom:a/kʌɳ"
    assert guard.expand_cell(by_key[("4", "KUK")]["Manual_Transcription"]) == ["bom:a", "kʌɳ"]
    assert by_key[("26", "TIR")]["Manual_Transcription"] == "kiɖ kiɖi"
    assert guard.expand_cell(by_key[("26", "TIR")]["Manual_Transcription"]) == ["kiɖ kiɖi"]


def test_coordinates_and_difficult_visual_readings():
    rows = guard.load_manual_cells()
    assert all(row["PDF_Page"] == "17" and row["Printed_Page"] == "12" for row in rows)
    by_key = {(row["Item"], row["Site_Code"]): row["Manual_Transcription"] for row in rows}
    assert by_key[("2", "TIR")] == "ʈɛl"
    assert by_key[("7", "TIR")] == "budʒ:am"
    assert by_key[("8", "TIR")] == "po:ɖo:m"
    assert by_key[("17", "DHA")] == "vʌʈ"
    assert by_key[("31", "TIR")] == "kaɖciɖ"
    assert by_key[("32", "DHA")] == "cɛʈ:al"
    assert by_key[("38", "U5")] == "ʌm:u"
    assert by_key[("40", "U5")] == "nɛliñ"


def test_blank_fifth_header_is_never_staged_or_named():
    registry = guard.load_registry()
    assert registry["U5"]["Printed_Header"] == ""
    assert registry["U5"]["Scope"] == "unresolved_list_identity"
    assert registry["U5"]["Install"] == "no"
    forms, audit, _ = guard.build_checkpoint(guard.load_manual_cells())
    assert all(":U5:" not in row[10] for row in forms)
    u5_audit = [row for row in audit if row["Site_Code"] == "U5"]
    assert len(u5_audit) == 41
    assert all("identity unresolved" in row["Disposition"] for row in u5_audit)


def test_guard_rejects_ocr_bearing_schema(tmp_path: Path):
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    bad = tmp_path / "ocr_bearing.tsv"
    fields = list(rows[0]) + ["OCR_Evidence"]
    with bad.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "inadmissible"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad)


def test_checkpoint_profile_has_complete_coverage():
    profile = guard.load_profile()
    forms, _, _ = guard.build_checkpoint(guard.load_manual_cells())
    converted = [guard.convert(row[2], profile) for row in forms]
    assert len(converted) == 164
    assert "bujːam" in converted
    assert "poːḍoːm" in converted
    assert "kiḍ#kiḍi" in converted
    assert all("�" not in form for form in converted)


def test_manifest_corrects_topology_and_hashes_match():
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    appendix = manifest["lexical_appendix"]
    assert appendix["physical_pdf_pages"] == "17-21"
    assert appendix["prompts"] == 200
    assert appendix["response_columns"] == 5
    assert appendix["conceptual_cells"] == 1000
    assert manifest["manual_review_checkpoint"]["remaining_cells"] == 0
    for name, artifact in manifest["artifacts"].items():
        artifacts = artifact if name == "manual_ledgers" else [artifact]
        for item in artifacts:
            path = HERE / item["path"]
            assert path.exists()
            assert file_sha256(path) == item["sha256"]
