import csv
import hashlib
import importlib.util
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_adi_2015"
LEDGER = PACKAGE / "manual_chunks/items_001_012_hand_keyed.tsv"
AUDITOR = PACKAGE / "preintegration_audit.py"
PDF = ROOT.parent / "tmp/pdfs/adi_2015/silesr2015_016.pdf"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard", PACKAGE / "import_adi_2015.py")


def test_complete_visually_reviewed_first_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 12 * 9 == 108
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 108
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())


def test_accounting_and_form_expansion():
    rows = guard.load_manual_cells(LEDGER)
    assert sum(row["Review_Status"] == "attested" for row in rows) == 101
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 7
    assert sum(row["Review_Status"] in {"ambiguous", "illegible"}
               for row in rows) == 0
    assert len(guard.stage_forms(rows)) == 103
    with pytest.raises(RuntimeError, match="2655 of 2763 cells unreviewed"):
        guard.require_full_review(rows)


def test_cumulative_ledgers_are_disjoint_and_accounted():
    paths = sorted((PACKAGE / "manual_chunks").glob("items_*_hand_keyed.tsv"))
    rows = guard.load_manual_ledgers(paths)
    assert len(rows) == 2763
    assert sum(row["Review_Status"] == "attested" for row in rows) == 2670
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 93
    assert len(guard.stage_forms(rows)) == 2770
    guard.require_full_review(rows)
    forms, audit = guard.build_source_package(
        rows, guard.load_registry(PACKAGE / "list_registry.tsv")
    )
    assert len(forms) == 2770 and len(audit) == 2763


def test_guard_rejects_ocr_bearing_or_undeclared_rows(tmp_path):
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8"), delimiter="\t"))
    bad_ocr = tmp_path / "bad_ocr.tsv"
    fieldnames = list(rows[0]) + ["OCR_Evidence"]
    with bad_ocr.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "not admissible"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_ocr)

    bad_declaration = tmp_path / "bad_declaration.tsv"
    rows[0]["Reviewer_Declaration"] = ""
    with bad_declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_declaration)


def test_frozen_staging_regenerates_exactly_and_has_immutable_locators():
    result = subprocess.run(
        ["python3", str(PACKAGE / "import_adi_2015.py"), "--pdf", str(PDF), "--stage"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "2670 attested; 93 source blanks; 0 ambiguous; 0 illegible" in result.stdout
    assert "staged 2770 forms and 2763 audit rows" in result.stdout

    assert hashlib.sha256((PACKAGE / "staged_forms.csv").read_bytes()).hexdigest() == (
        "edb29a8f65fea0600e3d54bfcf2adef81fd833c47b619de5cd701bd61df4031c"
    )
    assert hashlib.sha256((PACKAGE / "staged_audit.tsv").read_bytes()).hexdigest() == (
        "6fb69a145419fff42c6b48d8e965acf2dbd9dc06bd297edf2e19f62e4f88877b"
    )
    with (PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    assert len(forms) == 2770 and all(len(row) == 15 for row in forms)
    assert len({row[10] for row in forms}) == 2770
    assert Counter(row[0] for row in forms) == Counter(
        MisingPadamMiriMinyong=613, BoriKarko=610, BokarRamo=1249, Milang=298
    )
    assert {row[7].split("[", 1)[0] for row in forms} == {"padung-sako2015adi"}
    assert all("Appendix B, printed p. " in row[7] and ", item " in row[7]
               and ", list " in row[7] for row in forms)
    assert all(row[10].startswith("padung-sako2015adi:item:") for row in forms)
    assert all(row[14].startswith("dialect:") for row in forms)


def test_zero_unresolved_and_lossless_profile_inventory_are_exhaustive():
    with (PACKAGE / "unresolved_readings.tsv").open(encoding="utf-8", newline="") as stream:
        unresolved = list(csv.DictReader(stream, delimiter="\t"))
    assert unresolved == []
    with (PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    expected = Counter(char for row in forms for char in row[2])
    with (PACKAGE / "symbol_inventory.tsv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream, delimiter="\t"))
    assert len(inventory) == 42
    assert {row["Symbol"]: int(row["Count"]) for row in inventory} == expected
    assert "�" not in expected
    assert expected["?"] == 2
    assert expected["̪"] and expected["̃"] and expected["ː"]
    assert all(row["Decision"].startswith("preserve") for row in inventory)


def test_preintegration_manifest_render_and_registry_contract_are_exact():
    result = subprocess.run(
        ["python3", str(AUDITOR)], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "cells=2763 attested=2670 blanks=93 staged=2770 unresolved=0" in result.stdout
    manifest = json.loads((PACKAGE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "source-local-preintegration-audit-complete"
    assert manifest["pdf"] == {
        "bytes": 743089,
        "pages": 45,
        "path": "tmp/pdfs/adi_2015/silesr2015_016.pdf",
        "sha256": "8e1500383a02445252a3eb6973a1b011fabea71eb25ad79fc43ba5b78bd1135c",
    }
    assert manifest["manual_review"]["bundle_sha256"] == (
        "a9a1aac22c77c4cf66230c2fa014a7b151cd676db44810744b06748070bd92f0"
    )
    assert manifest["statuses"] == {
        "ambiguous": 0, "attested": 2670, "illegible": 0,
        "source_blank": 93, "unresolved": 0, "unreviewed": 0,
    }
    assert manifest["staging"]["rows"] == 2770
    assert manifest["staging"]["unique_entry_keys"] == 2770
    assert manifest["renders"] == {
        "artifacts": 22,
        "bytes": 15131560,
        "dpi": 400,
        "manifest_sha256": "85b5dee74f2614b3028c5dce9082476b55abd6f60e0c5a42a72ead9c28d84173",
        "physical_pages": "17--38",
        "tree_sha256": "0746c68daf48349570eb0d37e2d69afb79c22571d69a410484b8437e1efd794c",
    }
    assert manifest["registry_contract"]["base_languages"] == [
        "MisingPadamMiriMinyong", "BoriKarko", "BokarRamo", "Milang"
    ]
    assert len(manifest["registry_contract"]["dialect_ids"]) == 9
    assert manifest["integration_contract"]["install_exactly"] == 2770
    assert manifest["integration_contract"]["exclude_exactly"] == {
        "controls": 0, "source_blank_cells": 93, "unresolved": 0
    }

    with (PACKAGE / "render_hashes.tsv").open(encoding="utf-8", newline="") as stream:
        renders = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["Relative_Path"] for row in renders] == [
        f"page-{page}.png" for page in range(17, 39)
    ]


def test_shared_source_specific_installation_is_exact_and_fully_routed():
    installed = ROOT / "data/other/forms/20260829-sil-adi.csv"
    profile = ROOT / "conversion/sil-adi.txt"
    assert installed.read_bytes() == (PACKAGE / "staged_forms.csv").read_bytes()
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == (
        "edb29a8f65fea0600e3d54bfcf2adef81fd833c47b619de5cd701bd61df4031c"
    )
    assert profile.read_bytes() == (PACKAGE / "conversion_profile.tsv").read_bytes()
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == (
        "61f298367f3e9217c170797cc6c4dbebc3c4b86eb90936b3e2f52561ed013d71"
    )

    with installed.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    assert len(forms) == 2770 and all(len(row) == 15 for row in forms)
    assert len({row[10] for row in forms}) == 2770
    assert Counter(row[0] for row in forms) == Counter(
        MisingPadamMiriMinyong=613, BoriKarko=610, BokarRamo=1249, Milang=298
    )
    assert {row[7].split("[", 1)[0] for row in forms} == {"padung-sako2015adi"}
    assert all(row[1] == "" and row[8] == "" and row[9] == "" for row in forms)

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert {
        key: (languages[key]["Glottocode"], languages[key]["Quality"])
        for key in ("MisingPadamMiriMinyong", "BoriKarko", "BokarRamo", "Milang")
    } == {
        "MisingPadamMiriMinyong": ("misi1242", "C"),
        "BoriKarko": ("bori1243", "C"),
        "BokarRamo": ("boka1249", "C"),
        "Milang": ("mila1245", "C"),
    }

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    expected_dialects = {
        "sil-adi-2015-minyong-rayang": ("MisingPadamMiriMinyong", "MN"),
        "sil-adi-2015-bori-bogu-payum": ("BoriKarko", "BR"),
        "sil-adi-2015-ramo-ngorlung": ("BokarRamo", "RM"),
        "sil-adi-2015-milang-village": ("Milang", "ML"),
        "sil-adi-2015-pailibo-irgo": ("BokarRamo", "PL"),
        "sil-adi-2015-ashing-ningging": ("BokarRamo", "AS"),
        "sil-adi-2015-padam-siluk": ("MisingPadamMiriMinyong", "PD"),
        "sil-adi-2015-shimong-mobuk": ("BoriKarko", "SM"),
        "sil-adi-2015-bokar-manigong": ("BokarRamo", "BK"),
    }
    for dialect_id, (language_id, source_id) in expected_dialects.items():
        assert dialects[dialect_id]["Language_ID"] == language_id
        assert dialects[dialect_id]["Source_Language_ID"] == source_id
        assert dialects[dialect_id]["Latitude"] == dialects[dialect_id]["Longitude"] == ""
    assert {row[14] for row in forms} == {
        dialects[dialect_id]["Tag"] for dialect_id in expected_dialects
    }

    with (ROOT / "cldf/references.csv").open(encoding="utf-8", newline="") as stream:
        references = {row["ID"]: row for row in csv.DictReader(stream)}
    reference = references["padung-sako2015adi"]
    assert reference["Progress"].startswith(
        "Appendix B, printed pages 13--34: 2,770 manually verified"
    )
    assert reference["OCR"] == "No"
    assert reference["Etymology_Provenance"] == "none"

    build = (ROOT / "make_cldf.py").read_text(encoding="utf-8")
    assert 'if source_key == "padung-sako2015adi":' in build
    assert 'row_ipa = "sil-adi"' in build
    assert '"sil-adi",' in build

    manifest = json.loads((PACKAGE / "shared_integration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "shared-source-specific-integration-complete"
    assert manifest["scope"]["conceptual_cells"] == 2763
    assert manifest["scope"]["installed_expanded_forms"] == 2770
    assert manifest["scope"]["source_blank_cells_audit_only"] == 93
    assert manifest["unresolved_coordinates"] == []
