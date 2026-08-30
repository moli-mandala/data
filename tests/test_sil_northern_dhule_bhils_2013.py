from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_northern_dhule_bhils_2013"
IMPORTER = PACKAGE / "import_northern_dhule_bhils.py"
PREINTEGRATION_AUDITOR = PACKAGE / "preintegration_audit.py"
PDF = ROOT.parent / "tmp/pdfs/northern_dhule_bhils_2013/silesr2013_004.pdf"

spec = importlib.util.spec_from_file_location("sil_northern_dhule_importer", IMPORTER)
dhule = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(dhule)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_chunk(path: Path, entries: list[dict[str, str]], fields=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or dhule.FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(entries)


def chunk_row(base: dict[str, str], **updates: str) -> dict[str, str]:
    row = dict(base)
    row.update({
        "Gloss": "body",
        "Manual_Transcription": "1 ɖil",
        "Review_Status": "attested",
        "Confidence": "high",
        "Uncertainty": "",
        "Reviewer_Method": "manual-source-image; rendered-300dpi; OCR-not-accepted",
        "Reviewed_At": "2026-08-28",
        "Reviewer_Declaration": dhule.DECLARATION,
    })
    row.update(updates)
    return row


def test_source_pin_full_topology_and_list_roles():
    assert PDF.stat().st_size == 9_214_722
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == dhule.PDF_SHA256
    base = dhule.validate_base()
    assert len(base) == 2730
    assert len({(row["Item"], row["Site_Code"]) for row in base}) == 2730
    assert {int(row["Item"]) for row in base} == set(range(1, 211))
    assert {int(row["PDF_Page"]) for row in base} == set(range(91, 134))
    specs = dhule.validate_registry()
    assert Counter(row["Scope"] for row in specs) == Counter(target=12, comparison_control=1)
    assert specs[-1]["Site_Code"] == "TOR" and specs[-1]["Install"] == "no"


def test_reviewed_batches_exact_counts_coordinates_nfc_and_manual_stamps():
    effective = dhule.overlay_manual_chunks(dhule.validate_base())
    assert dhule.validate_effective(effective) == Counter(
        attested=2703, blank=24, ambiguous=3
    )
    reviewed = [row for row in effective if row["Review_Status"] != "unreviewed"]
    assert len(reviewed) == 2730
    assert {int(row["Item"]) for row in reviewed} == set(range(1, 211))
    assert {int(row["PDF_Page"]) for row in reviewed} == set(range(91, 134))
    assert all(int(row["Printed_Page"]) == int(row["PDF_Page"]) - 8 for row in reviewed)
    assert all(row["Reviewer_Method"].startswith("manual-source-image; rendered-") and row["Reviewer_Method"].endswith("; OCR-not-accepted") for row in reviewed)
    assert all(row["Reviewer_Declaration"] == dhule.DECLARATION for row in reviewed)
    assert all(unicodedata.is_normalized("NFC", value) for row in effective for value in row.values())
    assert Counter(row["Site_Code"] == "TOR" for row in reviewed) == Counter({False: 2520, True: 210})
    assert dhule.page_for(46) == 100 and dhule.page_for(49) == 100
    assert dhule.page_for(50) == 101 and dhule.page_for(194) == 129
    assert dhule.page_for(195) == 130 and dhule.page_for(199) == 130
    assert dhule.page_for(200) == 131 and dhule.page_for(204) == 131
    assert dhule.page_for(205) == 132 and dhule.page_for(209) == 132
    assert dhule.page_for(210) == 133


def test_all_batch_blanks_and_unresolved_have_exact_cell_evidence():
    effective = dhule.overlay_manual_chunks(dhule.validate_base())
    blanks = {(r["PDF_Page"], r["Printed_Page"], r["Item"], r["Site_Code"]) for r in effective if r["Review_Status"] == "blank"}
    assert blanks == {
        *(("93", "85", "11", site) for site in "KEL AMO BHU TOR".split()),
        *(("95", "87", "21", site) for site in "MUN AST MAN BHU AML KAN SHA TOR".split()),
        *(("96", "88", "27", site) for site in "AST MAN BHU KAN SHA".split()),
        ("102", "94", "56", "AMO"),
        ("126", "118", "177", "AML"), ("126", "118", "177", "KAN"),
        ("126", "118", "177", "SHA"), ("131", "123", "204", "BHU"),
        ("131", "123", "204", "SHA"), ("132", "124", "207", "TOR"),
    }
    unresolved = rows(PACKAGE / "unresolved_readings.tsv")
    assert [(r["PDF_Page"], r["Printed_Page"], r["Item"], r["Site_Code"], r["Column"], r["Review_Status"]) for r in unresolved] == [
        ("92", "84", "10", "KEL", "left", "ambiguous"),
        ("97", "89", "31", "MUN", "left", "ambiguous"),
        ("105", "97", "74", "TOR", "right", "ambiguous"),
    ]
    assert all(row["Uncertainty"] for row in unresolved)


def test_items_036_070_batch_accounting_and_high_resolution_rechecks():
    batch = rows(PACKAGE / "manual_chunks/items_036_070_hand_keyed.tsv")
    assert len(batch) == 455
    assert Counter(row["Review_Status"] for row in batch) == Counter(attested=454, blank=1)
    assert Counter(row["Site_Code"] == "TOR" for row in batch) == Counter({False: 420, True: 35})
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] != "TOR") == Counter(attested=419, blank=1)
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] == "TOR") == Counter(attested=35)
    blank = next(row for row in batch if row["Review_Status"] == "blank")
    assert (blank["PDF_Page"], blank["Printed_Page"], blank["Item"], blank["Site_Code"], blank["Column"]) == ("102", "94", "56", "AMO", "left")
    assert "horizontal dash rule" in blank["Uncertainty"]
    rereviewed = [row for row in batch if "+900dpi-rereview" in row["Reviewer_Method"]]
    assert {(row["Item"], row["Site_Code"]) for row in rereviewed} == {
        ("47", "TOR"), ("51", "DIG"), *(("62", site) for site in dhule.SITES),
    }
    assert next(row for row in batch if row["Item"] == "69" and row["Site_Code"] == "KEL")["Gloss"] == "wheat_(husked)"
    assert next(row for row in batch if row["Item"] == "70" and row["Site_Code"] == "MAN")["Manual_Transcription"] == "1 dʒuwar"


def test_items_071_105_batch_accounting_and_high_resolution_rechecks():
    batch = rows(PACKAGE / "manual_chunks/items_071_105_hand_keyed.tsv")
    assert len(batch) == 455
    assert Counter(row["Review_Status"] for row in batch) == Counter(attested=454, ambiguous=1)
    assert Counter(row["Site_Code"] == "TOR" for row in batch) == Counter({False: 420, True: 35})
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] != "TOR") == Counter(attested=420)
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] == "TOR") == Counter(attested=34, ambiguous=1)
    unresolved = next(row for row in batch if row["Review_Status"] == "ambiguous")
    assert (unresolved["PDF_Page"], unresolved["Printed_Page"], unresolved["Item"], unresolved["Site_Code"], unresolved["Column"]) == ("105", "97", "74", "TOR", "right")
    assert "(laŋa / (laɳa" in unresolved["Uncertainty"]
    rereviewed = [row for row in batch if "+900dpi-rereview" in row["Reviewer_Method"]]
    assert {(row["Item"], row["Site_Code"]) for row in rereviewed} == {
        ("74", "TOR"), ("87", "TOR"), ("91", "MAN"), ("91", "SEG"),
        ("101", "DIG"), ("105", "TOR"),
    }
    assert next(row for row in batch if row["Item"] == "101" and row["Site_Code"] == "DIG")["Manual_Transcription"] == "1 ɳaβ"


def test_items_106_140_batch_accounting_and_high_resolution_rechecks():
    batch = rows(PACKAGE / "manual_chunks/items_106_140_hand_keyed.tsv")
    assert len(batch) == 455
    assert Counter(row["Review_Status"] for row in batch) == Counter(attested=455)
    assert Counter(row["Site_Code"] == "TOR" for row in batch) == Counter({False: 420, True: 35})
    assert not [row for row in batch if row["Review_Status"] in {"blank", "ambiguous", "illegible"}]
    rereviewed = [row for row in batch if "+900dpi-rereview" in row["Reviewer_Method"]]
    assert len(rereviewed) == 156
    assert {int(row["Item"]) for row in rereviewed} == {*range(106, 115), 120, 125, 140}
    assert next(row for row in batch if row["Item"] == "120" and row["Site_Code"] == "SEG")["Manual_Transcription"] == "6 t̪s̪hoʈipʌr-ɖahaɖu"
    assert next(row for row in batch if row["Item"] == "140" and row["Site_Code"] == "SHA")["Manual_Transcription"] == "2 aɭaɳo, 4 haʈe"


def test_items_141_175_batch_accounting_and_high_resolution_rechecks():
    batch = rows(PACKAGE / "manual_chunks/items_141_175_hand_keyed.tsv")
    assert len(batch) == 455
    assert Counter(row["Review_Status"] for row in batch) == Counter(attested=455)
    assert Counter(row["Site_Code"] == "TOR" for row in batch) == Counter({False: 420, True: 35})
    assert not [row for row in batch if row["Review_Status"] in {"blank", "ambiguous", "illegible"}]
    rereviewed = [row for row in batch if "+900dpi-rereview" in row["Reviewer_Method"]]
    assert len(rereviewed) == 429
    assert {int(row["Item"]) for row in rereviewed} == {*range(141, 150), *range(151, 175)}
    assert next(row for row in batch if row["Item"] == "148" and row["Site_Code"] == "DIG")["Manual_Transcription"] == "1 udzʌlõ, 2 paɳɖo"
    assert next(row for row in batch if row["Item"] == "169" and row["Site_Code"] == "AMO")["Manual_Transcription"] == "4, 5 kʌlɪχ"
    assert next(row for row in batch if row["Item"] == "170" and row["Site_Code"] == "TOR")["Manual_Transcription"] == "4 kolak-dzaʈiɳ"


def test_items_176_210_batch_accounting_and_high_resolution_rechecks():
    batch = rows(PACKAGE / "manual_chunks/items_176_210_hand_keyed.tsv")
    assert len(batch) == 455
    assert Counter(row["Review_Status"] for row in batch) == Counter(attested=449, blank=6)
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] != "TOR") == Counter(attested=415, blank=5)
    assert Counter(row["Review_Status"] for row in batch if row["Site_Code"] == "TOR") == Counter(attested=34, blank=1)
    assert all("+900dpi-rereview" in row["Reviewer_Method"] for row in batch)
    assert next(row for row in batch if row["Item"] == "200" and row["Site_Code"] == "KEL")["PDF_Page"] == "131"
    assert next(row for row in batch if row["Item"] == "205" and row["Site_Code"] == "KEL")["PDF_Page"] == "132"


def test_overlay_rejects_duplicate_unknown_overlap_coordinate_and_ocr_fields(tmp_path: Path):
    base = dhule.validate_base()
    seed = next(row for row in base if row["Item"] == "36" and row["Site_Code"] == "KEL")
    good = tmp_path / "good.tsv"; write_chunk(good, [chunk_row(seed, Gloss="rope")])
    patched = dhule.overlay_manual_chunks(base, [good])
    assert next(row for row in patched if row["Item"] == "36" and row["Site_Code"] == "KEL")["Review_Status"] == "attested"

    duplicate = tmp_path / "duplicate.tsv"; write_chunk(duplicate, [chunk_row(seed, Gloss="rope")])
    with pytest.raises(ValueError, match="Duplicate review-chunk key"):
        dhule.overlay_manual_chunks(base, [good, duplicate])

    unknown = tmp_path / "unknown.tsv"; write_chunk(unknown, [chunk_row(seed, Item="999", Gloss="rope")])
    with pytest.raises(ValueError, match="Unknown review-chunk key"):
        dhule.overlay_manual_chunks(base, [unknown])

    reviewed_effective = dhule.overlay_manual_chunks(base)
    reviewed = next(row for row in reviewed_effective if row["Item"] == "1" and row["Site_Code"] == "KEL")
    overlap = tmp_path / "overlap.tsv"; write_chunk(overlap, [reviewed])
    with pytest.raises(ValueError, match="overlaps reviewed base"):
        dhule.overlay_manual_chunks(reviewed_effective, [overlap])

    bad_coord = tmp_path / "coordinate.tsv"; write_chunk(bad_coord, [chunk_row(seed, Gloss="rope", PDF_Page="999")])
    with pytest.raises(ValueError, match="coordinate mismatch"):
        dhule.overlay_manual_chunks(base, [bad_coord])

    ocr_fields = dhule.FIELDS + ["OCR_Candidate"]
    ocr_row = chunk_row(seed, Gloss="rope"); ocr_row["OCR_Candidate"] = "not admissible"
    ocr = tmp_path / "ocr.tsv"; write_chunk(ocr, [ocr_row], ocr_fields)
    with pytest.raises(ValueError, match="OCR-bearing review chunk is inadmissible"):
        dhule.overlay_manual_chunks(base, [ocr])


def test_review_chunks_are_ocr_blind_and_similarity_labels_are_notes_only():
    for name in ("items_001_035_hand_keyed.tsv", "items_036_070_hand_keyed.tsv", "items_071_105_hand_keyed.tsv", "items_106_140_hand_keyed.tsv", "items_141_175_hand_keyed.tsv", "items_176_210_hand_keyed.tsv"):
        chunk = PACKAGE / "manual_chunks" / name
        with chunk.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            assert not any(field.upper().startswith("OCR") for field in reader.fieldnames or ())
            assert sum(1 for _ in reader) == 455
    assert dhule.strip_similarity_labels("2 silpe, 1 lakaɖo") == "silpe, lakaɖo"
    assert dhule.strip_similarity_labels("1, 2 munɖkʌ") == "munɖkʌ"


def test_complete_review_stages_only_resolved_targets_and_incomplete_refuses():
    result = subprocess.run(
        ["python3", str(IMPORTER), "--verify-pdf", "--stage"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    staged = list(csv.reader((PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="")))
    assert len(staged) == 2497 and all(len(row) == 15 for row in staged)
    assert len({row[10] for row in staged}) == 2497
    assert {row[0] for row in staged} == {"Vasavi", "Noiri", "PauriBareli", "RathwiBareli"}
    assert not any("TOR" in row[7] for row in staged)
    partial = dhule.overlay_manual_chunks(dhule.validate_base(), dhule.chunk_paths()[:-1])
    with pytest.raises(RuntimeError, match="455 of 2,730 cells unreviewed"):
        dhule.require_complete(partial)


def test_manifest_and_audit_docs_match_complete_source_local_state():
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["conceptual_cells"] == 2730
    assert manifest["target_cells"] == 2520
    assert manifest["comparison_control_cells"] == 210
    assert manifest["cells_manually_reviewed"] == 2730
    assert manifest["cells_attested"] == 2703
    assert manifest["cells_blank"] == 24
    assert manifest["cells_ambiguous"] == 3
    assert manifest["cells_illegible"] == 0
    assert manifest["cells_unreviewed"] == 0
    assert (manifest["target_cells_attested"], manifest["target_cells_blank"], manifest["target_cells_ambiguous"]) == (2497, 21, 2)
    assert (manifest["control_cells_attested"], manifest["control_cells_blank"], manifest["control_cells_ambiguous"]) == (206, 3, 1)
    assert manifest["installed_forms"] == 0 and manifest["staged_forms"] == 2497 and manifest["ocr_authority"].startswith("none")
    checklist = (PACKAGE / "CHECKLIST.md").read_text(encoding="utf-8")
    audit = (PACKAGE / "manual_chunks/AUDIT_ITEMS_176_210.md").read_text(encoding="utf-8")
    assert "2,497" in checklist and "overall ingest is not complete" in checklist
    assert "455 reviewed" in audit and "0 illegible" in audit


def test_source_local_profile_covers_every_staged_input_character():
    with (PACKAGE / "conversion_profile.tsv").open(encoding="utf-8", newline="") as stream:
        profile = list(csv.DictReader(stream, delimiter="\t"))
    graphemes = {row["Grapheme"] for row in profile}
    staged = list(csv.reader((PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="")))
    chars = {char for row in staged for char in row[2]}
    assert chars <= graphemes
    assert "�" not in chars


def test_preintegration_freeze_reconciliation_and_render_contract_are_exact():
    result = subprocess.run(
        ["python3", str(PREINTEGRATION_AUDITOR)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "cells=2730 target_forms=2497 renders=234 reconciliation=1470" in result.stdout
    manifest = json.loads((PACKAGE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "source-local-preintegration-audit-complete"
    assert manifest["pdf"] == {
        "bytes": 9_214_722,
        "pages": 133,
        "sha256": "edeeeda98cb76624df1a0d70c765cc816ea463d75bc79ec20883c62e6fc1c482",
    }
    assert manifest["frozen_artifacts"]["manual_cell_bundle_sha256"] == (
        "046ff03ef2af36c51f1b25538f081aabb7c28d3ccd1776d3ec545fad6463e8c1"
    )
    assert manifest["staged_target_forms"] == {
        "rows": 2497,
        "sha256": "5641b9d7ecfb44e6e644efba35e65223260291b7a8724b1fd25fac2fc94d3ed4",
        "unique_entry_keys": 2497,
    }
    assert manifest["statuses"]["target"] == {
        "attested": 2497, "blank": 21, "ambiguous": 2,
        "illegible": 0, "unreviewed": 0,
    }
    assert manifest["statuses"]["control"] == {
        "attested": 206, "blank": 3, "ambiguous": 1,
        "illegible": 0, "unreviewed": 0,
    }
    assert manifest["renders"]["artifacts"] == 234
    assert manifest["renders"]["tree_sha256"] == (
        "816261371d0ada57996b2b1135267024629fcb3a7827b07bac4e53bc68f8ec43"
    )
    assert manifest["profile"]["missing_staged_input_characters"] == []
    assert manifest["reconciliation"]["rows"] == 1470
    assert manifest["reconciliation"]["counts"] == {
        "varghesekumar2015noira|literal-ledger-exact": 3,
        "varghesekumar2015noira|same-source-representation-differs": 627,
        "varkey-vunnamatla2018bareli|both-blank": 8,
        "varkey-vunnamatla2018bareli|dhule-attested-bareli-excluded": 4,
        "varkey-vunnamatla2018bareli|exact-single-form": 261,
        "varkey-vunnamatla2018bareli|same-source-representation-differs": 567,
    }
    reconciliation = rows(PACKAGE / "cross_source_reconciliation.tsv")
    assert len(reconciliation) == 1470
    assert Counter(row["Related_Source"] for row in reconciliation) == Counter({
        "varghesekumar2015noira": 630,
        "varkey-vunnamatla2018bareli": 840,
    })


def test_shared_source_specific_installation_is_exact_and_fully_routed():
    installed = ROOT / "data/other/forms/20260829-sil-northern-dhule-bhils.csv"
    profile = ROOT / "conversion/sil-northern-dhule-bhils.txt"
    assert installed.read_bytes() == (PACKAGE / "staged_forms.csv").read_bytes()
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == (
        "5641b9d7ecfb44e6e644efba35e65223260291b7a8724b1fd25fac2fc94d3ed4"
    )
    assert profile.read_bytes() == (PACKAGE / "conversion_profile.tsv").read_bytes()
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == (
        "b0bca6f983bbcf87dc43769c804ae02a73db45d58fad1e86f975fb8b9f7456ce"
    )

    with installed.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    assert len(forms) == 2497 and all(len(row) == 15 for row in forms)
    assert len({row[10] for row in forms}) == 2497
    assert Counter(row[0] for row in forms) == Counter(
        Vasavi=836, Noiri=416, PauriBareli=620, RathwiBareli=625
    )
    assert {row[7].split("[", 1)[0] for row in forms} == {"watters2013northerndhule"}
    assert not any("list TOR" in row[7] for row in forms)

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert {(languages[key]["Glottocode"], languages[key]["Latitude"], languages[key]["Longitude"])
            for key in ("Vasavi", "Noiri")} == {("vasa1239", "", ""), ("noir1238", "", "")}

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    new_dialects = {
        "sil-dhule-2013-vasavi-kelpada": "Vasavi",
        "sil-dhule-2013-vasavi-dhanoura": "Vasavi",
        "sil-dhule-2013-vasavi-digiamba": "Vasavi",
        "sil-dhule-2013-vasavi-amoda": "Vasavi",
        "sil-dhule-2013-noiri-mundalwad": "Noiri",
        "sil-dhule-2013-noiri-astamba": "Noiri",
        "sil-dhule-2013-pauri-bhusha": "PauriBareli",
        "sil-dhule-2013-rathwi-kangai": "RathwiBareli",
    }
    for dialect_id, language_id in new_dialects.items():
        assert dialects[dialect_id]["Language_ID"] == language_id
        assert dialects[dialect_id]["Latitude"] == dialects[dialect_id]["Longitude"] == ""
    installed_dialect_tags = {tag for row in forms for tag in row[14].split(";") if tag}
    assert installed_dialect_tags == {dialects[key]["Tag"] for key in new_dialects} | {
        dialects[key]["Tag"] for key in {
            "sil-bareli-2018-bareli-pauri-mandvi",
            "sil-bareli-2018-rathwi-pauri-amalwadi",
            "sil-bareli-2018-rathwi-pauri-segwi",
            "sil-bareli-2018-bareli-pauri-shahana",
        }
    }

    with (ROOT / "cldf/references.csv").open(encoding="utf-8", newline="") as stream:
        references = {row["ID"]: row for row in csv.DictReader(stream)}
    assert "bhildhule" not in references
    reference = references["watters2013northerndhule"]
    assert reference["Progress"].startswith("Appendix C, printed pages 83--125: 2,497")
    assert reference["OCR"] == "No"
    assert reference["Etymology_Provenance"] == "none"

    build = (ROOT / "make_cldf.py").read_text(encoding="utf-8")
    assert 'if source_key == "watters2013northerndhule":' in build
    assert 'row_ipa = "sil-northern-dhule-bhils"' in build
    assert '"sil-northern-dhule-bhils",' in build

    manifest = json.loads((PACKAGE / "shared_integration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "shared-source-specific-integration-complete"
    assert manifest["scope"]["installed_target_attestations"] == 2497
    assert manifest["scope"]["control_cells_audit_only"] == 210
    assert manifest["scope"]["target_blanks_audit_only"] == 21
    assert manifest["scope"]["target_ambiguities_audit_only"] == 2
