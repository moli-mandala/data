from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer
import pytest

import make_cldf


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_kurumba_2012"
LISTS = PACKAGE / "list_registry.tsv"
PAGES = PACKAGE / "page_review.tsv"
PROMPTS = PACKAGE / "prompt_review.tsv"
MANUAL = PACKAGE / "manual_transcription.tsv"
CHUNKS = PACKAGE / "manual_chunks"
IMPORTER = PACKAGE / "import_kurumba.py"
PDF = ROOT.parent / "tmp/pdfs/kurumba_2012/silesr2012_015.pdf"
TARGET_INSTALLER = PACKAGE / "install_target_forms.py"
FROZEN_FORMS = PACKAGE / "staged_forms.csv"
FROZEN_AUDIT = PACKAGE / "staged_audit.csv"
TARGET_FORMS = PACKAGE / "installed_target_forms.csv"
INTEGRATION_AUDIT = PACKAGE / "shared_integration_audit.csv"
INTEGRATION_MANIFEST = PACKAGE / "shared_integration_manifest.json"
INSTALLED_FORMS = ROOT / "data/other/forms/20260828-sil-kurumba.csv"
SHARED_PROFILE = ROOT / "conversion/sil-kurumba-2012.txt"
DIALECTS = ROOT / "cldf/dialects.csv"
SOURCES = ROOT / "cldf/sources.bib"
BUILD_SCRIPT = ROOT / "make_cldf.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_source_is_pinned_and_appendix_topology_is_exact():
    assert PDF.stat().st_size == 128_831_439
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == "250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137"
    lists = rows(LISTS)
    assert len(lists) == 19
    assert Counter(row["Scope"] for row in lists) == Counter(target=15, control=4)
    assert Counter(row["Language_ID"] for row in lists) == Counter(
        Kannada=15, Tamil=1, AluKurumba=1, Badaga=1, BettaKurumba=1
    )
    assert sum((int(row["PDF_Last"]) - int(row["PDF_First"]) + 1) * 25 for row in lists) == 10450


def test_manual_ledger_accounts_for_every_cell_without_accepting_ocr():
    lists = rows(LISTS)
    cells = rows(MANUAL)
    assert len(cells) == 10450
    assert len({row["Cell_Key"] for row in cells}) == 10450
    assert {(row["List_Key"], row["Item"]) for row in cells} == {
        (spec["List_Key"], str(item)) for spec in lists for item in range(1, 551)
    }
    assert Counter(row["Scope"] for row in cells) == Counter(target=8250, control=2200)
    assert Counter(row["Cell_Status"] for row in cells) == Counter(pending=10350, attested=92, blank=8)
    reviewed = [row for row in cells if row["Cell_Status"] == "attested"]
    assert {(row["List_Key"], row["Item"]) for row in reviewed} == {
        (list_key, str(item))
        for list_key in ("tamil_madras", "kannada_bangalore")
        for item in range(1, 51)
        if not (
            (list_key == "tamil_madras" and item in {42, 49})
            or (list_key == "kannada_bangalore" and item in {34, 35, 37, 42, 44, 50})
        )
    }
    assert all(row["Manual_Form"] and row["Confidence"] == "high" and row["Reviewer"] == "OpenAI Codex" for row in reviewed)
    assert all(row["Review_Method"].startswith("manual visual transcription from rendered source scan") for row in reviewed)
    # Corrupt OCR is retained for comparison, but the authoritative field is blank.
    assert any(row["OCR_Form_Scaffold"] for row in cells)


def test_parallel_manual_review_chunks_are_disjoint_and_ocr_blind():
    chunks = sorted(CHUNKS.glob("*.tsv"))
    assert {"p239-root-review.tsv", "p249-root-review.tsv"} <= {path.name for path in chunks}
    reviewed = [row for path in chunks for row in rows(path)]
    assert len(reviewed) >= 100
    assert len({row["Cell_Key"] for row in reviewed}) == len(reviewed)
    assert {row["Cell_Status"] for row in reviewed} <= {
        "attested", "blank", "ambiguous", "illegible"
    }
    assert all("OCR_" not in field for field in reviewed[0])
    assert all(
        row["Review_Method"]
        == "manual visual transcription from rendered source scan; OCR used only as locator/comparison scaffold"
        for row in reviewed
    )
    assert all(
        row["Manual_Form"] == unicodedata.normalize("NFC", row["Manual_Form"])
        for row in reviewed
    )


def test_page_and_prompt_ledgers_are_complete():
    pages = rows(PAGES)
    prompts = rows(PROMPTS)
    assert len(pages) == 220
    assert {int(row["PDF_Page"]) for row in pages} == set(range(217, 437))
    assert sum(int(row["Conceptual_Cells"]) for row in pages) == 10450
    assert sum(Counter(row["Review_Status"] for row in pages).values()) == 220
    assert Counter(row["Review_Status"] for row in pages) == Counter(complete=220)
    page217 = next(row for row in pages if row["PDF_Page"] == "217")
    assert (page217["Attested"], page217["Blank"], page217["Ambiguous"], page217["Illegible"]) == ("50", "0", "0", "0")
    page218 = next(row for row in pages if row["PDF_Page"] == "218")
    assert (page218["Attested"], page218["Blank"], page218["Ambiguous"], page218["Illegible"]) == ("42", "8", "0", "0")
    tamil_kannada_pages = [
        next(row for row in pages if row["PDF_Page"] == str(page))
        for page in range(219, 228)
    ]
    assert [
        (
            row["PDF_Page"], row["Printed_Page"], row["Items_First"],
            row["Items_Last"], row["Left_List"], row["Right_List"],
            row["Conceptual_Cells"], row["Review_Status"],
            row["Attested"], row["Blank"], row["Ambiguous"], row["Illegible"],
        )
        for row in tamil_kannada_pages
    ] == [
        ("219", "214", "51", "75", "tamil_madras", "kannada_bangalore", "50", "complete", "47", "3", "0", "0"),
        ("220", "215", "76", "100", "tamil_madras", "kannada_bangalore", "50", "complete", "46", "4", "0", "0"),
        ("221", "216", "101", "125", "tamil_madras", "kannada_bangalore", "50", "complete", "39", "11", "0", "0"),
        ("222", "217", "126", "150", "tamil_madras", "kannada_bangalore", "50", "complete", "43", "7", "0", "0"),
        ("223", "218", "151", "175", "tamil_madras", "kannada_bangalore", "50", "complete", "45", "5", "0", "0"),
        ("224", "219", "176", "200", "tamil_madras", "kannada_bangalore", "50", "complete", "31", "19", "0", "0"),
        ("225", "220", "201", "225", "tamil_madras", "kannada_bangalore", "50", "complete", "34", "16", "0", "0"),
        ("226", "221", "226", "250", "tamil_madras", "kannada_bangalore", "50", "complete", "39", "11", "0", "0"),
        ("227", "222", "251", "275", "tamil_madras", "kannada_bangalore", "50", "complete", "39", "11", "0", "0"),
    ]
    tamil_kannada_final_pages = [
        next(row for row in pages if row["PDF_Page"] == str(page))
        for page in range(228, 239)
    ]
    assert [
        (
            row["PDF_Page"], row["Printed_Page"], row["Items_First"],
            row["Items_Last"], row["Left_List"], row["Right_List"],
            row["Conceptual_Cells"], row["Review_Status"],
            row["Attested"], row["Blank"], row["Ambiguous"], row["Illegible"],
        )
        for row in tamil_kannada_final_pages
    ] == [
        ("228", "223", "276", "300", "tamil_madras", "kannada_bangalore", "50", "complete", "33", "17", "0", "0"),
        ("229", "224", "301", "325", "tamil_madras", "kannada_bangalore", "50", "complete", "50", "0", "0", "0"),
        ("230", "225", "326", "350", "tamil_madras", "kannada_bangalore", "50", "complete", "40", "10", "0", "0"),
        ("231", "226", "351", "375", "tamil_madras", "kannada_bangalore", "50", "complete", "41", "9", "0", "0"),
        ("232", "227", "376", "400", "tamil_madras", "kannada_bangalore", "50", "complete", "50", "0", "0", "0"),
        ("233", "228", "401", "425", "tamil_madras", "kannada_bangalore", "50", "complete", "46", "4", "0", "0"),
        ("234", "229", "426", "450", "tamil_madras", "kannada_bangalore", "50", "complete", "49", "1", "0", "0"),
        ("235", "230", "451", "475", "tamil_madras", "kannada_bangalore", "50", "complete", "48", "2", "0", "0"),
        ("236", "231", "476", "500", "tamil_madras", "kannada_bangalore", "50", "complete", "44", "6", "0", "0"),
        ("237", "232", "501", "525", "tamil_madras", "kannada_bangalore", "50", "complete", "46", "4", "0", "0"),
        ("238", "233", "526", "550", "tamil_madras", "kannada_bangalore", "50", "complete", "46", "4", "0", "0"),
    ]
    page239 = next(row for row in pages if row["PDF_Page"] == "239")
    assert (page239["Attested"], page239["Blank"], page239["Ambiguous"], page239["Illegible"]) == ("44", "5", "1", "0")
    page249 = next(row for row in pages if row["PDF_Page"] == "249")
    assert (page249["Attested"], page249["Blank"], page249["Ambiguous"], page249["Illegible"]) == ("3", "47", "0", "0")
    page414 = next(row for row in pages if row["PDF_Page"] == "414")
    assert (page414["Left_List"], page414["Right_List"], page414["Conceptual_Cells"]) == (
        "kalangal", "masinagudi_jennu", "50"
    )
    maddur_pages = [
        next(row for row in pages if row["PDF_Page"] == str(page))
        for page in range(415, 437)
    ]
    assert [
        (
            row["PDF_Page"], row["Items_First"], row["Items_Last"],
            row["Left_List"], row["Right_List"], row["Conceptual_Cells"],
        )
        for row in maddur_pages
    ] == [
        (str(page), str(1 + (page - 415) * 25), str(25 + (page - 415) * 25),
         "maddur_betta", "", "25")
        for page in range(415, 437)
    ]
    page415 = maddur_pages[0]
    assert (page415["Attested"], page415["Blank"], page415["Ambiguous"], page415["Illegible"]) == ("23", "2", "0", "0")
    page436 = maddur_pages[-1]
    assert (page436["Items_First"], page436["Items_Last"]) == ("526", "550")
    assert (page436["Attested"], page436["Blank"], page436["Ambiguous"], page436["Illegible"]) == ("20", "5", "0", "0")
    assert sum("fallback" in row["Notes"] for row in pages) == 5
    assert len(prompts) == 550
    assert {int(row["Item"]) for row in prompts} == set(range(1, 551))
    prompt_counts = Counter(row["Review_Status"] for row in prompts)
    assert sum(prompt_counts.values()) == 550
    assert prompt_counts["complete"] >= 75
    assert [row["Manual_Gloss"] for row in prompts[:5]] == ["stone", "soil", "water", "leaf", "flower"]


def test_importer_reports_complete_and_stages_only_manual_data():
    base_cells = rows(MANUAL)
    chunk_rows = [row for path in sorted(CHUNKS.glob("*.tsv")) for row in rows(path)]
    effective = Counter(row["Cell_Status"] for row in base_cells)
    effective["pending"] -= len(chunk_rows)
    effective.update(row["Cell_Status"] for row in chunk_rows)
    pages_pending = sum(row["Review_Status"] != "complete" for row in rows(PAGES))
    prompts_pending = sum(row["Review_Status"] != "complete" for row in rows(PROMPTS))
    result = subprocess.run(
        ["python3", str(IMPORTER), "--verify-pdf"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"pages_pending={pages_pending}" in result.stdout
    assert f"prompts_pending={prompts_pending}" in result.stdout
    for status in ("pending", "attested", "blank", "ambiguous", "illegible"):
        assert f"cells_{status}={effective[status]}" in result.stdout
    staged = subprocess.run(
        ["python3", str(IMPORTER), "--verify-pdf", "--stage"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "review_complete=1" in staged.stdout
    with (PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        staged_forms = list(csv.reader(stream))
    with (PACKAGE / "staged_audit.csv").open(encoding="utf-8", newline="") as stream:
        staged_audit = list(csv.DictReader(stream))
    assert len(staged_forms) == effective["attested"] == 4738
    assert len(staged_audit) == 10450
    build_source = IMPORTER.read_text(encoding="utf-8").split("def build(", 1)[1].split("def write(", 1)[0]
    assert "OCR_Form_Scaffold" not in build_source
    assert "OCR_Gloss_Scaffold" not in build_source


def test_importer_refuses_incomplete_manual_staging():
    spec = importlib.util.spec_from_file_location("sil_kurumba_2012_importer", IMPORTER)
    assert spec and spec.loader
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    _, pages, prompts, cells = importer.validate_topology()
    cells[0]["Cell_Status"] = "pending"
    with pytest.raises(RuntimeError, match="manual visual review incomplete"):
        importer.require_manual_completion(pages, prompts, cells)


def test_unresolved_readings_are_explicit_and_never_guessed():
    unresolved = rows(PACKAGE / "unresolved_readings.tsv")
    p239 = next(row for row in unresolved if row["Cell_Key"] == "kurumba2012:pudukkottai:i020")
    assert p239["Status"] == "ambiguous"
    assert "Excluded from staging" in p239["Resolution"]
    checklist = (PACKAGE / "CHECKLIST.md").read_text(encoding="utf-8")
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))
    assert f"{manifest['conceptual_cells'] - manifest['cells_pending']:,}" in checklist
    assert f"{manifest['cells_pending']:,}" in checklist
    metadata = rows(PACKAGE / "metadata_review.tsv")
    assert [row["PDF_Page"] for row in metadata] == ["214", "215", "216"]
    assert {row["Review_Status"] for row in metadata} == {"complete"}
    chunk_rows = [row for path in sorted(CHUNKS.glob("*.tsv")) for row in rows(path)]
    effective = Counter(row["Cell_Status"] for row in rows(MANUAL))
    effective["pending"] -= len(chunk_rows)
    effective.update(row["Cell_Status"] for row in chunk_rows)
    assert manifest["conceptual_cells"] - manifest["cells_pending"] == 10450 - effective["pending"]
    assert manifest["cells_pending"] == effective["pending"] and manifest["conceptual_cells"] == 10450
    for status in ("attested", "blank", "ambiguous", "illegible"):
        assert manifest[f"cells_{status}"] == effective[status]
    assert manifest["installed_forms"] == effective["attested"]
    assert manifest["ocr_authority"].startswith("none")


def test_shared_target_filter_is_exhaustive_and_preserves_frozen_inputs():
    before = (sha256(FROZEN_FORMS), sha256(FROZEN_AUDIT))
    subprocess.run(["python3", str(TARGET_INSTALLER)], cwd=ROOT, check=True)
    assert before == (
        "71e985690e76497ae276d43ad93b7a44ab75565ba9bd88f11de6a5d04a43b29b",
        "3ed3963bdbc309ec3926589e70942c9b52acb8a17b8687d2e47a503ea2e743bf",
    )
    assert (sha256(FROZEN_FORMS), sha256(FROZEN_AUDIT)) == before
    assert INSTALLED_FORMS.read_bytes() == TARGET_FORMS.read_bytes()
    assert sha256(INSTALLED_FORMS) == (
        "5ac00f37816119acb13bb2d833070fcf063e9dea0b2ee1c565a5d03281bdf137"
    )
    with INSTALLED_FORMS.open(encoding="utf-8", newline="") as stream:
        installed = list(csv.reader(stream))
    with INTEGRATION_AUDIT.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    assert len(installed) == 3204
    assert len({row[10] for row in installed}) == 3204
    assert len(audit) == 10450
    assert Counter(row["Integration_Status"] for row in audit) == Counter({
        "installed_target": 3204,
        "excluded_control": 1534,
        "excluded_blank_target": 5044,
        "excluded_blank_control": 666,
        "excluded_ambiguous_target": 1,
        "excluded_illegible_target": 1,
    })
    installed_keys = {row[10] for row in installed}
    assert installed_keys == {
        row["Entry_Key"] for row in audit if row["Integration_Status"] == "installed_target"
    }
    assert all(row["Scope"] == "target" for row in audit if row["Entry_Key"] in installed_keys)
    unresolved = {
        row["Cell_Key"]: row["Integration_Status"]
        for row in audit if "ambiguous" in row["Integration_Status"] or "illegible" in row["Integration_Status"]
    }
    assert unresolved == {
        "kurumba2012:pudukkottai:i020": "excluded_ambiguous_target",
        "kurumba2012:kotagiri_alu:i025": "excluded_illegible_target",
    }


def test_shared_registry_and_reference_match_the_frozen_list_metadata():
    lists = {row["Dialect_ID"]: row for row in rows(LISTS)}
    with DIALECTS.open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    assert len(lists) == 19
    for dialect_id, source in lists.items():
        row = dialects[dialect_id]
        assert row["Language_ID"] == source["Language_ID"]
        assert row["Source_Language_ID"] == source["List_Key"]
        assert row["Name"] == source["Source_Label"]
        assert row["Latitude"] == row["Longitude"] == ""
        assert row["Location"].startswith(source["Location"].split("; ")[0])
    text = SOURCES.read_text(encoding="utf-8")
    start = text.index("@techreport{blairetal2012kurumba,")
    entry = text[start:text.index("\n}\n", start) + 3]
    assert "number       = {2012-015}" in entry
    assert "10,450 conceptual cells" in entry
    assert "fifteen target lists only" in entry
    assert "comparison-control lists remain audit-only" in entry
    assert "supplied or verified no retained reading" in entry


def test_shared_profile_route_and_coverage_are_complete():
    tokenizer = Tokenizer(str(SHARED_PROFILE))
    with FROZEN_FORMS.open(encoding="utf-8", newline="") as stream:
        frozen = list(csv.reader(stream))
    assert len(frozen) == 4738
    for row in frozen:
        converted = tokenizer(row[2], column="IPA", segment_separator="", separator="")
        assert "�" not in converted
    assert tokenizer("su:rjʌn", column="IPA", segment_separator="", separator="") == "sūryan"
    assert tokenizer("ku:rʌ'", column="IPA", segment_separator="", separator="") == "kūra'"
    assert tokenizer("aɽɽja", column="IPA", segment_separator="", separator="") == "aṛṛya"
    assert tokenizer("na':", column="IPA", segment_separator="", separator="") == "na':"
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    route = 'if source_key == "blairetal2012kurumba":'
    assert route in build
    block = build[build.index(route):build.index(route) + 180]
    assert 'row_ipa = "sil-kurumba-2012"' in block
    assert "row_convert = True" in block
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(
        str(INSTALLED_FORMS), errors, file_num=1, param_counter={}
    )
    assert errors.getvalue() == ""
    assert len(parsed) == 3204
    assert stats == {"converted": 3204, "for_conversion": 3204}
    first = parsed[0]
    assert first.ipa == "bəɳɖe"
    assert first.form == "bəṇḍe"


def test_shared_integration_manifest_freezes_counts_hashes_and_deferred_gates():
    manifest = json.loads(INTEGRATION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "shared_source_specific_integration_complete"
    assert manifest["frozen_forms"] == {
        "rows": 4738,
        "sha256": "71e985690e76497ae276d43ad93b7a44ab75565ba9bd88f11de6a5d04a43b29b",
    }
    assert manifest["frozen_audit"] == {
        "rows": 10450,
        "sha256": "3ed3963bdbc309ec3926589e70942c9b52acb8a17b8687d2e47a503ea2e743bf",
    }
    assert manifest["installed_target_forms"]["rows"] == 3204
    assert manifest["installed_target_forms"]["sha256"] == sha256(INSTALLED_FORMS)
    assert manifest["sound_profile"]["sha256"] == sha256(SHARED_PROFILE)
    assert manifest["sound_profile"]["unresolved_mappings"] == []
    assert manifest["unresolved_coordinates"] == [
        "kurumba2012:pudukkottai:i020",
        "kurumba2012:kotagiri_alu:i025",
    ]
    assert manifest["post_freeze_reconciliation"] == {
        "prior_rows_with_source_key": 0,
        "prior_rows_with_kurumba2012_entry_key": 0,
        "result": "new source installation; no pre-existing row competed with the frozen target output",
    }
    assert len(manifest["deferred_gates"]) == 5
