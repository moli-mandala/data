"""Regression tests for SSNP volume 4's Pashto, Waneci, and Ormuri lists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/ssnp04_1992"
EXTRACTOR = SOURCE_DIR / "extract_ssnp04.py"
IMPORTER = SOURCE_DIR / "import_ssnp04.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-ssnp04.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-ssnp04-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-ssnp04-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "hallberg1992pashto"
PDF = ROOT.parent / "tmp/pdfs/ssnp04/32847_SSNP04.pdf"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def test_importer_rebuilds_checked_artifacts():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == "installed=7131 excluded=69 audit=7200"


def test_extractor_reproduces_frozen_snapshot_when_publisher_pdf_is_available():
    if not PDF.exists():
        return
    original = SNAPSHOT.read_bytes()
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == (
        "prompts=200 lists=36 cells=7200 responses=7131 no_entry=68 blank=1"
    )
    assert SNAPSHOT.read_bytes() == original


def test_snapshot_pins_every_printed_prompt_and_list_cell():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 7200
    expected_items = set(range(1, 211)) - {24, 29, 32, 50, 173, 174, 175, 176, 195, 208}
    assert {int(row["Item"]) for row in rows} == expected_items
    assert len({row["List_Code"] for row in rows}) == 36
    assert Counter(row["Status"] for row in rows) == Counter(
        response=7131, no_entry=68, blank=1,
    )
    assert sum(int(row["Continuation_Lines"]) > 0 for row in rows) == 42
    assert not any("�" in row["Form"] for row in rows)
    assert not any(0xE000 <= ord(char) <= 0xF8FF for row in rows for char in row["Form"])


def test_installed_scope_keys_and_chart_accurate_unicode_transcriptions():
    rows = forms()
    assert len(rows) == 7131
    assert Counter(row["Language_ID"] for row in rows) == Counter(
        Psht=6732, wne=199, Orm=200,
    )
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(not row["Parameter_ID"] and not row["Cognateset"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B, printed p. ") for row in rows)
    assert all(row["Tags"].startswith(f"dialect:{row['Language_ID']}:ssnp04-") for row in rows)
    observed = {(row["Gloss"], row["Form"], row["Language_ID"]) for row in rows}
    assert ("ear", "ɣwʌg", "Psht") in observed
    assert ("arm/hand", "ɭɑs", "Psht") in observed
    assert ("cauliflower", "gwʌl goɸi", "Psht") in observed
    assert ("spider", "ɣʌ̃ɳye / ɣʌ̃ɽye", "Psht") in observed
    assert ("to be hungry / The man was hungry.", "sʌɽʌi wʌgʌi ðɑi", "Psht") in observed


def test_ssnp_profile_covers_all_new_forms_and_preserves_phonemic():
    import io
    from make_cldf import parse_file

    errors = io.StringIO()
    rows, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 7131, "for_conversion": 7131}
    assert errors.getvalue() == ""
    assert all(row.ipa == row.old_form for row in rows)
    assert not any("�" in row.form for row in rows)


def test_complete_audit_and_manifest_reconcile_every_cell():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7200
    assert Counter(row["Status"] for row in rows) == Counter(installed=7131, excluded=69)
    assert sum(row["Reason"].startswith("source prints --") for row in rows) == 68
    assert sum(row["Reason"] == "source cell is blank" for row in rows) == 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "83e2d833c06ecb4e40bfb0d316061d6b398b743bac299dc870c90c88a4b96f18"
    )
    assert manifest["snapshot_sha256"] == (
        "e3be47472daf107beea441ed6d823bb07f1460d1c4ae5adb687e2773a1db8e15"
    )
    assert manifest["unparsed_cells"] == manifest["replacement_or_private_use_glyphs"] == 0
    assert manifest["ocr_used"] is False
    assert manifest["etymology_edges"] == 0


def test_language_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["wne"]["Glottocode"] == "wane1241"
    assert languages["Orm"]["Glottocode"] == "ormu1247"
    assert languages["Psht"]["Glottocode"] == "cent1973"

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [row for row in csv.DictReader(stream) if row["ID"].startswith("ssnp04-")]
    assert len(dialects) == 36
    assert Counter(row["Language_ID"] for row in dialects) == Counter(Psht=34, wne=1, Orm=1)
    assert all(row["Latitude"] and row["Longitude"] for row in dialects)
    assert all("not a source coordinate" in row["Location"] for row in dialects)
    assert next(row for row in dialects if row["ID"] == "ssnp04-krk")["Quality"] == "C"
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@book{{{SOURCE_KEY}," in bib


def test_compiled_rows_preserve_every_source_locator_and_dialect():
    installed = forms()
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [row for row in csv.DictReader(stream) if SOURCE_KEY in row["Source"]]
    compiled_citations = {
        citation
        for row in compiled
        for citation in row["Source"].split(";")
        if citation.startswith(f"{SOURCE_KEY}[")
    }
    assert compiled_citations == {row["Source"] for row in installed}
    compiled_dialects = {
        tag.split(":", 3)[2]
        for row in compiled for tag in row["Tags"].split()
        if tag.startswith("dialect:") and "ssnp04-" in tag
    }
    assert len(compiled_dialects) == 36
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
