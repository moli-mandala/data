"""Regression tests for SIL ESR 2010-012 Pahari/Pothwari wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_pahari_pothwari_2010"
EXTRACTOR = SOURCE_DIR / "extract_pahari_pothwari.py"
IMPORTER = SOURCE_DIR / "import_pahari_pothwari.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-pahari-pothwari.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-pahari-pothwari-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-pahari-pothwari-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
PDF = ROOT.parent / "tmp/pdfs/pahari-pothwari/silesr2010-012.pdf"
SOURCE_KEY = "lothers-lothers2010pahari"

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
    assert result.stdout.strip() == "installed=3038 excluded=434 audit=3472"


def test_extractor_reproduces_frozen_snapshot_when_publisher_pdf_is_available():
    if not PDF.exists():
        return
    original = SNAPSHOT.read_bytes()
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "rows=3472" in result.stdout
    assert SNAPSHOT.read_bytes() == original


def test_snapshot_pins_every_prompt_list_and_printed_code():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 3472
    assert {int(row["Item"]) for row in rows} == set(range(1, 218))
    assert Counter(row["Lect_Code"] for row in rows) == Counter({
        code: 217 for code in (
            "MOS", "GHO", "DEW", "AYU", "KOH", "NIL", "THA", "LOR",
            "OSI", "MUZ", "DUN", "BHA", "ABB", "MAN", "MIR", "GUJ",
        )
    })
    assert Counter(row["Status"] for row in rows) == Counter(response=3454, blank=18)
    assert sum(row["Raw_Lect_Code"] == "AUS" and row["Lect_Code"] == "OSI" for row in rows) == 14
    assert sum(row["Excluded_From_Similarity"] == "Yes" for row in rows) == 11 * 16
    assert not any("�" in row["Form"] for row in rows)


def test_installed_scope_keys_and_difficult_transcriptions():
    rows = forms()
    assert len(rows) == 3038
    assert Counter(row["Language_ID"] for row in rows) == Counter(poth=3038)
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(not row["Parameter_ID"] and not row["Cognateset"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B.1, printed p. ") for row in rows)
    assert all(row["Tags"].startswith("dialect:poth:sil-pahari-pothwari-2010-") for row in rows)
    observed = {(row["Gloss"], row["Form"], row["Entry_Key"]) for row in rows}
    assert ("wind", "vǎ̤", "silpaharipothwari2010:g051:dun") in observed
    assert ("wet", "sɪǰʸa", "silpaharipothwari2010:g132:tha") in observed
    assert ("mud", "čɪkʌṛ", "silpaharipothwari2010:g058:osi") in observed


def test_profile_covers_all_forms_and_preserves_phonemic():
    import io
    from make_cldf import parse_file

    errors = io.StringIO()
    rows, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 3038, "for_conversion": 3038}
    assert errors.getvalue() == ""
    assert all(row.ipa == row.old_form for row in rows)
    assert not any("�" in row.form for row in rows)


def test_complete_audit_and_manifest_reconcile_every_cell():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3472
    assert Counter(row["Status"] for row in rows) == Counter(installed=3038, excluded=434)
    assert sum("Hindko comparison" in row["Reason"] for row in rows) == 434
    assert sum(row["Source_Status"] == "blank" for row in rows) == 18
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "e3695a807c4856118303eca74b68b192817ea69251fa8be62abb7b27e4c1ad6f"
    )
    assert manifest["snapshot_sha256"] == (
        "9ef7a0f32c9b2d1d263c1d0fba213d9db67bf1927237915320227c1c6492e7e1"
    )
    assert manifest["unparsed_cells"] == manifest["replacement_or_private_use_glyphs"] == 0
    assert manifest["ocr_used"] is False
    assert manifest["etymology_edges"] == 0


def test_language_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["poth"]["Glottocode"] == "paha1251"
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith("sil-pahari-pothwari-2010-")
        ]
    assert len(dialects) == 14
    assert {row["Language_ID"] for row in dialects} == {"poth"}
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all("no point coordinate" in row["Location"] for row in dialects)
    assert Counter(row["Quality"] for row in dialects) == Counter(A=8, B=6)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib


def test_compiled_rows_preserve_every_source_locator_and_dialect():
    installed = forms()
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [row for row in csv.DictReader(stream) if SOURCE_KEY in row["Source"]]
    compiled_citations = {
        source
        for row in compiled
        for source in row["Source"].split(";")
        if source.startswith(f"{SOURCE_KEY}[")
    }
    assert compiled_citations == {row["Source"] for row in installed}
    compiled_dialects = {
        tag.split(":", 3)[2]
        for row in compiled for tag in row["Tags"].split()
        if tag.startswith("dialect:") and "sil-pahari-pothwari-2010-" in tag
    }
    assert len(compiled) == len(installed) == 3038
    assert len(compiled_dialects) == 14
    assert Counter(
        tag.split(":", 3)[2]
        for row in compiled for tag in row["Tags"].split()
        if tag.startswith("dialect:") and "sil-pahari-pothwari-2010-" in tag
    ) == Counter({f"sil-pahari-pothwari-2010-{code}": 217 for code in (
        "mos", "gho", "dew", "ayu", "koh", "nil", "tha", "lor", "osi",
        "muz", "dun", "bha", "mir", "guj",
    )})
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
