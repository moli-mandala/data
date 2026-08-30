"""Regression tests for SIL ESR 2007-017's Batote Dogri wordlist."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_dogri_2007"
EXTRACTOR = SOURCE_DIR / "extract_dogri.py"
IMPORTER = SOURCE_DIR / "import_dogri.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-dogri.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-dogri-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-dogri-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "brightbill-turner2007dogri"
PDF = ROOT.parent / "tmp/pdfs/dogri/silesr2007_017.pdf"

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
    assert result.stdout.strip() == "installed=207 blanks=3 audit=210"


def test_extractor_reproduces_frozen_snapshot_when_publisher_pdf_is_available(tmp_path):
    if not PDF.exists():
        return
    original = SNAPSHOT.read_bytes()
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == "items=210 responses=207 blanks=3 legacy_bytes=477"
    assert SNAPSHOT.read_bytes() == original


def test_snapshot_topology_and_official_map_census_are_pinned():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 210
    assert [int(row["Item"]) for row in rows] == list(range(1, 211))
    assert Counter(row["Status"] for row in rows) == Counter(response=207, blank=3)
    assert {int(row["Item"]) for row in rows if row["Status"] == "blank"} == {11, 23, 24}
    with (SOURCE_DIR / "sil_ipa93_used.tsv").open(encoding="utf-8", newline="") as stream:
        mapping = list(csv.DictReader(stream, delimiter="\t"))
    assert len(mapping) == 20
    assert sum(int(row["Occurrences"]) for row in mapping) == 477


def test_installed_scope_keys_and_representative_legacy_ipa_recovery():
    rows = forms()
    assert len(rows) == 207
    assert {row["Language_ID"] for row in rows} == {"dog"}
    assert {row["Tags"] for row in rows} == {
        "dialect:dog:sil-dogri-2007-batote:Batote"
    }
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B, printed p. ") for row in rows)
    by_gloss = {row["Gloss"]: row["Form"] for row in rows}
    assert by_gloss["body"] == "d͡ʒɪsɘm"
    assert by_gloss["horns"] == "sĩɡ"
    assert by_gloss["rainbow"] == "teɾkman, indɾadʰənʊʃ"
    assert by_gloss["white"] == "t͡ʃɪʈːə"
    assert by_gloss["run!"] == "dɔɽ, dɔɽɪjɛ"


def test_complete_audit_and_manifest_reconcile_every_prompt():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 210
    assert Counter(row["Status"] for row in rows) == Counter(installed=207, excluded=3)
    assert {int(row["Item"]) for row in rows if row["Status"] == "excluded"} == {11, 23, 24}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == "04fa21ccf3ca7317ef1a1b3e587b4f1c058b3fb773ea56724d726945a12622c0"
    assert manifest["snapshot_sha256"] == "3de760e2bd715383d9e27fc93321c3480e382d0c5b829eca86e9d394d138ebcf"
    assert manifest["official_converter_map_sha256"] == "f2bb1070e8393f83e6ea83d8b08ee0b07e23bbe0176ccb9ca97b76793809df31"
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0
    assert manifest["ocr_used"] is False
    assert manifest["comparison_wordlists_reported_but_not_published"] == 5


def test_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [row for row in csv.DictReader(stream) if row["ID"] == "sil-dogri-2007-batote"]
    assert len(dialects) == 1
    assert dialects[0]["Language_ID"] == "dog"
    assert dialects[0]["Latitude"] == "33.118262"
    assert dialects[0]["Longitude"] == "75.308893"
    assert dialects[0]["Quality"] == "C"
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib


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
    assert any("sil-dogri-2007-batote" in row["Tags"] for row in compiled)
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
