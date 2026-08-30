"""Regression tests for SIL ESR 2019-006's Lahul Valley wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_lahul_2019"
EXTRACTOR = SOURCE_DIR / "extract_lahul.py"
IMPORTER = SOURCE_DIR / "import_lahul.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-lahul.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-lahul-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-lahul-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "chamberlain-chamberlain2019lahul"
PDF = ROOT.parent / "tmp/pdfs/lahul/silesr2019_006.pdf"

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
    assert result.stdout.strip() == "installed=5027 excluded=1179 audit=6206"


def test_extractor_reproduces_frozen_snapshot_when_publisher_pdf_is_available():
    if not PDF.exists():
        return
    original = SNAPSHOT.read_bytes()
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == (
        "prompts=210 rows=6206 target_rows=5056 "
        "target_responses=5027 audit_only=1179"
    )
    assert SNAPSHOT.read_bytes() == original


def test_snapshot_pins_every_prompt_list_and_response_line():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 6206
    assert {int(row["Item"]) for row in rows} == set(range(1, 211))
    assert len({(row["Lect_Code"], row["Site"]) for row in rows}) == 27
    assert Counter(row["Source_Scope"] for row in rows) == Counter(
        target=5056, prior_list=676, comparison=474,
    )
    assert Counter(row["Status"] for row in rows) == Counter(
        response=6139, no_entry=67,
    )
    assert sum("wrapped form joined" in row["Review"] for row in rows) == 10
    assert all("�" not in row["Form"] for row in rows)
    assert not any(0xE000 <= ord(char) <= 0xF8FF for row in rows for char in row["Form"])


def test_installed_scope_keys_and_representative_unicode_transcriptions():
    rows = forms()
    assert len(rows) == 5027
    assert Counter(row["Language_ID"] for row in rows) == Counter({
        "cih": 427, "lhl": 427, "lae": 1807,
        "lbf": 441, "bfu": 668, "sbu": 1257,
    })
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(not row["Parameter_ID"] and not row["Cognateset"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A.4, printed p. ") for row in rows)
    assert all(row["Tags"].startswith(f"dialect:{row['Language_ID']}:sil-lahul-2019-") for row in rows)
    observed = {(row["Gloss"], row["Form"]) for row in rows}
    assert ("body", "ɸuk") in observed
    assert ("spider", "ɾəɳdʒ.kɾiɳdʒ") in observed
    assert ("those", "ode.tsʌŋ.mɐ") in observed
    assert ("we.[incl]", "ŋɐdʒɐthʌmtʃe") in observed
    assert ("you.[pl]", "bʌdhe.dʒhɐe") in observed
    assert not any(row["Form"].casefold() in {"no entry", "ɴo entry"} for row in rows)


def test_source_sound_profile_covers_difficult_lahul_symbols():
    from segments.tokenizer import Tokenizer

    profile = Tokenizer(str(ROOT / "conversion/sil-lahul.txt"))

    def convert(value):
        return unicodedata.normalize(
            "NFC", profile(value, column="IPA").replace(" ", "").replace("#", " ")
        )

    assert convert("ɾəɳdʒ.kɾiɳdʒ") == "rəṇj.kriṇj"
    assert convert("ŋɐʒɐtshʌmtʃe") == "ŋažaʦhamce"
    assert convert("k̚ t̪ɐ") == "k̚ ta"


def test_complete_audit_and_manifest_reconcile_every_response():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6206
    assert Counter(row["Status"] for row in rows) == Counter(installed=5027, excluded=1179)
    assert Counter(row["Source_Scope"] for row in rows) == Counter(
        target=5056, prior_list=676, comparison=474,
    )
    assert sum(
        row["Source_Scope"] == "target" and row["Status"] == "excluded"
        for row in rows
    ) == 29
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "17f8178505ef88879baecbd5d9fa6dd4f2bb885330722cbac21df70c71e47252"
    )
    assert manifest["snapshot_sha256"] == (
        "b0e83a40b929288d7b59a6ba3789096648a954b6ddab33ed6d6c182d5afcc963"
    )
    assert manifest["unparsed_lines"] == manifest["replacement_or_private_use_glyphs"] == 0
    assert manifest["wrapped_forms_joined_and_visually_checked"] == 10
    assert manifest["ocr_used"] is False
    assert manifest["etymology_edges"] == 0


def test_language_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    expected = {
        "cih": "chin1475", "lhl": "lahu1250", "lae": "patt1248",
        "lbf": "tina1246", "bfu": "gahr1239", "sbu": "stod1241",
    }
    assert {language_id: languages[language_id]["Glottocode"] for language_id in expected} == expected
    assert all(languages[language_id]["Quality"] == "C" for language_id in expected)

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [row for row in csv.DictReader(stream) if row["ID"].startswith("sil-lahul-2019-")]
    assert len(dialects) == 22
    assert Counter(row["Language_ID"] for row in dialects) == Counter(
        cih=2, lhl=2, lae=8, lbf=2, bfu=3, sbu=5,
    )
    assert all(row["Latitude"] and row["Longitude"] and row["Quality"] == "C" for row in dialects)
    assert all("not a source coordinate" in row["Location"] for row in dialects)
    tingrat = next(row for row in dialects if row["ID"] == "sil-lahul-2019-bh-tingrat")
    assert tingrat["Glottocode"] == "maya1278"
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
    compiled_dialects = {
        tag.split(":", 3)[2]
        for row in compiled for tag in row["Tags"].split()
        if tag.startswith("dialect:") and "sil-lahul-2019-" in tag
    }
    assert len(compiled_dialects) == 22
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
