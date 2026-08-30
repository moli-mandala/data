"""Regression tests for SIL ESR 2009-011's Malvi wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_malvi_2009"
IMPORTER = SOURCE_DIR / "import_malvi.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-malvi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-malvi-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-malvi-manifest.json"
NIMADI_AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-nimadi-audit.csv"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "varghese-john-samuel2009malvi"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_importer_rebuilds_checked_artifacts_without_pdf_or_pdfplumber():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == (
        "installed=6894 comparisons=1891 by_name=37 omitted=90 audit=8912"
    )


def test_snapshot_topology_and_manifest_are_pinned():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 8912
    assert {int(row["Concept"]) for row in rows} == set(range(1, 211))
    assert len({row["Lect"] for row in rows}) == 38
    assert Counter(row["Source_Status"] for row in rows) == Counter(
        response=8751, by_name=37, no_entry=10, omitted_prompt=114,
    )
    assert Counter(row["Response_Index"] for row in rows) == Counter(
        {"1": 7980, "2": 861, "3": 60, "4": 9, "5": 1, "6": 1}
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == (
        "e67e314974ab10eb8244b08dba56d08d1ce8cbf16eaef1be022071d49032a2dd"
    )
    assert manifest["snapshot_sha256"] == (
        "aca0edf2509f14920b13edb9594aeae481d670fd9eaa361d8956211093e51d10"
    )
    assert manifest["counts"]["installed_malvi_forms"] == 6894
    assert manifest["counts"]["target_lists"] == 30
    assert manifest["font_recovery"]["used_cids"] == 34


def test_installed_scope_keys_and_legacy_ipa_recovery():
    rows = forms()
    assert len(rows) == 6894
    assert {row["Language_ID"] for row in rows} == {"mewari_basad"}
    assert len({row["Tags"].split(":")[2] for row in rows}) == 30
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all("(cid:" not in row["Form"] and "�" not in row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B, printed p. ") for row in rows)

    by_source = {row["Source"]: row for row in rows}
    assert by_source[
        f"{SOURCE_KEY}[Appendix B, printed p. 149, item 182, Ujjaini-Malvi-Harsodan]"
    ]["Form"] == "kʰʌⁱlo, ɠʰajlijo"
    assert any(row["Form"] == "vʌdʒʌndɑ̻ɾ" for row in rows)
    assert any(row["Form"] == "mʌtʃːi" for row in rows)


def test_complete_audit_reconciles_controls_by_name_and_removed_prompts():
    rows = audited()
    assert len(rows) == 8912
    assert Counter(row["Status"] for row in rows) == Counter(installed=6894, excluded=2018)
    assert Counter(row["Reason"] for row in rows if row["Status"] == "excluded") == Counter({
        "borrowed or standard comparison list": 1891,
        "source records a by-name response, not a lexical form": 37,
        "prompt disqualified and absent from the published appendix": 90,
    })
    assert {int(row["Concept"]) for row in rows if "disqualified" in row["Reason"]} == {11, 23, 24}
    assert {row["Category"] for row in rows if row["Status"] == "installed"} >= {"a", "b"}


def test_thillorkhurd_cross_source_unicode_check():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        malvi = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if row["Lect"] == "Ujjaini-Malvi-Thillorkhurd" and row["Source_Status"] == "response"
        ]
    with NIMADI_AUDIT.open(encoding="utf-8", newline="") as stream:
        later = [row for row in csv.DictReader(stream) if row["Lect"] == "Malvi"]
    first: dict[str, list[str]] = defaultdict(list)
    second: dict[str, list[str]] = defaultdict(list)
    for row in malvi:
        first[row["Concept"]].append(row["Form"])
    for row in later:
        second[row["Concept"]].append(row["Form"])
    agreements = [
        (concept, left)
        for concept in first.keys() & second.keys()
        for left in first[concept]
        for right in second[concept]
        if unicodedata.normalize("NFC", left) == unicodedata.normalize("NFC", right)
    ]
    assert len(agreements) == 132
    assert len({concept for concept, _ in agreements}) == 126


def test_dialect_and_bibliographic_registration():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [row for row in csv.DictReader(stream) if row["ID"].startswith("sil-malvi-2009-")]
    assert len(dialects) == 30
    assert {row["Language_ID"] for row in dialects} == {"mewari_basad"}
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib


def test_compiled_rows_preserve_every_source_locator_and_dialect():
    installed = forms()
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [row for row in csv.DictReader(stream) if SOURCE_KEY in row["Source"]]
    assert len(compiled) == 2128
    compiled_citations = {
        citation
        for row in compiled
        for citation in row["Source"].split(";")
        if citation.startswith(f"{SOURCE_KEY}[")
    }
    assert compiled_citations == {row["Source"] for row in installed}
    assert len(compiled_citations) == 6182
    assert len({
        tag for row in compiled for tag in row["Tags"].split()
        if "sil-malvi-2009-" in tag
    }) == 30
    assert all(row["Original"] and "�" not in row["Original"] for row in compiled)
