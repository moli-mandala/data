"""Focused source-input checks; no full data or browser database build."""

import csv
import importlib.util
import io
import json
import unicodedata
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/gaddi_grammar_2026"
CSV = DATA / "data/other/forms/20260925-kumari-gaddi.csv"
PROFILE = DATA / "conversion/kumari-gaddi-ipa.txt"
spec = importlib.util.spec_from_file_location("kumari_gaddi", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_audit_counts_keys_and_scope():
    audit = [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]
    assert len(audit) == 393
    assert len({row["entry_key"] for row in audit}) == 393
    assert {int(row["item"].split(".")[0]) for row in audit if not row.get("table")} == set(range(1, 222))
    assert {row["pdf_page"] for row in audit} == set(range(136, 146))
    assert sum(row["status"] == "ingested" for row in audit) == 392
    assert [row["entry_key"] for row in audit if row["excluded"]] == ["kumari-gaddi:C1:104"]
    assert all(row["printed_page"] == row["pdf_page"] - 17 for row in audit)
    assert all(row["language_id"] == "ga" for row in audit)
    assert [row["item"] for row in audit if row["uncertainty"] and not row.get("table")] == ["44", "88", "132", "155.3", "207.2"]


def test_installed_rows_reproduce_from_frozen_audit():
    audit = [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]
    expected, regenerated = source.output(audit)
    installed = list(csv.reader(CSV.open()))
    assert expected == installed
    assert len(regenerated) == 393 and len(installed) == 392
    audit = [r for r in audit if not r["excluded"]]
    assert all(len(row) == 15 and row[0] == "ga" and row[7].startswith("kumari2026gaddi[") for row in installed)
    assert all(row[10] == audit[i]["entry_key"] and not row[1] and not row[8] for i, row in enumerate(installed))
    assert all("item " + audit[i]["item"] in row[7] for i, row in enumerate(installed))


def test_pinned_pdf_reproduces_all_attestations_and_layout_repairs():
    if not source.PDF.exists():
        pytest.skip("Pinned external PDF not present")
    assert source.sha256(source.PDF) == source.PDF_SHA256
    rows = source.read_rows() + source.read_appendix()
    audit = [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]
    generated, regenerated_audit = source.output(rows)
    assert generated == list(csv.reader(CSV.open()))
    assert regenerated_audit == audit
    by_item = {row["item"]: row for row in rows if not row.get("table")}
    assert by_item["73"]["source_gloss"] == "heart"
    assert by_item["98"]["form"] == "dʒuː̃"
    assert len([row for row in rows if not row.get("table") and "joined PDF-spaced aspiration" in row["extraction_repairs"]]) == 6
    assert len([row for row in rows if "joined PDF-wrapped glyph inside IPA brackets" in row["extraction_repairs"]]) == 5


def test_printed_anomalies_and_grammatical_labels_remain_distinct():
    rows = {r[10].split(":")[-1]: r for r in csv.reader(CSV.open()) if ":B1:" in r[10]}
    assert rows["132"][2] == "lu:ɳ" and "uncertain" in rows["132"][14]
    assert rows["155.3"][3] == "split" and "uncertain" in rows["155.3"][14]
    assert rows["207.2"][2] == rows["121"][2] and rows["207.2"][3] == "elder brother"
    assert rows["99.1"][3] == "many" and rows["99.1"][14] == "m"
    assert rows["99.2"][3] == "many" and rows["99.2"][14] == "f"
    assert rows["148"][3] == "smell" and rows["148"][14] == "noun"
    assert rows["149"][3] == "smell" and rows["149"][14] == "verb"
    assert rows["16.1"][3] == "blow (as in wind)"


def test_profile_covers_complete_source_in_both_unicode_normalizations():
    converter = Tokenizer(str(PROFILE))
    def convert(text):
        return unicodedata.normalize("NFC", converter(text, column="IPA").replace(" ", "").replace("#", " "))
    for row in csv.reader(CSV.open()):
        assert "�" not in convert(row[2]), row[10]
        assert convert(row[2]) == convert(unicodedata.normalize("NFD", row[2])), row[10]
    assert convert("dʒɑnʋər") == "janvər"
    assert convert("tʃʰɑti") == "cʰati"
    assert convert("ʃᵊkɑr") == "śᵊkar"
    assert convert("lu:ɳ") == "lu:ṇ"


def test_installed_rows_parse_with_registered_metadata():
    import make_cldf
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-kumari-gaddi")
    raw = {r[10]: r for r in csv.reader(CSV.open())}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 392
    assert {row.entry_key for row in rows} == set(raw)
    assert all(row.old_form == raw[row.entry_key][2] and row.gloss == raw[row.entry_key][3] for row in rows)
    assert all(row.lang == "ga" and row.form and "�" not in row.form for row in rows)


def test_sample_audit_has_twenty_print_comparisons():
    sample = json.loads((PACKAGE / "sample-audit-2026092501.json").read_text())
    assert sample["population"] == 254 and sample["sample_size"] == len(sample["rows"]) == 20
    assert sample["material_errors"] == 0
    assert {row["pdf_page"] for row in sample["rows"]} == set(range(136, 142))
    assert all(row["status"] == "matches-printed-source" for row in sample["rows"])


def test_complete_numeral_scope_and_preserved_source_anomalies():
    from collections import Counter
    rows = {r[10]:r for r in csv.reader(CSV.open())}
    audit = [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]
    appendix = [r for r in audit if r.get("table")]
    assert Counter(r["table"] for r in appendix) == {"C.1":105, "C.2":22, "C.3":7, "C.4":5}
    assert len([k for k in rows if ":B1:" in k]) == 254
    assert rows["kumari-gaddi:C1:45"][2] == "pɛ̃tɑli"
    assert rows["kumari-gaddi:C1:65"][2] == "pɛ̃ʈʰ"
    assert rows["kumari-gaddi:C2:17"][2] == "sətɑrma"
    assert rows["kumari-gaddi:C3:1"][2:4] == ["pɔ:ni", "quarter"]
    assert rows["kumari-gaddi:C3:5"][2] == "trijɑ coutʰɑ"
    assert rows["kumari-gaddi:C3:6"][2:4] == ["əkʰ bəʈɑ solɑ", "tenth"]
    assert rows["kumari-gaddi:C3:7"][2:4] == ["əkʰ bəʈɑ dəs", "sixteenth"]
    assert all("uncertain" in rows[f"kumari-gaddi:C3:{i}"][14] for i in (1,5,6,7))
    assert [rows[f"kumari-gaddi:C4:{i}"][2:4] for i in range(1, 6)] == [
        ["ək ək", "one each"], ["trɛ trɛ", "three each"],
        ["pəndʒ pəndʒ", "five each"], ["ək ək kəri kəre", "one by one"],
        ["do do kəri kəre", "two by two"],
    ]
    assert all("Kumari Mamta" in r["collector"] for r in appendix)
    assert all("num" in r["tags"].split() for r in appendix)
    assert all(r["tags"] == "num ord" for r in appendix if r["table"] == "C.2")


def test_independent_appendix_audit_pins_current_install():
    import hashlib
    sample = json.loads((PACKAGE / "independent-appendix-audit-20260926-pass2.json").read_text())
    assert len(sample["entries"]) == 20
    assert sample["material_errors"] == 0
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == sample["hashes"]["full-staged.csv"]
    assert hashlib.sha256((PACKAGE / "audit.jsonl").read_bytes()).hexdigest() == sample["hashes"]["full-staged-audit.jsonl"]
