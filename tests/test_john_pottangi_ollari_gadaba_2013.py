"""Focused checks for John's archived 2013 Pottangi Ollar Gadaba table."""

import csv
import importlib.util
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/john_pottangi_ollari_gadaba_2013"
CSV = DATA / "data/other/forms/20260925-john-pottangi-ollar-gadaba-numerals.csv"
PROFILE = DATA / "conversion/john-pottangi-ollar-gadaba-2013.txt"
spec = importlib.util.spec_from_file_location("john_ollari", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_source_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 80 and len(rows) == 46
    assert Counter(a["status"] for a in audit) == {"ingested": 40, "excluded_control": 40}
    assert Counter(a["table_index"] for a in audit) == {1: 40, 7: 40}
    assert sum(a["control_blank"] for a in audit) == 11
    assert len({r[10] for r in rows}) == 46


def test_gender_class_loan_scope_and_controls():
    target = {a["number"]: a for a in audited() if a["status"] == "ingested"}
    control = [a for a in audited() if a["status"] == "excluded_control"]
    assert [x["source_form"] for x in target[1]["answers"]] == ["okuʈ", "ukur", "okal"]
    assert [x["tags"] for x in target[1]["answers"]] == ["num", "num m sg", "num f sg"]
    assert [x["tags"] for x in target[2]["answers"]] == ["num", "num m", "num f"]
    assert [x["tags"] for x in target[3]["answers"]] == ["num", "num m", "num f"]
    assert target[4]["answers"][0]["source_form"] == "t͡sari"
    assert target[4]["answers"][0]["source_qualifier"] == "< Oriya"
    assert target[4]["answers"][0]["tags"] == "num loanword"
    assert sum("loanword" in r[14] for r in installed()) == 1
    assert {x["number"] for x in control if x["control_blank"]} == set(range(21, 30)) | {200, 2000}
    assert (target[1]["table_row"], target[1]["table_column"]) == (1, 1)
    assert (target[2000]["table_row"], target[2000]["table_column"]) == (20, 2)


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["OllariGadaba"][2] == "pott1240"
    assert languages["Gadaba"][2] == "mudh1235"
    for row in installed():
        assert len(row) == 15 and row[0] == "OllariGadaba" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("john2013pottangiollargadaba[Pottangi Ollar Gadaba table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{john2013pottangiollargadaba,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("okuʈ", column="IPA").replace(" ", "") == "okuṭ"
    assert tokenizer("d͡ʒoɖek", column="IPA").replace(" ", "") == "joḍek"
    assert tokenizer("t͡sari", column="IPA").replace(" ", "") == "t͡sari"
    assert tokenizer("bajːs", column="IPA").replace(" ", "") == "bayys"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 46
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "OllariGadaba" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(
        [a for a in audited() if a["status"] == "ingested"], 20
    ), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "john-pottangi-ollar-gadaba-2013" not in profile_policy.audit(profile_policy.source_inventory())
