"""Focused checks for Penny's archived Adilabad Gondi numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/penny_adilabad_gondi_2017"
CSV = DATA / "data/other/forms/20260925-penny-adilabad-gondi-numerals.csv"
PROFILE = DATA / "conversion/penny-adilabad-gondi-2017.txt"
spec = importlib.util.spec_from_file_location("penny_adilabad", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_cell_answer_and_loan_inventory():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 40 and len(rows) == 50
    assert [a["number"] for a in audit] == source.NUMBERS
    assert Counter(len(a["answers"]) for a in audit) == {1: 32, 2: 6, 3: 2}
    assert sum("loanword" in r[14] for r in rows) == 8
    assert len({r[10] for r in rows}) == 50
    assert all(a["status"] == "ingested" for a in audit)


def test_alternatives_and_borrowing_scope():
    by_number = {a["number"]: a for a in audited()}
    assert [(a["source_form"], a["source_qualifier"]) for a in by_number[9]["answers"]] == [
        ("nov", "Indic loan"), ("piʈ", "native")
    ]
    assert [(a["source_form"], a["source_qualifier"]) for a in by_number[10]["answers"]] == [
        ("pad", "native"), ("daha", "Indic loan")
    ]
    assert [a["source_form"] for a in by_number[1000]["answers"]] == ["badra", "vey", "had͡ʒar"]
    assert [a["source_form"] for a in by_number[2000]["answers"]] == [
        "raɳɖ badraŋɡ", "raɳɖ veyk", "raɳɖ hadʒark"
    ]
    assert all(not a["source_qualifier"] for a in by_number[2000]["answers"])
    assert by_number[20]["answers"][1]["source_form"] == "vi:sa"
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)


def test_precise_new_canonical_preserves_historical_generic_dialect():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    target = languages["Adilabad Gondi"]
    assert target[2] == "utno1237" and target[3:5] == ["", ""]
    assert target[5] == "S. Dravidian II" and "historical" in target[6].lower()
    assert languages["Gondi"][2] == "gond1265"
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    assert dialects["adil"][2] == "Gondi" and dialects["adil"][5] == "gond1265"
    for row in installed():
        assert len(row) == 15 and row[0] == "Adilabad Gondi" and "num" in row[14]
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("penny2017adilabadgondi[Western Southern Gondi table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{penny2017adilabadgondi,") == 1
    assert "20260925-penny-adilabad-gondi-numerals.csv" in bib


def test_profile_coverage_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("vi:sa", column="IPA").replace(" ", "") == "vīsa"
    assert tokenizer("d͡ʒure", column="IPA").replace(" ", "") == "jure"
    assert tokenizer("hadʒark", column="IPA").replace(" ", "") == "hajark"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 50
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Adilabad Gondi" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_sample_and_full_inventory_profile():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda x: x["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "penny-adilabad-gondi-2017" not in profile_policy.audit(profile_policy.source_inventory())
