"""Focused checks for the archived 2013 Eastern Muria numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/riezen_eastern_muria_2013"
CSV = DATA / "data/other/forms/20260925-riezen-eastern-muria-numerals.csv"
PROFILE = DATA / "conversion/riezen-eastern-muria-2013.txt"
spec = importlib.util.spec_from_file_location("riezen_muria", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_source_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 40 and len(rows) == 39
    assert Counter(a["status"] for a in audit) == {"ingested": 39, "held": 1}
    assert sorted(a["number"] for a in audit) == source.NUMBERS
    assert len({r[10] for r in rows}) == 39


def test_local_hindi_scope_and_held_bracket():
    by_number = {a["number"]: a for a in audited()}
    assert by_number[29]["raw_cell"] == "29. ʊnt̪is]"
    assert by_number[29]["status"] == "held" and not by_number[29]["entry_key"]
    assert {a["number"] for a in audited() if a["local_hindi"]} == {7, 21}
    assert {int(r[10].split(":")[-1]) for r in installed() if "loanword" in r[14]} == {7, 21}
    assert by_number[40]["source_comment"] == ["( 2 x 20 )"]
    assert by_number[1]["table_row"] == 1 and by_number[1]["table_column"] == 1
    assert by_number[2000]["table_row"] == 20 and by_number[2000]["table_column"] == 2


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Eastern Muria"][2] == "east2340"
    assert not languages["Eastern Muria"][3] and not languages["Eastern Muria"][4]
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    assert dialects["muria"][2] == "Gondi" and dialects["muria"][5] == "muri1262"
    for row in installed():
        assert len(row) == 15 and row[0] == "Eastern Muria" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("riezen2013easternmuria[Eastern Muria table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{riezen2013easternmuria,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("t͡ʃʰʌbːis", column="IPA").replace(" ", "") == "cʰabbis"
    assert tokenizer("ʌʈʰais", column="IPA").replace(" ", "") == "aṭʰais"
    assert tokenizer("t̪eːra", column="IPA").replace(" ", "") == "t̪ēra"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 39
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Eastern Muria" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "riezen-eastern-muria-2013" not in profile_policy.audit(profile_policy.source_inventory())
