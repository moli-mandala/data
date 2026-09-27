"""Focused checks for the archived 1995 Koya numeral table."""

import csv
import importlib.util
import io
import json
import random
import sys
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/andronov_koya_1995"
CSV = DATA / "data/other/forms/20260925-andronov-koya-numerals.csv"
PROFILE = DATA / "conversion/andronov-koya-1995.txt"
spec = importlib.util.spec_from_file_location("andronov_koya", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_source_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 40 and len(rows) == 58
    assert sum(len(a["answers"]) for a in audit) == 62
    assert sum(x["status"] == "held" for a in audit for x in a["answers"]) == 4
    assert [a["number"] for a in audit] == source.NUMBERS
    assert len({r[10] for r in rows}) == len(rows)
    assert sum("loanword" in r[14] for r in rows) == 8


def test_compact_slash_and_loan_scopes():
    cells = {a["number"]: a for a in audited()}
    assert [x["source_form"] for x in cells[10]["answers"]] == ["padi", "das", "des"]
    assert [x["source_qualifier"] for x in cells[10]["answers"]] == ["", "Indic loan", "Indic loan"]
    assert [x["status"] for x in cells[22]["answers"]] == ["ingested", "held"]
    assert [x["status"] for x in cells[23]["answers"]] == ["ingested", "held"]
    assert [x["status"] for x in cells[200]["answers"]] == ["held", "ingested"]
    assert [x["status"] for x in cells[2000]["answers"]] == ["held", "ingested"]
    assert [x["source_form"] for x in cells[100]["answers"]] == ["nuːru", "eyŋg koːɽek"]
    assert cells[100]["comments"] == ["( 2 x 20)"]
    assert (cells[1]["table_row"], cells[1]["table_column"]) == (1, 1)
    assert (cells[2000]["table_row"], cells[2000]["table_column"]) == (20, 2)


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Koya"][2] == "koya1251"
    assert languages["Koya"][3:5] == ["", ""]
    assert languages["Koya"][5] == "S. Dravidian II"
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    assert dialects["koya"][2] == "Gondi" and dialects["gommu"][2] == "Gondi"
    for row in installed():
        assert len(row) == 15 and row[0] == "Koya" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("andronov1995koya[Koya table, numeral ")
        assert set(row[14].split()) <= {"num", "loanword"}
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{andronov1995koya,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("muːɳɖu", column="IPA").replace(" ", "") == "mūṇḍu"
    assert tokenizer("oroʈu", column="IPA").replace(" ", "") == "oroṭu"
    assert tokenizer("muːɽu", column="IPA").replace(" ", "") == "mūṛu"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 58
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Koya" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_full_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "andronov-koya-1995" not in profile_policy.audit(profile_policy.source_inventory())
