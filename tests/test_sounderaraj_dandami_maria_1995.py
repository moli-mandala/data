"""Focused checks for the archived 1995 Bison-Horn Madiya numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/sounderaraj_dandami_maria_1995"
CSV = DATA / "data/other/forms/20260925-sounderaraj-dandami-maria-numerals.csv"
PROFILE = DATA / "conversion/sounderaraj-dandami-maria-1995.txt"
spec = importlib.util.spec_from_file_location("sounderaraj_dandami", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_source_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 40 and len(rows) == 42
    assert Counter(a["status"] for a in audit) == {"ingested": 40}
    assert sorted(a["number"] for a in audit) == source.NUMBERS
    assert len({r[10] for r in rows}) == 42


def test_slash_answers_and_arithmetic_annotations():
    by_number = {a["number"]: a for a in audited()}
    assert by_number[20]["raw_cell"] == "20. koːɽi / biːs"
    assert by_number[20]["source_forms"] == ["koːɽi", "biːs"]
    assert by_number[100]["source_forms"] == ["eyŋg koːɽi", "sʌv"]
    assert by_number[100]["source_comment"] == ["( 5 x 20 )"]
    assert by_number[50]["source_comment"] == ["( 2 x 20+ 10 )"]
    assert by_number[1]["table_row"] == 1 and by_number[1]["table_column"] == 1
    assert by_number[2000]["table_row"] == 20 and by_number[2000]["table_column"] == 2
    assert all(not r[11] and not r[12] and not r[13] for r in installed())


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Dandami Maria"][2] == "dand1238"
    assert not languages["Dandami Maria"][3] and not languages["Dandami Maria"][4]
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    for dialect in ("beine_bhb", "beine_bhm", "beine_bhs"):
        assert dialects[dialect][2] == "Gondi"
    for row in installed():
        assert len(row) == 15 and row[0] == "Dandami Maria" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("sounderaraj1995dandamimaria[Bison-Horn Madiya table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
        assert row[14] == "num"
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{sounderaraj1995dandamimaria,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("ɾeːɳɖ", column="IPA").replace(" ", "") == "rēṇḍ"
    assert tokenizer("ʌd͡ʒaɾ", column="IPA").replace(" ", "") == "ajar"
    assert tokenizer("koːɽi", column="IPA").replace(" ", "") == "kōṛi"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 42
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Dandami Maria" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "sounderaraj-dandami-maria-1995" not in profile_policy.audit(profile_policy.source_inventory())
