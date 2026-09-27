"""Focused checks for Kapp's archived 1995 Alu Kurumba numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/kapp_alu_kurumba_1995"
CSV = DATA / "data/other/forms/20260925-kapp-alu-kurumba-numerals.csv"
PROFILE = DATA / "conversion/kapp-alu-kurumba-1995.txt"
spec = importlib.util.spec_from_file_location("kapp_alu", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == len(rows) == 40
    assert {a["number"] for a in audit} == set(source.NUMBERS)
    assert {a["status"] for a in audit} == {"ingested"}
    assert len({r[10] for r in rows}) == 40


def test_source_edge_cases_and_readings():
    by_number = {a["number"]: a for a in audited()}
    assert by_number[7]["raw_cell"] == "7. ëːɭu"
    assert by_number[7]["source_form"] == "ëːɭu"
    assert by_number[30]["source_form"] == "moaⁿttu"
    assert by_number[1000]["printed_label"] == "1000 "
    assert by_number[2000]["printed_label"] == "2 000"
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)
    assert all(r[14] == "num" for r in installed())


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["AluKurumba"][2] == "aluk1238"
    for row in installed():
        assert len(row) == 15 and row[0] == "AluKurumba" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("kapp1995alukurumba[Alu Kurumba table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{kapp1995alukurumba,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("ëːɭu", column="IPA").replace(" ", "") == "ē̈ḷu"
    assert tokenizer("moaⁿttu", column="IPA").replace(" ", "") == "moaⁿttu"
    assert tokenizer("raːɖu", column="IPA").replace(" ", "") == "rāḍu"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 40
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "AluKurumba" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "kapp-alu-kurumba-1995" not in profile_policy.audit(profile_policy.source_inventory())
