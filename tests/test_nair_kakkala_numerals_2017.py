"""Focused Kakkala numeral checks; no full database build."""

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
PACKAGE = DATA / "data/other/forms/raw_data/nair_kakkala_numerals_2017"
CSV = DATA / "data/other/forms/20260925-nair-kakkala-numerals.csv"
PROFILE = DATA / "conversion/nair-kakkala-numerals-2017.txt"
spec = importlib.util.spec_from_file_location("nair_kakkala", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_table_count_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 40 and len(rows) == 43
    assert Counter(a["status"] for a in audit) == {"ingested": 40}
    assert len({r[10] for r in rows}) == 43
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40


def test_first_last_multiple_readings_and_excluded_paradigm():
    by_number = {a["number"]: a for a in audited()}
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)
    assert by_number[1000]["printed_label"] == "1000 "
    assert by_number[2000]["printed_label"] == "2 000"
    assert by_number[10]["parsed_forms"] == ["padumaːji", "paɹumaːji"]
    assert by_number[1000]["parsed_forms"] == ["aːriyam", "aːriyamaːji", "aːri"]
    assert by_number[9]["parsed_forms"] == ["tommaːji"]
    assert by_number[60]["parsed_forms"] == ["aɹupaɹumaːji"]
    assert all(not row[11] for row in installed())


def test_canonical_registry_reference_and_unlinked_rows():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    language = languages["Kakkala"]
    assert language[2] == "kakk1234" and language[3:5] == ["", ""] and language[5] == "S. Dravidian I"
    assert "Kerala" in language[6] and "Kuḷupe:ccu" in language[6]
    for row in installed():
        assert len(row) == 15 and row[0] == "Kakkala" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("nair2017kakkala[Kakkala table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{nair2017kakkala,") == 1
    assert "20260925-nair-kakkala-numerals.csv" in bib


def test_profile_and_parser_preserve_distinct_layers():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    assert tokenizer("oruma:ji", column="IPA").replace(" ", "") == "orumāji"
    assert tokenizer("irupaɹumaːji", column="IPA").replace(" ", "") == "iruparumāji"
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-nair-kakkala-numerals")
    original = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 43
    assert {r.entry_key for r in rows} == set(original)
    assert all(r.old_form == original[r.entry_key][2] and r.form and "�" not in r.form for r in rows)
    assert all(r.lang == "Kakkala" and r.param == "" for r in rows)


def test_seeded_20_prompt_audit_is_reproducible():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    rng = random.Random(sample["seed"])
    expected = sorted(rng.sample(audited(), 20), key=lambda r: r["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in expected]
