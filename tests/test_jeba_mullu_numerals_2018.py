"""Focused Mullu Kurumba numeral checks; no full data or browser database build."""

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
PACKAGE = DATA / "data/other/forms/raw_data/jeba_mullu_numerals_2018"
CSV = DATA / "data/other/forms/20260925-jeba-mullu-numerals.csv"
PROFILE = DATA / "conversion/jeba-mullu-numerals-2018.txt"
spec = importlib.util.spec_from_file_location("jeba_mullu", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_table_count_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 42 and len(rows) == 42
    assert Counter(a["status"] for a in audit) == {"ingested": 42}
    assert len({r[10] for r in rows}) == 42
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40


def test_boundary_double_cells_and_source_typography():
    by_number = {a["number"]: a for a in audited()}
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)
    assert by_number[100]["raw_cell"] == by_number[200]["raw_cell"]
    assert by_number[400]["raw_cell"] == by_number[800]["raw_cell"]
    assert by_number[20]["printed_label"] == "20 "
    assert by_number[400]["printed_label"] == "400"
    assert by_number[2000]["printed_label"] == "2 000"
    assert by_number[26]["source_form"] == by_number[26]["parsed_form"] == "iɾʋɐt̪ t̪ɐːɾɨ"
    assert by_number[50]["arithmetic_note"] == ""
    assert by_number[2000]["source_form"] == "ɾɐɳɖɐːjiɾõ"


def test_canonical_registry_reference_and_unlinked_rows():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    language = languages["Mullu Kurumba"]
    assert language[2] == "mull1244" and language[3:5] == ["", ""] and language[5] == "S. Dravidian I"
    assert "Kerala" in language[6] and "Attappady" in language[6]
    for row in installed():
        assert len(row) == 15 and row[0] == "Mullu Kurumba" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("jeba2018mullu[Mullu Kurumba table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{jeba2018mullu,") == 1
    assert "20260925-jeba-mullu-numerals.csv" in bib


def test_profile_and_parser_preserve_distinct_layers():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        converted = tokenizer(row[2], column="IPA")
        assert "�" not in converted, row[10]
    assert tokenizer("oɳɳɨ", column="IPA").replace(" ", "") == "oṇṇɨ"
    assert tokenizer("iɾʋɐt̪ t̪ɐːɾɨ", column="IPA").replace(" ", "").replace("#", " ") == "irvɐt̪ t̪ɐ̄rɨ"
    assert tokenizer("ɐːjiɾõ", column="IPA").replace(" ", "") == "ɐ̄yirõ"
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-jeba-mullu-numerals")
    original = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 42
    assert {r.entry_key for r in rows} == set(original)
    assert all(r.old_form == original[r.entry_key][2] and r.form and "�" not in r.form for r in rows)
    assert all(r.lang == "Mullu Kurumba" and r.param == "" for r in rows)


def test_seeded_20_item_audit_is_reproducible():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    rng = random.Random(sample["seed"])
    expected = sorted(rng.sample([a for a in audited() if a["status"] == "ingested"], 20), key=lambda r: r["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in expected]
