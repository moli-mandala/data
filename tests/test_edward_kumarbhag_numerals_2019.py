"""Focused Kumarbhag Paharia numeral checks; no full database build."""

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
PACKAGE = DATA / "data/other/forms/raw_data/edward_kumarbhag_numerals_2019"
CSV = DATA / "data/other/forms/20260925-edward-kumarbhag-numerals.csv"
PROFILE = DATA / "conversion/edward-kumarbhag-numerals-2019.txt"
spec = importlib.util.spec_from_file_location("edward_kumarbhag", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_table_count_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 42 and len(rows) == 43
    assert Counter(a["status"] for a in audit) == {"ingested": 42}
    assert len({r[10] for r in rows}) == 43
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40


def test_boundary_double_cells_and_transcription_notes():
    by_number = {a["number"]: a for a in audited()}
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)
    assert by_number[100]["raw_cell"] == by_number[200]["raw_cell"]
    assert by_number[400]["raw_cell"] == by_number[800]["raw_cell"]
    assert by_number[400]["printed_label"] == "4 00"
    assert by_number[5]["parsed_forms"] == ["pa:t͡ʃ"]
    assert by_number[5]["printed_parentheses"] == ["(paːc)"]
    assert by_number[30]["parsed_forms"] == ["ɖeɖ koːɽi"]
    assert by_number[30]["printed_parentheses"] == ["( ½ x 20)", "(DeːD koːRi)"]
    assert by_number[20]["parsed_forms"] == ["biːs", "koːɽjoːnd"]
    assert by_number[20]["entry_keys"] == [
        "edward2019kumarbhag:number:20:reading:1",
        "edward2019kumarbhag:number:20:reading:2",
    ]


def test_canonical_registry_reference_and_unlinked_rows():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    language = languages["Kumarbhag Paharia"]
    assert language[2] == "kuma1274" and language[3:5] == ["", ""] and language[5] == "N. Dravidian"
    assert "Bihar and Jharkhand" in language[6] and "saur1249" in language[6]
    for row in installed():
        assert len(row) == 15 and row[0] == "Kumarbhag Paharia" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("edward2019kumarbhag[Kumarbhag Paharia table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{edward2019kumarbhag,") == 1
    assert "20260925-edward-kumarbhag-numerals.csv" in bib


def test_profile_and_parser_preserve_distinct_layers():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    assert tokenizer("pa:t͡ʃ", column="IPA").replace(" ", "") == "pāc"
    assert tokenizer("koːɽjoːnd", column="IPA").replace(" ", "") == "kōṛyōnd"
    assert tokenizer("t͡ʃawd̪aːr", column="IPA").replace(" ", "") == "cavd̪ār"
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-edward-kumarbhag-numerals")
    original = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 43
    assert {r.entry_key for r in rows} == set(original)
    assert all(r.old_form == original[r.entry_key][2] and r.form and "�" not in r.form for r in rows)
    assert all(r.lang == "Kumarbhag Paharia" and r.param == "" for r in rows)


def test_seeded_20_prompt_audit_is_reproducible():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    rng = random.Random(sample["seed"])
    expected = sorted(rng.sample(audited(), 20), key=lambda r: r["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in expected]
