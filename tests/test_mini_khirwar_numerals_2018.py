"""Focused Khirwar numeral checks; no full database build."""

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
PACKAGE = DATA / "data/other/forms/raw_data/mini_khirwar_numerals_2018"
CSV = DATA / "data/other/forms/20260925-mini-khirwar-numerals.csv"
PROFILE = DATA / "conversion/mini-khirwar-numerals-2018.txt"
spec = importlib.util.spec_from_file_location("mini_khirwar", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_table_count_holds_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 42 and len(rows) == 40
    assert Counter(a["status"] for a in audit) == {"ingested": 40, "deferred_transcription": 2}
    assert {a["number"] for a in audit if a["status"] == "deferred_transcription"} == {5, 50}
    assert len({r[10] for r in rows}) == 40
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40


def test_boundary_double_cells_and_source_typography():
    by_number = {a["number"]: a for a in audited()}
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)
    assert by_number[100]["raw_cell"] == by_number[200]["raw_cell"]
    assert by_number[400]["raw_cell"] == by_number[800]["raw_cell"]
    assert by_number[800]["printed_label"] == "8 00"
    assert by_number[2000]["printed_label"] == "2 000"
    assert by_number[5]["raw_cell"] == "5. p ã tʃɡo" and not by_number[5]["parsed_form"]
    assert by_number[50]["raw_cell"] == "50. p ã tʃasɡo" and not by_number[50]["parsed_form"]
    assert all("neighboring Indo-Aryan" in a["borrowing_claim"] for a in audited())


def test_canonical_registry_reference_and_loanword_scope():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    language = languages["Khirwar"]
    assert language[2] == "khir1237" and language[3:5] == ["", ""] and language[5] == "S. Dravidian II"
    assert "Garhwa and Latehar" in language[6] and "Palamu" in language[6]
    for row in installed():
        assert len(row) == 15 and row[0] == "Khirwar" and row[14] == "num loanword"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("mini2018khirwar[Khirwar table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{mini2018khirwar,") == 1
    assert "20260925-mini-khirwar-numerals.csv" in bib


def test_profile_and_parser_preserve_distinct_layers():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    assert tokenizer("tʃʰɑvɡo", column="IPA").replace(" ", "") == "cʰɑvgo"
    assert tokenizer("aʈʰɡo", column="IPA").replace(" ", "") == "aṭʰgo"
    assert tokenizer("tʃɑʊd̪ɐɡo", column="IPA").replace(" ", "") == "cɑud̪ɐgo"
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-mini-khirwar-numerals")
    original = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 40
    assert {r.entry_key for r in rows} == set(original)
    assert all(r.old_form == original[r.entry_key][2] and r.form and "�" not in r.form for r in rows)
    assert all(r.lang == "Khirwar" and r.param == "" for r in rows)


def test_seeded_20_item_audit_is_reproducible():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    rng = random.Random(sample["seed"])
    expected = sorted(rng.sample([a for a in audited() if a["status"] == "ingested"], 20), key=lambda r: r["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in expected]


def test_printed_seventeen_and_seventy_homographs_keep_distinct_identity():
    from make_cldf import Row, source_entry_dedupe_key
    from source_meta import SourceMeta
    meta = SourceMeta()
    candidates = [r for r in installed() if r[10] in {
        "mini2018khirwar:number:17", "mini2018khirwar:number:70"}]
    assert len(candidates) == 2
    assert {r[2] for r in candidates} == {"sɑt̪rahɡo"}
    assert {r[3] for r in candidates} == {"seventeen", "seventy"}
    keys = []
    for raw in candidates:
        row = Row(raw, "test")
        row.input_file = CSV.name
        keys.append(source_entry_dedupe_key(row, meta))
    assert len(set(keys)) == 2 and all(keys)
