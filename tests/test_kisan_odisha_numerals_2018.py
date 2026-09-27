"""Focused source-local Kisan (Odisha) table checks; no full database build."""

import csv
import importlib.util
import io
import json
import random
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/kisan_odisha_numerals_2018"
STAGED = PACKAGE / "staged_forms.csv"
CSV = DATA / "data/other/forms/20260925-kisan-odisha-numerals.csv"
PROFILE = DATA / "conversion/kujur-perumalsamy-kisan-2018.txt"
spec = importlib.util.spec_from_file_location("kisan_odisha", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def forms():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audit():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_archived_page_counts_and_regeneration():
    rows, records = source.generate()
    assert rows == forms() and records == audit()
    assert rows == list(csv.reader(STAGED.open(encoding="utf-8", newline="")))
    assert len(rows) == 73 and len(records) == 82
    assert Counter((r["table"], r["status"]) for r in records) == {
        ("Kujur", "ingested"): 41,
        ("Kujur", "held_ambiguous_prompt"): 1,
        ("Perumalsamy", "ingested"): 27,
        ("Perumalsamy", "held_ambiguous_prompt"): 1,
        ("Perumalsamy", "source_blank"): 12,
    }
    assert len({row[10] for row in rows}) == 73
    assert all(len(row) == 15 for row in rows)


def test_repeated_four_and_blank_cells_are_not_silently_repaired():
    records = audit()
    assert {r["number"] for r in records if r["status"] == "source_blank"} == source.BLANK_SECOND
    held = [r for r in records if r["status"] == "held_ambiguous_prompt"]
    assert len(held) == 2 and {r["table"] for r in held} == {"Kujur", "Perumalsamy"}
    assert all(r["number"] == 4 and r["table_row"] == 5 and not r["entry_keys"] for r in held)
    assert all(r["number"] != 5 for r in records)
    assert all(r[3] != "five" for r in forms())
    assert records[0]["number"] == 1 and records[-1]["number"] == 2000


def test_two_tables_multiple_answers_and_source_claims():
    rows = forms()
    assert {row[0] for row in rows} == {"Kurux"}
    assert all(row[14].endswith("dialect:Kurux:kisan-odisha:Kisan") for row in rows)
    assert all(row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == "" for row in rows)
    assert sum("loanword" in row[14] for row in rows) == 65
    assert sum("uncertain" in row[14] for row in rows) == 7
    assert sum(row[10].endswith(":answer:1") for row in rows) == 5
    assert sum(row[10].endswith(":answer:2") for row in rows) == 5
    assert len([r for r in rows if ":kujur:" in r[10]]) == 46
    assert len([r for r in rows if ":perumalsamy:" in r[10]]) == 27
    assert all(row[7].startswith("kujur-perumalsamy2018kisan[") for row in rows)


def test_installed_profile_covers_every_attested_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in forms():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    assert tokenizer("ca:lisʈa", column="IPA").replace(" ", "") == "ʦālisṭa"
    assert tokenizer("muːnʈa", column="IPA").replace(" ", "") == "mūnṭa"
    assert tokenizer("potʃi", column="IPA").replace(" ", "") == "poci"
    assert any(row[5].startswith("t͡s") for row in forms())  # bracketed IPA is preserved


def test_registry_bibliography_and_scoped_parser():
    import make_cldf

    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Kurux"][2] == "kuru1301"
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    dialect = dialects["kisan-odisha"]
    assert dialect[1:6] == ["dialect:Kurux:kisan-odisha:Kisan", "Kurux", "Kisan", "Kisan", "kisa1261"]
    assert dialect[6:8] == ["", ""] and dialect[8] == "N. Dravidian"
    assert languages["KisanIA"][2] != "kisa1261"
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{kujur-perumalsamy2018kisan,") == 1
    assert "20260925-kisan-odisha-numerals.csv" in bib
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-kisan-odisha-numerals")
    assert not errors.getvalue() and len(parsed) == stats["converted"] == 73
    assert all(r.lang == "Kurux" and r.param == "" and r.form and "�" not in r.form for r in parsed)


def test_seeded_twenty_cell_review():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 68
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    selected = random.Random(sample["seed"]).sample([r for r in audit() if r["status"] == "ingested"], 20)
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in selected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])


def test_contributor_attestations_keep_distinct_identity_after_compilation():
    from make_cldf import Row, source_entry_dedupe_key
    from source_meta import SourceMeta
    meta = SourceMeta()
    candidates = [r for r in forms() if r[3] == "twelve"]
    assert len(candidates) == 2
    assert {r[2] for r in candidates} == {"baːro:ʈa", "ba:ro:ʈa"}
    keys = []
    for raw in candidates:
        row = Row(raw, "test")
        row.input_file = CSV.name
        keys.append(source_entry_dedupe_key(row, meta))
    assert len(set(keys)) == 2 and all(keys)
    assert any(":kujur:" in key for key in keys)
    assert any(":perumalsamy:" in key for key in keys)
