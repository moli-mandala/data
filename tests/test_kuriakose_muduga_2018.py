"""Focused checks for the archived 2018 Muduga numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/kuriakose_muduga_2018"
CSV = DATA / "data/other/forms/20260925-kuriakose-muduga-numerals.csv"
PROFILE = DATA / "conversion/kuriakose-muduga-2018.txt"
spec = importlib.util.spec_from_file_location("kuriakose_muduga", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_table_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == len(rows) == 42
    assert [a["number"] for a in audit] == source.NUMBERS
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40
    assert len({r[10] for r in rows}) == 42


def test_adjacent_spans_and_paired_labels():
    cells = {a["number"]: a for a in audited()}
    assert cells[15]["source_form"] == "pɑt̠inɑɲd͡ʒʉ"
    assert cells[25]["source_form"] == "ɪɾʉʋat̠ːɪɑɲd͡ʒʉ"
    assert "</span><span" in cells[15]["raw_markup"]
    assert "</span><span" in cells[25]["raw_markup"]
    assert cells[100]["joined_cell"] == cells[200]["joined_cell"]
    assert cells[400]["joined_cell"] == cells[800]["joined_cell"]
    assert (cells[1]["table_row"], cells[1]["table_column"]) == (1, 1)
    assert (cells[2000]["table_row"], cells[2000]["table_column"]) == (20, 2)


def test_existing_canonical_and_reference():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Muduga"][2] == "mudu1239"
    for row in installed():
        assert len(row) == 15 and row[0] == "Muduga" and row[2] == row[5]
        assert row[2] and "�" not in row[2] and row[14] == "num"
        assert row[7].startswith("kuriakose2018muduga[Muduga table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{kuriakose2018muduga,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("pɑt̠inɑɲd͡ʒʉ", column="IPA").replace(" ", "") == "pat̠inañjʉ"
    assert tokenizer("ɪɾʉʋat̠ːɪɑɲd͡ʒʉ", column="IPA").replace(" ", "") == "irʉvat̠t̠iañjʉ"
    assert tokenizer("əɳːurʉ", column="IPA").replace(" ", "") == "aṇṇurʉ"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 42
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Muduga" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_audit_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "kuriakose-muduga-2018" not in profile_policy.audit(profile_policy.source_inventory())
