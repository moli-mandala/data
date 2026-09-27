"""Focused checks for Babu's archived Malasar numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/babu_malasar_2018"
CSV = DATA / "data/other/forms/20260925-babu-malasar-numerals.csv"
PROFILE = DATA / "conversion/babu-malasar-2018.txt"
spec = importlib.util.spec_from_file_location("babu_malasar", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_table_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == len(rows) == 41
    assert [a["number"] for a in audit] == source.NUMBERS
    assert len({(a["table_row"], a["table_column"]) for a in audit}) == 40
    assert len({r[10] for r in rows}) == 41


def test_paired_cell_and_source_boundaries():
    cells = {a["number"]: a for a in audited()}
    assert cells[400]["joined_cell"] == cells[800]["joined_cell"]
    assert (cells[400]["table_row"], cells[400]["table_column"]) == (18, 2)
    assert cells[400]["source_form"] == "n̠aːɳuːrə"
    assert cells[800]["source_form"] == "eɳuːrə"
    assert (cells[1]["table_row"], cells[1]["table_column"]) == (1, 1)
    assert (cells[2000]["table_row"], cells[2000]["table_column"]) == (20, 2)
    assert 200 not in cells


def test_existing_canonical_and_reference():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Malasar"][2] == "mala1458"
    assert languages["MalaMalasar"][2] == "mala1457"
    for row in installed():
        assert len(row) == 15 and row[0] == "Malasar" and row[2] == row[5]
        assert row[2] and "�" not in row[2] and row[14] == "num"
        assert row[7].startswith("babu2018malasar[Malasar table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{babu2018malasar,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("muːɳə", column="IPA").replace(" ", "") == "mūṇa"
    assert tokenizer("and͡ʒə", column="IPA").replace(" ", "") == "anja"
    assert tokenizer("n̠aːɳuːrə", column="IPA").replace(" ", "") == "n̠āṇūra"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 41
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Malasar" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 41
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "babu-malasar-2018" not in profile_policy.audit(profile_policy.source_inventory())
