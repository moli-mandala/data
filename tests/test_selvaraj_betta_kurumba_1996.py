"""Focused checks for Selvaraj's archived 1996 Betta Kurumba table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/selvaraj_betta_kurumba_1996"
CSV = DATA / "data/other/forms/20260925-selvaraj-betta-kurumba-numerals.csv"
PROFILE = DATA / "conversion/selvaraj-betta-kurumba-1996.txt"
spec = importlib.util.spec_from_file_location("selvaraj_betta", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_both_tables_audited_and_regenerate():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 80 and len(rows) == 40
    assert Counter(a["status"] for a in audit) == {"ingested": 40, "excluded_control": 40}
    assert Counter(a["table_index"] for a in audit) == {1: 40, 7: 40}
    assert [a["number"] for a in audit if a["status"] == "ingested"] == source.NUMBERS
    assert len({r[10] for r in rows}) == 40


def test_source_boundaries_and_control_exclusion():
    target = {a["number"]: a for a in audited() if a["status"] == "ingested"}
    control = [a for a in audited() if a["status"] == "excluded_control"]
    assert target[1]["source_form"] == "oɳɖə"
    assert target[2000]["source_form"] == "əɖsaʋɾe"
    assert (target[1]["table_row"], target[1]["table_column"]) == (1, 1)
    assert (target[2000]["table_row"], target[2000]["table_column"]) == (20, 2)
    assert len({(a["table_row"], a["table_column"]) for a in control}) == 40
    assert {a["number"] for a in control} == set(source.NUMBERS)
    assert all(a["entry_key"] == "" and a["control_cell"] for a in control)
    assert Counter(a["control_markup_status"] for a in control) == {"well_delimited": 38, "malformed_or_complex": 2}
    assert next(a for a in control if a["number"] == 1)["control_phonemic"] == "ʋan(rə)"
    assert next(a for a in control if a["number"] == 1)["control_phonetic"] == "ʋan(də) ~ won(də)"
    assert "/ərrnuːru/" not in target[200]["source_form"]
    assert "ərrnuːru/" in next(a["control_cell"] for a in control if a["number"] == 200)


def test_existing_canonical_and_reference():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["BettaKurumba"][2] == "bett1235"
    for row in installed():
        assert len(row) == 15 and row[0] == "BettaKurumba" and row[2] == row[5]
        assert row[2] and "�" not in row[2] and row[14] == "num"
        assert row[7].startswith("selvaraj1996bettakurumba[Selvaraj Betta Kurumba table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{selvaraj1996bettakurumba,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("oɳɖə", column="IPA").replace(" ", "") == "oṇḍa"
    assert tokenizer("əːɭu", column="IPA").replace(" ", "") == "āḷu"
    assert tokenizer("saʋɾe", column="IPA").replace(" ", "") == "savre"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 40
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "BettaKurumba" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(
        [a for a in audited() if a["status"] == "ingested"], 20
    ), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "selvaraj-betta-kurumba-1996" not in profile_policy.audit(profile_policy.source_inventory())
