"""Focused checks for Vijayan's archived 2018 Eravallan numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/vijayan_eravallan_2018"
CSV = DATA / "data/other/forms/20260925-vijayan-eravallan-numerals.csv"
PROFILE = DATA / "conversion/vijayan-eravallan-2018.txt"
spec = importlib.util.spec_from_file_location("vijayan_eravallan", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_two_tables_audited_and_regenerate():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 89 and len(rows) == 42
    assert Counter(a["status"] for a in audit) == {"ingested": 42, "excluded_control": 47}
    assert Counter(a["table_index"] for a in audit) == {1: 42, 5: 47}
    assert [a["number"] for a in audit if a["status"] == "ingested"] == source.NUMBERS
    assert len({r[10] for r in rows}) == 42


def test_paired_cell_and_control_boundaries():
    target = {a["number"]: a for a in audited() if a["status"] == "ingested"}
    controls = [a for a in audited() if a["status"] == "excluded_control"]
    assert target[100]["joined_cell"] == target[200]["joined_cell"]
    assert target[400]["joined_cell"] == target[800]["joined_cell"]
    assert target[100]["source_form"] == "nuːɾɨ"
    assert target[2000]["source_form"] == "rendaːi̯rɔ̃"
    assert (target[1]["table_row"], target[1]["table_column"]) == (1, 1)
    assert (target[2000]["table_row"], target[2000]["table_column"]) == (20, 2)
    assert len({(a["table_row"], a["table_column"]) for a in controls}) == 40
    assert {a["number"] for a in controls} == set(source.CONTROL_NUMBERS)
    assert all(a["entry_key"] == "" and a["source_form"] for a in controls)


def test_existing_canonical_and_reference():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Eravallan"][2] == "erav1242"
    for row in installed():
        assert len(row) == 15 and row[0] == "Eravallan" and row[2] == row[5]
        assert row[2] and "�" not in row[2] and row[14] == "num"
        assert row[7].startswith("vijayan2018eravallan[Vijayan Eravallan table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{vijayan2018eravallan,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("muːndɨ", column="IPA").replace(" ", "") == "mūndɨ"
    assert tokenizer("ʌɲɟɨ", column="IPA").replace(" ", "") == "añjɨ"
    assert tokenizer("i̯iɾɨvʌt̪t̪ǒndɨ", column="IPA").replace(" ", "") == "i̯irɨvat̪t̪ǒndɨ"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 42
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Eravallan" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_audit_and_full_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(
        [a for a in audited() if a["status"] == "ingested"], 20
    ), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "vijayan-eravallan-2018" not in profile_policy.audit(profile_policy.source_inventory())
