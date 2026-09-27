"""Focused checks for the archived Kardile Southeastern Kolami table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/kardile_southeastern_kolami_2013"
CSV = DATA / "data/other/forms/20260925-kardile-southeastern-kolami-numerals.csv"
PROFILE = DATA / "conversion/kardile-southeastern-kolami-2013.txt"
spec = importlib.util.spec_from_file_location("kardile_kolami", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_both_tables_reconciled_with_exact_keys():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 80 and len(rows) == 43
    assert Counter(a["status"] for a in audit) == {"ingested": 40, "excluded_control": 40}
    assert Counter(a["table_index"] for a in audit) == {1: 40, 6: 40}
    assert len({r[10] for r in rows}) == 43
    assert {a["number"] for a in audit if a["table_index"] == 1} == set(source.NUMBERS)
    assert {a["number"] for a in audit if a["table_index"] == 6} == set(source.NUMBERS)


def test_boundary_multi_answer_and_comparator():
    audit = audited()
    target = {a["number"]: a for a in audit if a["status"] == "ingested"}
    controls = {a["number"]: a for a in audit if a["status"] == "excluded_control"}
    assert (target[1]["table_row"], target[1]["table_column"]) == (1, 1)
    assert (target[2000]["table_row"], target[2000]["table_column"]) == (20, 2)
    assert target[16]["answers"] == ["sola", "soɖa"]
    assert target[40]["answers"] == ["t͡ʃalis", "t͡ʃaɖis"]
    assert target[200]["answers"] == ["donse", "donʃe"]
    assert controls[3]["source_form_cell"] == "muːndiŋ"
    assert target[3]["source_form_cell"] == "mundiŋ"
    assert all(a["entry_keys"] == [] for a in controls.values())
    assert all(a["uncertainty"] == "source_heading_conflict" for a in target.values())


def test_existing_naikri_canonical_and_citations():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Naikri"][2] == "sout1549"
    assert languages["Kolami"][2] == "nort2699"
    for row in installed():
        assert len(row) == 15 and row[0] == "Naikri" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("kardile2013southeasternkolami[Southeastern Kolami table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{kardile2013southeasternkolami,") == 1
    assert "20260925-kardile-southeastern-kolami-numerals.csv" in bib


def test_profile_full_coverage_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("t͡səwda", column="IPA").replace(" ", "") == "ʦavda"
    assert tokenizer("t͡ʃəvis", column="IPA").replace(" ", "") == "cavis"
    assert tokenizer("həd͡ʒar", column="IPA").replace(" ", "") == "hajar"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 43
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Naikri" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_sample_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = random.Random(sample["seed"]).sample(
        [a for a in audited() if a["status"] == "ingested"], 20
    )
    assert [r["source_cell_key"] for r in sample["rows"]] == [
        a["source_cell_key"] for a in sorted(expected, key=lambda x: x["number"])
    ]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "kardile-southeastern-kolami-2013" not in profile_policy.audit(profile_policy.source_inventory())
